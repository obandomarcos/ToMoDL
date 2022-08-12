"""
Library for loading datasets. 

Current classes:
  
  * ZebrafishDataset

author: obanmarcos
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import re
import torch, torchvision
from tqdm import tqdm
import SimpleITK as sitk
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import pickle
import h5py
import cv2
import math
import scipy.ndimage as ndi

device = torch.device('cuda')

class ZebraDataset:
  '''
  Zebra dataset 
  Params:
    - folder_path (string): full path folder 
  '''
  def __init__(self, folder_path, dataset_folder, experiment_name):

    self.folder_path = pathlib.Path(folder_path)
    self.folder_name = pathlib.PurePath(self.folder_path).name

    self.objective = 10
    self.dataset_folder = dataset_folder
    self.experiment_name = experiment_name
    self.registered_volume = {}
    self.shifts_path = self.folder_path 

    if '4X' in self.folder_name:

      self.objective = 4

    # For loaded dataset
    self.fish_part = {'head': 0,
                     'body': 1,
                     'upper tail': 2, 
                     'lower tail' : 3}
    # For file string
    self.fish_part_code = {'head': 's000',
                        'body': 's001',
                        'upper tail': 's002', 
                        'lower tail' : 's003'}                      

    self.dataset = {}
    self.registered_dataset = None
    self.shifts = {}

    # Get available fish parts
    self.file_list = self._searchAllFiles(self.folder_path)
    self.load_list = [f for f in self.file_list if ('tif' in str(f))]
    self.fish_parts_available = []

    # Check available parts - sample
    for (fish_part_key, fish_part_value) in self.fish_part_code.items():
      
      for s in self.load_list:
        
        if fish_part_value in str(s):
          
          # printprint(fish_part_key)(s, fish_part_value, fish_part_key)
          self.fish_parts_available.append(fish_part_key)
          break
      
  def _searchAllFiles(self, x):

    dirpath = x
    assert(dirpath.is_dir())
    file_list = []
    
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(self._searchAllFiles(x))

    return file_list

  def loaded_images(self, sample = None):
    '''
    Params:
      - sample (string): {None, body, head, tail}
    '''

    sample2idx =  self.fish_part[sample]    
    load_list = [f for f in self.file_list if ('tif' in str(f))]

    if sample is not None:
      # If it's None, then it loads all fish parts/sample
      fish_part = self.fish_part_code[sample]
      load_list = [f for f in load_list if (fish_part in str(f))]
    
      
    loaded_images = []
    loaded_angles = []
    loaded_sample = []
    loaded_channel = []
    loaded_time = []

    angle = re.compile('a\d+')
    sample_re = re.compile('s\d+')
    channel = re.compile('c\d+')
    time = re.compile('t\d+')
    
    for f in tqdm(load_list):
      
      loaded_images.append(np.array(Image.open(f)))
      loaded_angles.append(float(angle.findall(str(f))[0][1:]))
      loaded_sample.append(float(sample_re.findall(str(f))[0][1:]))
      loaded_channel.append(float(channel.findall(str(f))[0][1:]))
      loaded_time.append(float(time.findall(str(f))[0][1:]))

    self.dataset = pd.DataFrame({'Filename':load_list, 
                                 'Image':loaded_images,
                                 'Angle':loaded_angles,
                                 'Sample':loaded_sample,
                                 'Channel':loaded_channel,
                                 'Time':loaded_time})

    # Create Registered Dataset - empty till reggistering
    self.registered_dataset = pd.DataFrame(columns = ['Image', 'Angle', 'Sample'])

    # Sort dataset by sample and angle
    self.dataset = self.dataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)

    self.registered_volume[sample] = np.stack(self.dataset[self.dataset.Sample == sample2idx]['Image'].to_numpy())
    
    self.imageVolume = np.moveaxis(np.stack(self.dataset['Image'].to_numpy()), 1, 2)
    del self.dataset
  
  def correctRotationAxis(self,  max_shift = 200, shift_step = 4, center_shift_top = 0, center_shift_bottom = 0, sample = 'head', load_shifts = False, save_shifts = True):
    '''
    Corrects rotation axis by finding optimal registration via maximising reconstructed image's intensity variance.

    Based on 'Walls, J. R., Sled, J. G., Sharpe, J., & Henkelman, R. M. (2005). Correction of artefacts in optical projection tomography. Physics in Medicine & Biology, 50(19), 4645.'

    
    '''

    if load_shifts == True:
      
      with open(str(self.shifts_path)+"_{}".format(sample)+".pickle", 'rb') as f:
        
        self.shifts[sample] = pickle.load(f)
    
    else:
    
      # Grab top and bottom sinograms (automate to grab non-empty sinograms)
      top_index, bottom_index = self._grabImageIndexes()

      self.top_sino = np.copy(self.registered_volume[sample][:,:,top_index].T)
      self.bottom_sino = np.copy(self.registered_volume[sample][:,:,bottom_index].T)
      self.angles = np.linspace(0, 2*180, self.top_sino.shape[1] ,endpoint = False)

      # Iteratively sweep from -maxShift pixels to maxShift pixels
      (top_shift_max, bottom_shift_max) = self._searchShifts(max_shift, shift_step, center_shift_top, center_shift_bottom)

      # Interpolation 
      # (top_shift_max, bottom_shift_max) = (abs(top_shift_max), abs(bottom_shift_max))
      m = (top_shift_max-bottom_shift_max)/(top_index-bottom_index)
      b = top_shift_max-m*top_index
      self.shifts[sample] = (m*np.arange(0, self.registered_volume[sample].shape[2]-1)+b).astype(int)

    if save_shifts == True:
      
      with open(str(self.shifts_path)+"_{}".format(sample)+".pickle", 'wb') as f:

        pickle.dump(self.shifts, f)

    # Create Registered volume[sample] with the shifts
    self._registerVolume(sample)

  def _registerVolume(self, sample):
    """
    Register volume with interpolated shifts
    """
    assert(self.shifts is not None)

    # Shift according to shifts
    for idx, shift in enumerate(self.shifts[sample]):
      
      self.registered_volume[sample][:,:,idx] = ndi.shift(self.registered_volume[sample][:,:,idx], (0, shift), mode = 'nearest')

  def datasetResize(self, sample, img_resize, number_projections):
    """
    Resizes sinograms according to reconstruction image size
    """
    # Move axis to (N_projections, n_detector, n_slices)
    self.registered_volume[sample] = np.rollaxis(self.registered_volume[sample], 2)
    # Resize projection number % 16

    det_count = int((img_resize+0.5)*np.sqrt(2))
  
    self.registered_volume[sample] = np.array([cv2.resize(img, (det_count, number_projections)) for img in self.registered_volume[sample]])
    
    self.registered_volume[sample] = np.moveaxis(self.registered_volume[sample], 0,-1)

  def _grabImageIndexes(self, threshold = 50):
    """
    Grabs top and bottom non-empty indexes
    """
    img_max = self.imageVolume.min(axis = 0)
    img_max = (((img_max-img_max.min())/(img_max.max()-img_max.min()))*255.0).astype(np.uint8)
    img_max = ndi.gaussian_filter(img_max,(11,11))

    top_index, bottom_index = (np.where(img_max.std(axis = 0)>threshold)[0][0],np.where(img_max.std(axis = 0)>threshold)[0][-1])
    
    print('Top index:', top_index)
    print('Bottom index:', bottom_index)
    
    return top_index, bottom_index

  def _searchShifts(self, max_shift, shift_step, center_shift_top, center_shift_bottom):

    # Sweep through all shifts
    top_shifts = np.arange(-max_shift, max_shift, shift_step)+center_shift_top
    bottom_shifts = np.arange(-max_shift, max_shift, shift_step)+center_shift_bottom
    
    top_image_std = []
    bottom_image_std = []

    for i, (top_shift, bottom_shift) in enumerate(zip(top_shifts, bottom_shifts)):

      print('Shift {}, top shift {}, bottom shift {}'.format(i, top_shift, bottom_shift))

      top_shift_sino = ndi.shift(self.top_sino, (top_shift, 0), mode = 'nearest')
      bottom_shift_sino = ndi.shift(self.bottom_sino, (bottom_shift, 0), mode = 'nearest')

      # Get image reconstruction
      top_shift_iradon =  iradon(top_shift_sino, self.angles, circle = False)
      bottom_shift_iradon =  iradon(bottom_shift_sino, self.angles, circle = False)
      
      # Calculate variance
      top_image_std.append(np.std(top_shift_iradon))
      bottom_image_std.append(np.std(bottom_shift_iradon))
    
    plt.plot(top_shifts, top_image_std)
    plt.plot(bottom_shifts, bottom_image_std)

    max_shift_top = top_shifts[np.argmax(top_image_std)]
    max_shift_bottom = bottom_shifts[np.argmax(bottom_image_std)]

    return (max_shift_top, max_shift_bottom)

  def getfish_parts(self):

    return self.fish_parts_available

  def registerDataset(self, sample, inPlace = False):

    """
    Registers full dataset, by sample. (deprecated method)
    Params:
      -sample (string): fish part sample.
    """  
    
    # self. = self.dataset[df.dataset.Sample == '000'].sort_values('Angle', axis = 0).reset_index(drop=True)
    if sample is not None:

      sample = self.fish_part[sample]
      dataset = self.dataset[self.dataset.Sample == sample].filter(['Angle','Image'])
    
    else:
      
      dataset = self.dataset.filter(['Angle', 'Image'])

    # Assert angle step {360, 720} -> (1, 0.5)
    if dataset['Angle'].max() in [359, 360]:
      self.maxAngle = 360
    else:
      self.maxAngle = 720
    # {0-179, 0-359}
    #pruebita paso max angle
    angles = np.arange(0, self.maxAngle//2, 1).astype(float)
    # Parámetros de transformación
    self.Tparams = pd.DataFrame(columns = ['theta', 'Tx', 'Ty', 'Sample', 'Angle'])

    # Filtrado Laplaciano mas grayscale
    rglaplacianfilter = sitk.LaplacianRecursiveGaussianImageFilter()
    rglaplacianfilter.SetSigma(6)
    rglaplacianfilter.SetNormalizeAcrossScale(True)

    grayscale_dilate_filter = sitk.GrayscaleDilateImageFilter()
    IsoData = sitk.IsoDataThresholdImageFilter()
    
    # Registration algorithm
    R = sitk.ImageRegistrationMethod()
    
    # Similarity metric, optimizer and interpolator
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=0.1,
                                              numberOfIterations=10)
    R.SetInterpolator(sitk.sitkLinear)
    
    for angle in tqdm(angles):
      
      fixed =  dataset[dataset.Angle == angle].iloc[0]['Image'].astype(float)
      moving = np.flipud(dataset[dataset.Angle == angle+self.maxAngle//2].iloc[0]['Image'].astype(float))
      # pair of images sitk
      fixed_s = sitk.Cast(sitk.GetImageFromArray(fixed), sitk.sitkFloat32)
      moving_s = sitk.Cast(sitk.GetImageFromArray(moving), sitk.sitkFloat32)

      # f stands for filtered image
      fixed_s_f  = rglaplacianfilter.Execute(fixed_s)
      fixed_s_f = grayscale_dilate_filter.Execute(fixed_s)
      fixed_s_f = sitk.BinaryFillhole(IsoData.Execute(fixed_s))

      moving_s_f  = rglaplacianfilter.Execute(moving_s)
      moving_s_f = grayscale_dilate_filter.Execute(moving_s)
      moving_s_f = sitk.BinaryFillhole(IsoData.Execute(moving_s))

      # Initial Transform - Aligns center of mass (same modality - no processing/filtering)
      initialT = sitk.CenteredTransformInitializer(fixed_s, 
                                        moving_s, 
                                        sitk.Euler2DTransform(), 
                                        sitk.CenteredTransformInitializerFilter.MOMENTS)
  
      R.SetInitialTransform(initialT)
      
      fixed_s_f = sitk.Cast(fixed_s_f, sitk.sitkFloat32)
      moving_s_f = sitk.Cast(moving_s_f, sitk.sitkFloat32)

      outTx = R.Execute(fixed_s_f, moving_s_f) # Rotation + traslation
      params = outTx.GetParameters()
      self.Tparams = self.Tparams.append({'theta':params[0],
                                          'Tx':params[1],
                                          'Ty':params[2],
                                          'Sample':sample,
                                          'Angle':angle}, ignore_index=True)      

      # Check rotation

      # If inPlace, registration is applied to all the dataset, translating images
      # to half the value of vertical translation
      if inPlace == True:
        
        F2C_T = sitk.TranslationTransform(2)
        M2C_T = sitk.TranslationTransform(2)

        F2C_T.SetParameters((0, -params[2]/2))  # Fixed image to center
        M2C_T.SetParameters((0, params[2]/2))  # Moving image to center

        fixed_s_T = sitk.Resample(fixed_s, 
                                  F2C_T, 
                                  sitk.sitkLinear, 
                                  0.0,
                                  fixed_s.GetPixelID())

        moving_s_T = sitk.Resample(moving_s, 
                                  M2C_T, 
                                  sitk.sitkLinear, 
                                  0.0,
                                  moving_s.GetPixelID())
        # Append to registered dataset
        self.registered_dataset = self.registered_dataset.append({'Image' : sitk.GetArrayFromImage(fixed_s_T),
                                      'Angle': angle,
                                      'Sample': sample}, ignore_index=True)
        self.registered_dataset = self.registered_dataset.append({'Image' : np.flipud(sitk.GetArrayFromImage(moving_s_T)),
                                      'Angle': angle+self.maxAngle,
                                      'Sample': sample}, ignore_index=True)
    
    # Order by angle
    self.registered_dataset = self.registered_dataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)
  
  def applyRegistration(self):
    """
    Applies mean registration for dataset from registration params
    """

    assert(self.Tparams is not None)
    
    self.registered_dataset = pd.DataFrame(columns = ['Image', 'Angle', 'Sample'])

    for sample in self.dataset.Sample.unique():
      
      print(sample)
      dataset = self.dataset[self.dataset.Sample == sample].filter(['Angle','Image'])

      # Assert angle step {360, 720} -> (1, 0.5)
      if dataset['Angle'].max() in [359, 360]:
        self.maxAngle = 360
      else:
        self.maxAngle = 720

      print(self.maxAngle)
      angles = np.arange(0, self.maxAngle//2, 1).astype(float)
  
      for angle in tqdm(angles):

        fixed =  dataset[dataset.Angle == angle].iloc[0]['Image'].astype(float)
        moving = dataset[dataset.Angle == angle+self.maxAngle//2].iloc[0]['Image'].astype(float)

        fixed_s = sitk.Cast(sitk.GetImageFromArray(fixed), sitk.sitkFloat32)
        moving_s = sitk.Cast(sitk.GetImageFromArray(moving), sitk.sitkFloat32)
        
        # setting moving transform
        transform = sitk.TranslationTransform(2)
        transform.SetParameters((0, -self.meanDisplacement/2))  # Fixed image to center
        
        fixed_s_T = sitk.Resample(fixed_s, 
                                      transform, 
                                      sitk.sitkLinear, 
                                      0.0,
                                      fixed_s.GetPixelID())

        moving_s_T = sitk.Resample(moving_s, 
                                      transform, 
                                      sitk.sitkLinear, 
                                      0.0,
                                      moving_s.GetPixelID())
        # Append to registered dataset
        self.registered_dataset = self.registered_dataset.append({'Image' : sitk.GetArrayFromImage(fixed_s_T),
                                          'Angle': angle,
                                          'Sample': sample}, ignore_index=True)
        self.registered_dataset = self.registered_dataset.append({'Image' : sitk.GetArrayFromImage(moving_s_T),
                                          'Angle': angle+self.maxAngle//2,
                                          'Sample': sample}, ignore_index=True)
      
        self.dataset = self.dataset.drop(self.dataset[self.dataset.Sample == sample].index)
    
    self.registered_dataset = self.registered_dataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)
    del self.dataset
  
  def getregistered_volume(self, sample ,saveDataset = True, margin = 10, useSegmented = False):
    '''
    Returns registered and stacked numpy volume, ordered by angle
    Calculates lower and upper non-zero limits for sinograms, with a safety
    margin given by margin.
    '''
    assert(self.registered_dataset is not None)

    # if self.registered_volume is not None:

      # return self.registered_volume

    # Filter by sample
    self.registered_volume = np.stack(self.registered_dataset[self.registered_dataset.Sample == sample]['Image'].to_numpy())
    self.registeredAngles = np.stack(self.registered_dataset[self.registered_dataset.Sample == sample]['Angle'].to_numpy())
    
    # Calculates non-zero boundary limit for segmenting the volume
    self.upperLimit = self.registered_volume.shape[1]-margin
    self.lowerLimit = margin
    
    # save dataset in HDF5
    if saveDataset == True:
      
      with h5py.File(self.dataset_folder+'/'+'OPTdatasets.hdf5', 'a') as datasets_file:
        
        # If experiment isn't in the current folder, creates experiment
        if self.experiment_name not in datasets_file.keys():

          datasets_file.create_group(self.experiment_name)
        
        # Creates experiment specifics 
        if self.folder_name not in datasets_file[self.experiment_name]:

          datasets_file[self.experiment_name].create_group(self.folder_name)
        
        datasets_file[self.experiment_name][self.folder_name].create_dataset(sample, data = self.registered_volume)
        datasets_file[self.experiment_name][self.folder_name].create_dataset(sample+'_angles', data = self.registeredAngles)
    
    # Normalize volume
    if useSegmented == True:
  
      return self.registered_volume[:, self.lowerLimit:self.upperLimit, :]
    
    else:
    
      return self.registered_volume
  
  def saveRegTransforms(self):
    
    with open(str(self.folder_path)+'transform.pickle', 'wb') as h:
      pickle.dump(self.Tparams,  h)

  def loadRegTransforms(self):

    with open(str(self.folder_path)+'transform.pickle', 'rb') as h:
      self.Tparams = pickle.load(h)
    # save mean displacement for operations
    self.meanDisplacement = self.Tparams['Ty'].mean()
  
  def saveregistered_dataset(self, name = '', mode = 'hdf5'):
    '''
    Saves registered dataset for DL usage (HDF5) or just pickle for binary storage
    params :
    '''

    if mode == 'pickle':

      with open(str(self.folder_path)+name+'.pickle', 'wb') as pickleFile:
      
        pickle.dump({'reg_dataset' : self.registered_dataset,
                    'reg_transform' : self.Tparams}, pickleFile)

    elif mode == 'hdf5':
      
      with pd.HDFStore(self.dataset_folder+'/'+'OPTdatasets.hdf5', 'a') as datasets_file:
        # Take each sample and creates a new dataset
        for sample in self.registered_dataset.Sample.unique():
            
            # Using Pandas built-in HDF5 converter save images
          datasets_file.put(key = self.experiment_name+'/'+self.folder_name+'/'+sample,
                            value = self.registered_dataset[self.registered_dataset.Sample == sample],
                            data_columns = True)
            
          # Metadata includes angles and eventually other parameters
          # print(self.experiment_name+'/'+self.folder_name+'/'+sample+'/'+'values')
          # datasets_file[self.experiment_name+'/'+self.folder_name+'/'+sample+'/'+'values'].attrs['Angle'] = self.registered_dataset[self.registered_dataset.Sample == 'head']['Angle'].to_numpy()

  def loadregistered_dataset(self):

    with open(str(self.folder_path)+pickleName+'.pickle', 'rb') as pickleFile:
      
      reg = pickle.load(pickleFile)
      self.registered_dataset = reg['reg_dataset']
      self.Tparams = reg['reg_transform']

  def deleteSection(self, sample):

    self.registered_dataset = self.registered_dataset.drop(self.registered_dataset[self.registered_dataset.Sample == sample].index)






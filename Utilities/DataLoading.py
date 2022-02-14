"""
16/03
Functions for data loading

# Notas del dataset : 
# 1 - Algunos datasets hacen el recorrido 0-360, aunque otros hacen 0-359. 
# Chequear condición para dado caso
# 
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

class FlatfishDataset:
    
  def __init__(self, folderPath):

    self.folderPath = pathlib.Path(folderPath)
    self.folderName = pathlib.PurePath(self.folderPath).name

    self.shifts_path = self.folderPath 
    self.fileList = self._searchAllFiles(self.folderPath)
    self.loadList = [f for f in self.fileList if ('tif' in str(f))]
    self.imageVolume = None

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
    
  def loadImages(self):
    '''
    Params:
      - sample (string): {None, body, head, tail}
    '''
    
    loadImages = []
    loadAngle = []

    angle = re.compile('theta\d+')
    
    for f in tqdm(self.loadList):
      
      loadImages.append(np.array(Image.open(f)))
      loadAngle.append(float(angle.findall(str(f))[0].strip('theta')))

    self.dataset = pd.DataFrame({'Filename':self.loadList, 
                                 'Image':loadImages,
                                 'Angle':loadAngle
                                 })

    # Create Registered Dataset - empty till reggistering
    self.registeredDataset = pd.DataFrame(columns = ['Image', 'Angle'])

    # Sort dataset by sample and angle
    self.dataset = self.dataset.sort_values(['Angle'], axis = 0).reset_index(drop=True)

    if self.imageVolume is None:
    
      self.imageVolume = np.moveaxis(np.stack(self.dataset['Image'].to_numpy()), 1, 2)

  def correctRotationAxis(self,  max_shift = 200, shift_step = 4, center_shift_top = 0, center_shift_bottom = 0, load_shifts = False, save_shifts = True, sample = 'full'):
    
    if load_shifts == True:
      
      with open(str(self.shifts_path)+"_{}".format(sample)+".pickle", 'rb') as f:
        
        self.shifts = pickle.load(f)
    
    else:
    
      # Grab top and bottom sinograms (automate to grab non-empty sinograms)
      top_index, bottom_index = self._grabImageIndexes(threshold = 30)
      # top_index, bottom_index = (0,self.imageVolume.shape[2]-1)

      self.top_sino = np.copy(self.imageVolume[:,:,top_index].T)
      self.bottom_sino = np.copy(self.imageVolume[:,:,bottom_index].T)
      self.angles = np.linspace(0, 2*180, self.top_sino.shape[1] ,endpoint = False)

      # Iteratively sweep from -maxShift pixels to maxShift pixels
      (top_shift_max, bottom_shift_max) = self._searchShifts(max_shift, shift_step, center_shift_top, center_shift_bottom)

      # Interpolation 
      # (top_shift_max, bottom_shift_max) = (abs(top_shift_max), abs(bottom_shift_max))
      m = (top_shift_max-bottom_shift_max)/(top_index-bottom_index)
      b = top_shift_max-m*top_index
      self.shifts = (m*np.arange(0, self.imageVolume.shape[2]-1)+b).astype(int)

    if save_shifts == True:
      
      with open(str(self.shifts_path)+"_{}".format(sample)+".pickle", 'wb') as f:

        pickle.dump(self.shifts, f)

    # Create Registered volume[sample] with the shifts
    self._registerVolume()

  def _registerVolume(self):
    """
    Register volume with interpolated shifts
    """
    assert(self.shifts is not None)

    self.registeredVolume = np.empty_like(self.imageVolume)

    # Shift according to shifts
    for idx, shift in enumerate(self.shifts):
      
      self.registeredVolume[:,:,idx] = ndi.shift(self.imageVolume[:,:,idx], (0, shift), mode = 'nearest')
    
    self.imageVolume = None
  
  def _grabImageIndexes(self, threshold = 50):
    """
    Grabs top and bottom non-empty indexes
    """
    img_max = self.imageVolume.min(axis = 0)
    img_max = (((img_max-img_max.min())/(img_max.max()-img_max.min()))*255.0).astype(np.uint8)
    img_max = ndi.gaussian_filter(img_max,(11,11))

    plt.plot(img_max.std(axis = 0))
    top_index, bottom_index = (np.where(img_max.std(axis = 0)>threshold)[0][0],np.where(img_max.std(axis = 0)>threshold)[0][-1])
    
    print('Top index:', top_index)
    print('Bottom index:', bottom_index)
    
    return top_index, bottom_index
  
  def _searchShifts(self, max_shift, shift_step, center_shift_top, center_shift_bottom):

    # Sweep through all shifts
    top_shifts = np.arange(-max_shift, max_shift, shift_step)+center_shift_top
    bottom_shifts = np.arange(-max_shift, max_shift, shift_step)+center_shift_bottom
    
    self.top_image_std = []
    self.bottom_image_std = []

    for i, (top_shift, bottom_shift) in enumerate(zip(top_shifts, bottom_shifts)):

      print('Shift {}, top shift {}, bottom shift {}'.format(i, top_shift, bottom_shift))

      top_shift_sino = ndi.shift(self.top_sino, (top_shift, 0), mode = 'nearest')
      bottom_shift_sino = ndi.shift(self.bottom_sino, (bottom_shift, 0), mode = 'nearest')

      # Get image reconstruction
      top_shift_iradon =  iradon(top_shift_sino, self.angles, circle = False)
      bottom_shift_iradon =  iradon(bottom_shift_sino, self.angles, circle = False)
      
      # Calculate variance
      self.top_image_std.append(np.std(top_shift_iradon))
      self.bottom_image_std.append(np.std(bottom_shift_iradon))
    
    plt.plot(top_shifts, top_image_std)
    plt.plot(bottom_shifts, bottom_image_std)

    max_shift_top = top_shifts[np.argmax(top_image_std)]
    max_shift_bottom = bottom_shifts[np.argmax(bottom_image_std)]

    return (max_shift_top, max_shift_bottom)

class ZebraDataset:
  '''
  Zebra dataset 
  Params:
    - folderPath (string): full path folder 
  '''
  def __init__(self, folderPath, datasetsFolder, experimentName):

    self.folderPath = pathlib.Path(folderPath)
    self.folderName = pathlib.PurePath(self.folderPath).name

    self.objective = 10
    self.datasetFolder = datasetsFolder
    self.experimentName = experimentName
    self.registeredVolume = {}
    self.shifts_path = self.folderPath 

    if '4X' in self.folderName:

      self.objective = 4

    # For loaded dataset
    self.fishPart = {'head': 0,
                     'body': 1,
                     'upper tail': 2, 
                     'lower tail' : 3}
    # For file string
    self.fishPartCode = {'head': 's000',
                        'body': 's001',
                        'upper tail': 's002', 
                        'lower tail' : 's003'}                      

    self.dataset = {}
    self.registeredDataset = None
    self.shifts = {}

    # Get available fish parts
    self.fileList = self._searchAllFiles(self.folderPath)
    self.loadList = [f for f in self.fileList if ('tif' in str(f))]
    self.fishPartsAvailable = []

    # Check available parts - sample
    for (fishPart_key, fishPart_value) in self.fishPartCode.items():

      for s in self.loadList:
        
        if fishPart_value in str(s):

          self.fishPartsAvailable.append(fishPart_key)
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

  def loadImages(self, sample = None):
    '''
    Params:
      - sample (string): {None, body, head, tail}
    '''

    sample2idx =  self.fishPart[sample]    
    loadList = [f for f in self.fileList if ('tif' in str(f))]

    if sample is not None:
      # If it's None, then it loads all fish parts/sample
      fishPart = self.fishPartCode[sample]
      loadList = [f for f in loadList if (fishPart in str(f))]
    
      
    loadImages = []
    loadAngle = []
    loadSample = []
    loadChannel = []
    loadTime = []

    angle = re.compile('a\d+')
    sample = re.compile('s\d+')
    channel = re.compile('c\d+')
    time = re.compile('t\d+')
    
    for f in tqdm(loadList):
      
      loadImages.append(np.array(Image.open(f)))
      loadAngle.append(float(angle.findall(str(f))[0][1:]))
      loadSample.append(float(sample.findall(str(f))[0][1:]))
      loadChannel.append(float(channel.findall(str(f))[0][1:]))
      loadTime.append(float(time.findall(str(f))[0][1:]))

    self.dataset = pd.DataFrame({'Filename':loadList, 
                                 'Image':loadImages,
                                 'Angle':loadAngle,
                                 'Sample':loadSample,
                                 'Channel':loadChannel,
                                 'Time':loadTime})

    # Create Registered Dataset - empty till reggistering
    self.registeredDataset = pd.DataFrame(columns = ['Image', 'Angle', 'Sample'])

    # Sort dataset by sample and angle
    self.dataset = self.dataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)

    self.registeredVolume[sample] = np.stack(self.dataset[self.dataset.Sample == sample2idx]['Image'].to_numpy())
    # SUPER UGLY BUT NECESSARY
    del self.dataset
  
  def correctRotationAxis(self,  max_shift = 200, shift_step = 4, center_shift_top = 0, center_shift_bottom = 0, sample = 'head', load_shifts = False, save_shifts = True):
    
    sample2idx =  self.fishPart[sample]

    self.registeredVolume[sample] = np.stack(self.dataset[self.dataset.Sample == sample2idx]['Image'].to_numpy())
    del self.dataset

    if load_shifts == True:
      
      with open(str(self.shifts_path)+"_{}".format(sample)+".pickle", 'rb') as f:
        
        self.shifts[sample] = pickle.load(f)
    
    else:
    
      # Grab top and bottom sinograms (automate to grab non-empty sinograms)
      top_index, bottom_index = self._grabImageIndexes()

      self.top_sino = np.copy(self.registeredVolume[sample][:,:,top_index].T)
      self.bottom_sino = np.copy(self.registeredVolume[sample][:,:,bottom_index].T)
      self.angles = np.linspace(0, 2*180, self.top_sino.shape[1] ,endpoint = False)

      # Iteratively sweep from -maxShift pixels to maxShift pixels
      (top_shift_max, bottom_shift_max) = self._searchShifts(max_shift, shift_step, center_shift_top, center_shift_bottom)

      # Interpolation 
      # (top_shift_max, bottom_shift_max) = (abs(top_shift_max), abs(bottom_shift_max))
      m = (top_shift_max-bottom_shift_max)/(top_index-bottom_index)
      b = top_shift_max-m*top_index
      self.shifts = (m*np.arange(0, self.registeredVolume[sample].shape[2]-1)+b).astype(int)

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
      
      self.registeredVolume[sample][:,:,idx] = ndi.shift(self.registeredVolume[sample][:,:,idx], (0, shift), mode = 'nearest')


  def datasetResize(self, sample, img_resize, number_projections):
    """
    Resizes sinograms according to reconstruction image size
    """
    # Move axis to (N_projections, n_detector, n_slices)
    self.registeredVolume[sample] = np.rollaxis(self.registeredVolume[sample], 2)
    # Resize projection number % 16

    det_count = int((img_resize+0.5)*np.sqrt(2))
  
    self.registeredVolume[sample] = np.array([cv2.resize(img, (det_count, number_projections)) for img in self.registeredVolume[sample]])
    
    self.registeredVolume[sample] = np.moveaxis(self.registeredVolume[sample], 0,-1)

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

  def getFishParts(self):

    return self.fishPartsAvailable

  def registerDataset(self, sample, inPlace = False):

    """
    Registers full dataset, by sample. (deprecated method)
    Params:
      -sample (string): fish part sample.
    """  
    
    # self. = self.dataset[df.dataset.Sample == '000'].sort_values('Angle', axis = 0).reset_index(drop=True)
    if sample is not None:

      sample = self.fishPart[sample]
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
        self.registeredDataset = self.registeredDataset.append({'Image' : sitk.GetArrayFromImage(fixed_s_T),
                                      'Angle': angle,
                                      'Sample': sample}, ignore_index=True)
        self.registeredDataset = self.registeredDataset.append({'Image' : np.flipud(sitk.GetArrayFromImage(moving_s_T)),
                                      'Angle': angle+self.maxAngle,
                                      'Sample': sample}, ignore_index=True)
    
    # Order by angle
    self.registeredDataset = self.registeredDataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)
  
  def applyRegistration(self):
    """
    Applies mean registration for dataset from registration params
    """

    assert(self.Tparams is not None)
    
    self.registeredDataset = pd.DataFrame(columns = ['Image', 'Angle', 'Sample'])

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
        self.registeredDataset = self.registeredDataset.append({'Image' : sitk.GetArrayFromImage(fixed_s_T),
                                          'Angle': angle,
                                          'Sample': sample}, ignore_index=True)
        self.registeredDataset = self.registeredDataset.append({'Image' : sitk.GetArrayFromImage(moving_s_T),
                                          'Angle': angle+self.maxAngle//2,
                                          'Sample': sample}, ignore_index=True)
      
        self.dataset = self.dataset.drop(self.dataset[self.dataset.Sample == sample].index)
    
    self.registeredDataset = self.registeredDataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)
    del self.dataset
  
  def getRegisteredVolume(self, sample ,saveDataset = True, margin = 10, useSegmented = False):
    '''
    Returns registered and stacked numpy volume, ordered by angle
    Calculates lower and upper non-zero limits for sinograms, with a safety
    margin given by margin.
    '''
    assert(self.registeredDataset is not None)

    # if self.registeredVolume is not None:

      # return self.registeredVolume

    # Filter by sample
    self.registeredVolume = np.stack(self.registeredDataset[self.registeredDataset.Sample == sample]['Image'].to_numpy())
    self.registeredAngles = np.stack(self.registeredDataset[self.registeredDataset.Sample == sample]['Angle'].to_numpy())
    
    # Calculates non-zero boundary limit for segmenting the volume
    self.upperLimit = self.registeredVolume.shape[1]-margin
    self.lowerLimit = margin
    
    # save dataset in HDF5
    if saveDataset == True:
      
      with h5py.File(self.datasetFolder+'/'+'OPTdatasets.hdf5', 'a') as datasets_file:
        
        # If experiment isn't in the current folder, creates experiment
        if self.experimentName not in datasets_file.keys():

          datasets_file.create_group(self.experimentName)
        
        # Creates experiment specifics 
        if self.folderName not in datasets_file[self.experimentName]:

          datasets_file[self.experimentName].create_group(self.folderName)
        
        datasets_file[self.experimentName][self.folderName].create_dataset(sample, data = self.registeredVolume)
        datasets_file[self.experimentName][self.folderName].create_dataset(sample+'_angles', data = self.registeredAngles)
    
    # Normalize volume
    if useSegmented == True:
  
      return self.registeredVolume[:, self.lowerLimit:self.upperLimit, :]
    
    else:
    
      return self.registeredVolume
  
  def saveRegTransforms(self):
    
    with open(str(self.folderPath)+'transform.pickle', 'wb') as h:
      pickle.dump(self.Tparams,  h)

  def loadRegTransforms(self):

    with open(str(self.folderPath)+'transform.pickle', 'rb') as h:
      self.Tparams = pickle.load(h)
    # save mean displacement for operations
    self.meanDisplacement = self.Tparams['Ty'].mean()
  
  def saveRegisteredDataset(self, name = '', mode = 'hdf5'):
    '''
    Saves registered dataset for DL usage (HDF5) or just pickle for binary storage
    params :
    '''

    if mode == 'pickle':

      with open(str(self.folderPath)+name+'.pickle', 'wb') as pickleFile:
      
        pickle.dump({'reg_dataset' : self.registeredDataset,
                    'reg_transform' : self.Tparams}, pickleFile)

    elif mode == 'hdf5':
      
      with pd.HDFStore(self.datasetFolder+'/'+'OPTdatasets.hdf5', 'a') as datasets_file:
        # Take each sample and creates a new dataset
        for sample in self.registeredDataset.Sample.unique():
            
            # Using Pandas built-in HDF5 converter save images
          datasets_file.put(key = self.experimentName+'/'+self.folderName+'/'+sample,
                            value = self.registeredDataset[self.registeredDataset.Sample == sample],
                            data_columns = True)
            
          # Metadata includes angles and eventually other parameters
          # print(self.experimentName+'/'+self.folderName+'/'+sample+'/'+'values')
          # datasets_file[self.experimentName+'/'+self.folderName+'/'+sample+'/'+'values'].attrs['Angle'] = self.registeredDataset[self.registeredDataset.Sample == 'head']['Angle'].to_numpy()

  def loadRegisteredDataset(self):

    with open(str(self.folderPath)+pickleName+'.pickle', 'rb') as pickleFile:
      
      reg = pickle.load(pickleFile)
      self.registeredDataset = reg['reg_dataset']
      self.Tparams = reg['reg_transform']

  def deleteSection(self, sample):

    self.registeredDataset = self.registeredDataset.drop(self.registeredDataset[self.registeredDataset.Sample == sample].index)

def getDataset(sample, experimentName, randomChoice = True, datasetFilename = 'Datasets/OPTdatasets.hdf5'):
  '''
  Gets specific(s) dataset(s) from the box.
  Params:
    - experiment_name (str): indicates experiment's maker
    - sample (str) : if needed, selects kind of sample part (for instance, head) 
    - dataset_filename (str) : filename for looking up datasets
  '''

  print('Loading registered dataset\n')

  with h5.File(datasetFilename, 'r') as f:

    if randomChoice == True:

      randomFolder = random.choice(list(f[experimentName].keys()))
      
    volume, angles = f[experimentName][randomFolder][sample][:], f[experimentName][randomFolder][sample+'_angles'][:]

  return volume, angles


def subsample(volume, max_angle, angle_step, subsampling_type = 'linear'):
  '''
  Subsamples projections according to maximum angle. 
  Params : 
    volume (np.ndarray) : Projection volume, [angles, x-axis, z-axis]
    angle_step : For linear subsampling, reconstruction
    subsampling_type (string) :  Type of subsampling (linear, golden angle)
  
  Returns subsampled volume and angles.
  '''

  if subsampling_type == 'linear':

    beams = int(max_angle/angle_step)
    angles = np.linspace(0., max_angle-max_angle/beams, beams)
    
    return angles, volume[angles.astype(int),:,:]
  
  else:

    return angles, volume

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    # plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    print('Metric :', registration_method.GetMetricValue())
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask






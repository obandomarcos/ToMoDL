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
import os 
from tqdm import tqdm
import SimpleITK as sitk
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import pickle
import h5py
import cv2
import scipy.ndimage as ndi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ZebraDataset:
  '''
  Zebra dataset 
  Params:
    - folder_path (string): full path folder 
  '''
  def __init__(self, folder_path, dataset_folder, experiment_name):
    '''
    Initialises Zebrafish dataset according to Bassi format.
    Folder
    '''

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
    self.file_list = self._search_all_files(self.folder_path)
    self.load_list = [f for f in self.file_list if ('tif' in str(f))]
    self.fish_parts_available = []

    # Check available parts - sample
    for (fish_part_key, fish_part_value) in self.fish_part_code.items():
      
      for s in self.load_list:
        
        if fish_part_value in str(s):
          
          self.fish_parts_available.append(fish_part_key)
          break
      
  def _search_all_files(self, x):
    '''
    Searches all files corresponding to images of the dataset

    Params:
      - x (string): Folder path
    '''
    dirpath = x
    assert(dirpath.is_dir())
    file_list = []
    
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(self._search_all_files(x))

    return file_list

  def loaded_images(self, sample = None):
    '''
    Loads images in volume

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
    
    self.image_volume = np.moveaxis(np.stack(self.dataset['Image'].to_numpy()), 1, 2)
    del self.dataset
  
  def correct_rotation_axis(self,  max_shift = 200, shift_step = 4, center_shift_top = 0, center_shift_bottom = 0, sample = 'head', load_shifts = False, save_shifts = True):
    '''
    Corrects rotation axis by finding optimal registration via maximising reconstructed image's intensity variance.

    Based on 'Walls, J. R., Sled, J. G., Sharpe, J., & Henkelman, R. M. (2005). Correction of artefacts in optical projection tomography. Physics in Medicine & Biology, 50(19), 4645.'
    '''
    
    if load_shifts == True:
      
      with open(str(self.shifts_path)+"_{}".format(sample)+".pickle", 'rb') as f:
        
        self.shifts[sample] = pickle.load(f)
    
    else:
    
      # Grab top and bottom sinograms (automate to grab non-empty sinograms)
      top_index, bottom_index = self._grab_image_indexes()

      self.top_sino = np.copy(self.registered_volume[sample][:,:,top_index].T)
      self.bottom_sino = np.copy(self.registered_volume[sample][:,:,bottom_index].T)
      self.angles = np.linspace(0, 2*180, self.top_sino.shape[1] ,endpoint = False)

      # Iteratively sweep from -maxShift pixels to maxShift pixels
      (top_shift_max, bottom_shift_max) = self._search_shifts(max_shift, shift_step, center_shift_top, center_shift_bottom)

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
    Register volume with interpolated shifts.
    Params:
      - sample (string): Name of sample
    """
    assert(self.shifts is not None)

    # Shift according to shifts
    for idx, shift in enumerate(self.shifts[sample]):
      
      self.registered_volume[sample][:,:,idx] = ndi.shift(self.registered_volume[sample][:,:,idx], (0, shift), mode = 'nearest')

  def dataset_resize(self, sample, img_resize, number_projections):
    """
    Resizes sinograms according to reconstruction image size
    """
    # Move axis to (N_projections, n_detector, n_slices)
    self.registered_volume[sample] = np.rollaxis(self.registered_volume[sample], 2)
    # Resize projection number % 16

    det_count = int((img_resize+0.5)*np.sqrt(2))
  
    self.registered_volume[sample] = np.array([cv2.resize(img, (det_count, number_projections)) for img in self.registered_volume[sample]])
    
    self.registered_volume[sample] = np.moveaxis(self.registered_volume[sample], 0,-1)

  def _grab_image_indexes(self, threshold = 50):
    """
    Grabs top and bottom non-empty indexes
    """
    img_max = self.image_volume.min(axis = 0)
    img_max = (((img_max-img_max.min())/(img_max.max()-img_max.min()))*255.0).astype(np.uint8)
    img_max = ndi.gaussian_filter(img_max,(11,11))

    top_index, bottom_index = (np.where(img_max.std(axis = 0)>threshold)[0][0],np.where(img_max.std(axis = 0)>threshold)[0][-1])
    
    print('Top index:', top_index)
    print('Bottom index:', bottom_index)
    
    return top_index, bottom_index

  def _search_shifts(self, max_shift, shift_step, center_shift_top, center_shift_bottom):

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

  def get_fish_parts(self):

    return self.fish_parts_available

  def register_dataset(self, sample, inPlace = False):

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
  
  def get_registered_volume(self, sample ,saveDataset = True, margin = 10, useSegmented = False):
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
            
  def loadregistered_dataset(self):

    with open(str(self.folder_path)+pickleName+'.pickle', 'rb') as pickleFile:
      
      reg = pickle.load(pickleFile)
      self.registered_dataset = reg['reg_dataset']
      self.Tparams = reg['reg_transform']

  def delete_section(self, sample):

    self.registered_dataset = self.registered_dataset.drop(self.registered_dataset[self.registered_dataset.Sample == sample].index)

  def delete_registered_volume(self, sample):
    
    del self.registered_volume[sample]

# Multi-dataset to dataloader
class ZebraDataloader:

  def __init__(self, folder_paths):
    '''
    Initializes dataloader with paths of folders
    '''    
    self.folder_paths = folder_paths

  def formRegDatasets(self, img_resize = 100, number_projections = 640, experiment = 'Bassi', fish_parts = None):
    """
    Forms registered datasets from raw projection data.
    params:
        - folder_paths (string): paths to the training and test folders
        - img_resize (int): image resize, detector count is calculated to fit this number and resize sinograms
        - n_proy (int): Number of projections to resize the sinogram, in order to work according to Torch Radon specifications
        - sample (string): name of the part of the boy to be sampled, defaults to head
        - experiment (string): name of the experiment
    """
    # Paths of pickled registered datasets
    datasets_registered = []

    for dataset_num, folder_path in enumerate(self.folder_paths):                                     
        
        # Loads dataset registered
        df = ZebraDataset(folder_path, 'Datasets', 'Bassi')
        fish_parts = df.get_fish_parts()    
        
        for sample in fish_parts:
            # If the registered dataset exist, just add it to the list

            registered_dataset_path = str(folder_path)+'_'+sample+'_registered'+'.pkl'

            if os.path.isfile(registered_dataset_path) == True:
                
                datasets_registered.append(registered_dataset_path)

            else:
                # Load sample dataset
                df.loadImages(sample = sample)
                # Load corresponding registrations
                
                df.correct_rotation_axis(sample = sample, max_shift = 200, shift_step = 1, load_shifts = True, save_shifts = False)
                
                print(df.registeredVolume[sample].shape)
                # Append volumes        
                print("Dataset {}/{} loaded - {} {}".format(dataset_num+1, len(folder_paths), str(df.folderName), sample))
                
                # Resize registered volume to desired
                df.dataset_resize(sample, img_resize, number_projections)

                with open(registered_dataset_path, 'wb') as f:
                    
                    print(df.registeredVolume[sample].shape)
                    pickle.dump(df.registeredVolume[sample], f)
                    datasets_registered.append(registered_dataset_path)
                
                # Save memory deleting sample volume
                df.delete_registered_volume(sample)
                
            
    return datasets_registered

  def openDataset(self, dataset_path):
              
      with open(str(dataset_path), 'rb') as f:
                      
          datasets_reg = pickle.load(f)

      return datasets_reg

  def formDataloaders(datasets, number_projections, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor = 1, load_tensor = True, save_tensor = False, use_rand = True, k_fold_datasets = True):
      """
      Form torch dataloaders for training and testing, full and undersampled
      params:
          - number_projections is the number of projections the sinogram is reconstructed with for undersampled FBP
          - train size (test_size) is the number of training (test) images to be taken from the volumes. They are taken randomly using formDatasets
          - batch_size is the number of images a batch contains in the dataloader
          - img_size is the size of the new images to be reconstructed
          - augment_factor determines how many times the dataset will be resampled with different seed angles
          - k_fold_datasets sets the number of datasets to be used for k-folding
      """

      fullX = []
      fullY = []
      filtFullX = []

      testX = []
      testY = []
      filtTestX = []
      
      if load_tensor == False:

          l = len(datasets)*augment_factor
          # Augment factor iterates over the datasets for data augmentation
          for i in range(augment_factor):
              
              # Seed angle for data augmentation
              rand_angle = np.random.randint(0, number_projections)

              # Dataset train
              # Masks chosen dataset with the number of projections required
              for k_dataset, dataset_path in enumerate(tqdm(datasets)):
                  
                  dataset = openDataset(dataset_path).astype(float)

                  #print(dataset.shape, dataset_path)
                  print(total_size)
                  tY, tX, filtX = maskDatasets(dataset, number_projections, total_size//l, img_size, rand_angle, use_rand = use_rand)

                  if k_dataset < k_fold_datasets:
                      
                      fullX.append(tX)
                      fullY.append(tY)
                      filtFullX.append(filtX)

                  else:

                      testX.append(tX)
                      testY.append(tY)
                      filtTestX.append(filtX)
                      
          # Stack augmented datasets
          fullX = torch.vstack(fullX)
          filtFullX = torch.vstack(filtFullX)
          fullY = torch.vstack(fullY)
          
          # Stack test dataset separately
          testX = torch.vstack(testX)
          filtTestX = torch.vstack(filtTestX)
          testY = torch.vstack(testY)
          
      else:
          # In order to prevent writing numerous copies of these tensors, loading should be avoided

          fullX = torch.load(tensor_path+'FullX.pt')
          filtFullX = torch.load(tensor_path+'FiltFullX.pt')
          fullY = torch.load(tensor_path+'FullY.pt')
      
      if save_tensor == True:
          # In order to prevent writing numerous copies of these tensors, loading should be avoided
          torch.save(fullX, tensor_path+'FullX.pt')
          torch.save(filtFullX, tensor_path+'FiltFullX.pt')
          torch.save(fullY, tensor_path+'FullY.pt')

      if use_rand == True:
          
          # Randomly shuffle the images
          idx = torch.randperm(fullX.shape[0])
          fullX = fullX[idx].view(fullX.size())
          filtFullX = filtFullX[idx].view(fullX.size())
          fullY = fullY[idx].view(fullX.size())

          # Stack test dataset separately and random shuffle
          idx_test = torch.randperm(testX.shape[0])
          testX = testX[idx_test].view(testX.size())
          filtTestX = filtTestX[idx_test].view(filtTestX.size())
          testY = testY[idx_test].view(testY.size())
          

      len_full = fullX.shape[0]

      # Grab validation slice 
      valX = torch.clone(fullX[:int(val_factor*len_full),...])
      filtValX = torch.clone(filtFullX[:int(val_factor*len_full),...])
      valY = torch.clone(fullY[:int(val_factor*len_full),...])
      
      # Grab train slice
      trainX = torch.clone(fullX[int(val_factor*len_full):,...])
      filtTrainX = torch.clone(filtFullX[int(val_factor*len_full):,...])
      trainY = torch.clone(fullY[int(val_factor*len_full):,...])

      # Build dataloaders
      trainX = torch.utils.data.DataLoader(trainX,
                                            batch_size=batch_size,
                                            shuffle=False, num_workers=0)

      filtTrainX = torch.utils.data.DataLoader(filtTrainX,
                                                batch_size=batch_size,
                                                shuffle=False, num_workers=0)

      trainY = torch.utils.data.DataLoader(trainY, batch_size=batch_size,shuffle=False, num_workers=0)                                 

      testX = torch.utils.data.DataLoader(testX, batch_size=1,shuffle=False, num_workers=0)
      filtTestX = torch.utils.data.DataLoader(filtTestX, batch_size=1, shuffle=False, num_workers=0)
      testY = torch.utils.data.DataLoader(testY, batch_size=1,shuffle=False, num_workers=0)
      
      valX = torch.utils.data.DataLoader(valX, batch_size=batch_size, shuffle=False, num_workers=0)
      filtValX = torch.utils.data.DataLoader(filtValX,batch_size=batch_size, shuffle=False, num_workers=0)
      valY = torch.utils.data.DataLoader(valY, batch_size=batch_size, shuffle=False, num_workers=0)

      # Dictionary reshape
      dataloaders = {'train':{'x':trainX, 'filtX':filtTrainX, 'y':trainY}, 'val':{'x':valX, 'filtX':filtValX, 'y':valY}, 'test':{'x':testX, 'filtX':filtTestX, 'y':testY}}

      return dataloaders

  def maskDatasets(full_sino, num_beams, dataset_size, img_size, angle_seed = 0, use_rand = True):
      '''
      Mask datasets in order to undersample sinograms, obtaining undersampled and fully reconstruction datasets for training.
      Params:
          - full_sino (ndarray): Fully sampled volume of sinograms, with size (n_projections, detector_number, z-slices)
          - num_beams (int): Number of beams to undersample the dataset. The function builds an masking array clamping to zero the values
          that are not sampled in the sinogram.
          - dataset_size (int): number to slices to take from the original sinogram's volume 
          - img_size (int): Size of reconstructed images, in pixels.
          - angle_seed (int): Starting angle to subsample evenly spaced
      '''
      # Copy of input sinogram dataset
      det_count = int((img_size+0.5)*np.sqrt(2))
      undersampled_sino = np.copy(full_sino)

      # Using boolean mask, keep values sampled and clamp to zero others
      print('Init mask datasets')
      zeros_idx = np.linspace(0, full_sino.shape[0], num_beams, endpoint = False).astype(int)
      zeros_idx = (zeros_idx+angle_seed)%full_sino.shape[0]
      zeros_mask = np.full(full_sino.shape[0], True, dtype = bool)
      zeros_mask[zeros_idx] = False
      undersampled_sino[zeros_mask, :, :] = 0

      # Grab number of angles
      n_angles = full_sino.shape[0]
      angles = np.linspace(0, 2*np.pi, n_angles, endpoint = False)
      
      radon = Radon(img_size, angles, clip_to_circle = False, det_count = det_count)
      
      undersampled = []
      undersampled_filtered = []
      desired = []
      # print('Rand {}'.format(use_rand))
      # print(dataset_size)
      # print('Shape sino samples{}'.format(full_sino.shape[2]))
      # Grab random slices
      assert(dataset_size <= full_sino.shape[2])
      if use_rand == True:
          rand = np.random.choice(range(full_sino.shape[2]), dataset_size, replace=False)
      else:
          rand = np.arange(full_sino.shape[2], dataset_size)
      
      # print(rand)
      # print('Zero masked')
      # Inputs
      for i, sino in enumerate(np.rollaxis(undersampled_sino[:,:,rand], 2)):
          
          # Normalization of input sinogram
          sino = torch.FloatTensor(sino).to(device)
          sino = (sino - sino.min())/(sino.max()-sino.min())
          img = radon.backward(sino)*np.pi/n_angles 
          img = (img-img.min())-(img.max()-img.min())

          undersampled.append(img)
      print('Undersampled reconstruction raw')
      # Grab filtered backprojection
      
      for sino in np.rollaxis(undersampled_sino[:,:,rand],2):
          
          sino = torch.FloatTensor(sino).to(device)
          img = radon.filter_sinogram(sino)
          #sino = (sino - sino.min())/(sino.max()-sino.min())
          img = radon.backward(radon.filter_sinogram(sino))
          img = (img - img.min())/(img.max()-img.min())
          del sino

          undersampled_filtered.append(img)
      
      del undersampled_sino
      print('Undersampled reconstruction filtered')
      # Target
      for sino in np.rollaxis(full_sino[:,:,rand], 2):
          
          # Normalization of input sinogram
          sino = torch.FloatTensor(sino).to(device)
          #sino = (sino - sino.min())/(sino.max()-sino.min())
          img = radon.backward(radon.filter_sinogram(sino))
          img = (img - img.min())/(img.max()-img.min())

          desired.append(img)

      print('Undersampled reconstruction full')
      
      # Format dataset to feed network
      desired = torch.unsqueeze(torch.stack(desired), 1)
      undersampled = torch.unsqueeze(torch.stack(undersampled), 1)
      undersampled_filtered = torch.unsqueeze(torch.stack(undersampled_filtered), 1)

      return desired, undersampled, undersampled_filtered

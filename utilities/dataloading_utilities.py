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
import torch
import os 
from tqdm import tqdm
import SimpleITK as sitk
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import pickle
import h5py
import cv2
import scipy.ndimage as ndi

# Modify for multi-gpu
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
    self.image_volume = {}
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

  def load_images(self, sample = None):
    '''
    Loads images in volume

    Params:
      - sample (string): {None, body, head, tail}
    '''
    samples = self.get_fish_parts()

    # Load all samples in dataset
    for sample in samples:

      sample2idx =  self.fish_part[sample]    
      load_list = [f for f in self.file_list if ('tif' in str(f))]

      if sample is not None:
        # If it's None, then it loads all fish parts/sample
        fish_part = self.fish_part_code[sample]
        load_list = [f for f in load_list if (fish_part in str(f))]
      
        
      load_images = []
      loaded_angles = []
      loaded_sample = []
      loaded_channel = []
      loaded_time = []

      angle = re.compile('a\d+')
      sample_re = re.compile('s\d+')
      channel = re.compile('c\d+')
      time = re.compile('t\d+')
      
      for f in tqdm(load_list):
        
        load_images.append(np.array(Image.open(f)))
        loaded_angles.append(float(angle.findall(str(f))[0][1:]))
        loaded_sample.append(float(sample_re.findall(str(f))[0][1:]))
        loaded_channel.append(float(channel.findall(str(f))[0][1:]))
        loaded_time.append(float(time.findall(str(f))[0][1:]))

      self.dataset = pd.DataFrame({'Filename':load_list, 
                                    'Image':load_images,
                                    'Angle':loaded_angles,
                                    'Sample':loaded_sample,
                                    'Channel':loaded_channel,
                                    'Time':loaded_time})

      # Create Registered Dataset - empty till reggistering
      self.registered_dataset = pd.DataFrame(columns = ['Image', 'Angle', 'Sample'])

      # Sort dataset by sample and angle
      self.dataset = self.dataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)

      self.registered_volume[sample] = np.stack(self.dataset[self.dataset.Sample == sample2idx]['Image'].to_numpy())

      del self.dataset, self.registered_dataset
  
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
    '''
    Resizes sinograms according to reconstruction image size
    '''
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
    Registers full dataset, by sample. (DEPRECATED METHOD)
    Params:
      -sample (string): fish part sample.
    """  
    
    # self. = self.dataset[dataset.dataset.Sample == '000'].sort_values('Angle', axis = 0).reset_index(drop=True)
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
  
  def apply_registration(self):
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
    
    # save dataset in Hdataset5
    if saveDataset == True:
      
      with h5py.File(self.dataset_folder+'/'+'OPTdatasets.hdataset5', 'a') as datasets_file:
        
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
  
  def save_reg_transforms(self):
    
    with open(str(self.folder_path)+'transform.pickle', 'wb') as h:
      pickle.dump(self.Tparams,  h)

  def load_reg_transforms(self):

    with open(str(self.folder_path)+'transform.pickle', 'rb') as h:
      self.Tparams = pickle.load(h)
    # save mean displacement for operations
    self.meanDisplacement = self.Tparams['Ty'].mean()
  
  def save_registered_dataset(self, name = '', mode = 'hdataset5'):
    '''
    Saves registered dataset for DL usage (Hdataset5) or just pickle for binary storage
    params :
    '''

    if mode == 'pickle':

      with open(str(self.folder_path)+name+'.pickle', 'wb') as pickleFile:
      
        pickle.dump({'reg_dataset' : self.registered_dataset,
                    'reg_transform' : self.Tparams}, pickleFile)

    elif mode == 'hdataset5':
      
      with pd.HdatasetStore(self.dataset_folder+'/'+'OPTdatasets.hdataset5', 'a') as datasets_file:
        # Take each sample and creates a new dataset
        for sample in self.registered_dataset.Sample.unique():
            
            # Using Pandas built-in Hdataset5 converter save images
          datasets_file.put(key = self.experiment_name+'/'+self.folder_name+'/'+sample,
                            value = self.registered_dataset[self.registered_dataset.Sample == sample],
                            data_columns = True)
            
  def load_registered_dataset(self):

    with open(str(self.folder_path)+pickleName+'.pickle', 'rb') as pickleFile:
      
      reg = pickle.load(pickleFile)
      self.registered_dataset = reg['reg_dataset']
      self.Tparams = reg['reg_transform']

  def delete_section(self, sample):

    self.registered_dataset = self.registered_dataset.drop(self.registered_dataset[self.registered_dataset.Sample == sample].index)

  def delete_registered_volume(self, sample):
    
    del self.registered_volume[sample]
  
  @staticmethod
  def _normalize_image(img):
    '''
    Normalizes images between 0 and 1.
    Params: 
      - img (ndarray): Image to normalize
    '''
    img = (img-img.max())/(img.max()-img.min())

    return img

# Multi-dataset to dataloader
class ZebraDataloader:
  '''
  List of tasks:
    1 - Load ZebraDatasets
    2 - Reshape
    3 - Register and correct artifacts
    4 - Create torch.Dataset
    5 - Load Dataloader
  
  '''
  def __init__(self, kw_dictionary, experiment_name = 'Bassi'):
    '''
    Initializes dataloader with paths of folders.
    Params: 
      - folder_paths (list of strings): Folders from where to load data
      - img_resize (int): New size in tensor
      - experiment_name (string): string maps to experiment dataset constitution
    '''    
    
    self.zebra_datasets = {}
    self.datasets_registered = []

    self.__process_dataloader_kwdict(kw_dictionary)
    
    for dataset_num, folder_path in enumerate(self.folder_paths):                                     
       
      # Loads dataset registered
      self.zebra_datasets[folder_path] = ZebraDataset(folder_path, 'Datasets', experiment_name)
  
  def __getitem__(self, folder_path):
      
    if folder_path not in self.zebra_datasets.keys():
      raise KeyError

    return self.zebra_datasets[folder_path]

  def __process_dataloader_kwdict(self, kw_dictionary):
    '''
    Load keyword arguments for dataloader
    Params:
     - kw_dictionary (dict): Dictionary containing keywords
    '''

    self.folder_paths = kw_dictionary.pop('folder_paths')
    self.img_resize = kw_dictionary.pop('img_resize')
    self.det_count = int((self.img_resize+0.5)*np.sqrt(2))
    self.total_size = kw_dictionary.pop('total_size')
    self.tensor_path = kw_dictionary.pop('tensor_path')
    self.train_factor = kw_dictionary.pop('train_factor')
    self.val_factor = kw_dictionary.pop('val_factor')
    self.test_factor = kw_dictionary.pop('test_factor')
    self.augment_factor = kw_dictionary.pop('augment_factor')
    self.load_shifts = kw_dictionary.pop('load_shifts')
    self.save_shifts = kw_dictionary.pop('save_shifts')
    self.load_tensor = kw_dictionary.pop('load_tensor')
    self.save_tensor = kw_dictionary.pop('save_tensor')
    self.use_rand = kw_dictionary.pop('use_rand')
    self.k_fold_datasets = kw_dictionary.pop('k_fold_datasets')

    # Define number of angles and radon transform to undersample  
    self.number_projections_total = kw_dictionary.pop('number_projections_total')
    self.number_projections_undersample = kw_dictionary.pop('number_projections_undersample')
    self.acceleration_factor = self.number_projections_total//self.number_projections_undersample

    self._create_radon()

    self.sampling_method = kw_dictionary.pop('sampling_method')
    self.batch_size = kw_dictionary.pop('batch_size')

  def register_datasets(self):
    """
    Forms registered datasets from raw projection data. Corrects axis shift, resizes for tensor and saves volume.

    Params:
        - folder_paths (string): paths to the training and test folders
        - img_resize (int): image resize, detector count is calculated to fit this number and resize sinograms
        - n_proy (int): Number of projections to resize the sinogram, in order to work according to Torch Radon specifications
        - sample (string): name of the part of the boy to be sampled, defaults to head
        - experiment (string): name of the experiment
    """

    for dataset_num, (folder_path, dataset) in enumerate(self.zebra_datasets.items()):                                     
        # Check fish parts available
        fish_parts = dataset.get_fish_parts() 
        
        # Load images from all samples
        dataset.load_images(sample = None)

        for sample in fish_parts:
            
            # If the registered dataset exist, just add it to the list
            registered_dataset_path = str(folder_path)+'_'+sample+'_registered'+'_size_{}'.format(self.img_resize)+'.pkl'

            if os.path.isfile(registered_dataset_path) == True:
                
                self.datasets_registered.append(registered_dataset_path)

            else:

                # Load corresponding registrations
                dataset.correct_rotation_axis(sample = sample, max_shift = 200, shift_step = 1, load_shifts = self.load_shifts, save_shifts = self.save_shifts)
                
                # Append volumes        
                print("Dataset {}/{} loaded - {} {}".format(dataset_num+1, len(self.folder_paths), str(dataset.folder_name), sample))
                
                # Resize registered volume to desired
                dataset.dataset_resize(sample, self.img_resize, self.number_projections_total)

                with open(registered_dataset_path, 'wb') as f:
                    
                    print(dataset.registered_volume[sample].shape)
                    pickle.dump(dataset.registered_volume[sample], f)
                    self.datasets_registered.append(registered_dataset_path)
                
                # Save memory deleting sample volume
                dataset.delete_registered_volume(sample)

  def open_dataset(self, dataset_path):
    '''
    Opens pickled registered dataset
    Params: 
      - dataset_path (string)
    '''
              
    with open(str(dataset_path), 'rb') as f:
                    
        datasets_reg = pickle.load(f)

    return datasets_reg

  def get_registered_datasets(self):

    return self.datasets_registered

  def build_dataloaders(self):
    """
    Build dataloaders from registered datasets. 
    To-Do: 
      - Compartimentalize and do K-Folding separately
      - Reduce intensive memory consumption
    """

    full_x = []
    full_y = []
    filt_full_x = []

    test_x = []
    test_y = []
    filt_test_x = []
    
    if self.load_tensor == False:

      l = len(self.datasets_registered)*self.augment_factor
      # Augment factor iterates over the datasets for data augmentation
      for i in range(self.augment_factor):

        # Dataset train
        # Masks chosen dataset with the number of projections required
        for k_dataset, dataset_path in enumerate(tqdm(self.datasets_registered)):
            
          dataset = self.open_dataset(dataset_path).astype(float)

          tY, tX, filtX = self.mask_datasets(dataset, self.total_size//l)

          if k_dataset < self.k_fold_datasets:
              
              full_x.append(tX)
              full_y.append(tY)
              filt_full_x.append(filtX)

          else:

              test_x.append(tX)
              test_y.append(tY)
              filt_test_x.append(filtX)
                
      # Stack augmented datasets
      full_x_tensor = torch.vstack(full_x)
      filt_full_x_tensor = torch.vstack(filt_full_x)
      full_y_tensor = torch.vstack(full_y)
      
      # Stack test dataset separately
      test_x_tensor = torch.vstack(test_x)
      test_filt_x_tensor = torch.vstack(filt_test_x)
      test_y_tensor = torch.vstack(test_y)

      del full_x, filt_full_x, full_y, test_x, filt_test_x, test_y
      
    else:
      # In order to prevent writing numerous copies of these tensors, loading should be avoided

      full_x_tensor = torch.load(self.tensor_path+'full_x.pt')
      filt_full_x_tensor = torch.load(self.tensor_path+'filt_full_x.pt')
      full_y_tensor = torch.load(self.tensor_path+'full_y.pt')
    
    if self.save_tensor == True:
      # In order to prevent writing numerous copies of these tensors, loading should be avoided
      torch.save(full_x_tensor, self.tensor_path+'full_x.pt')
      torch.save(filt_full_x_tensor, self.tensor_path+'filt_full_x.pt')
      torch.save(full_y_tensor, self.tensor_path+'full_y.pt')

    if self.use_rand == True:
        
      # Randomly shuffle the images
      idx = torch.randperm(full_x_tensor.shape[0])
      full_x_tensor = full_x_tensor[idx].view(full_x_tensor.size()).to(device)
      filt_full_x_tensor = filt_full_x_tensor[idx].view(filt_full_x_tensor.size()).to(device)
      full_y_tensor = full_y_tensor[idx].view(full_x_tensor.size()).to(device)

      # Stack test dataset separately and random shuffle
      idx_test = torch.randperm(test_x_tensor.shape[0])
      test_x_tensor = test_x_tensor[idx_test].view(test_x_tensor.size())
      test_filt_x_tensor = test_filt_x_tensor[idx_test].view(test_filt_x_tensor.size())
      test_y_tensor = test_y_tensor[idx_test].view(test_y_tensor.size())

    len_full = full_x_tensor.shape[0]

    # Grab validation slice 
    val_x_tensor = torch.clone(full_x_tensor[:int(self.val_factor*len_full),...])
    val_filt_x_tensor = torch.clone(filt_full_x_tensor[:int(self.val_factor*len_full),...])
    val_y_tensor = torch.clone(full_y_tensor[:int(self.val_factor*len_full),...])
    
    # Grab train slice
    train_x_tensor = torch.clone(full_x_tensor[int(self.val_factor*len_full):,...])
    train_filt_x_tensor = torch.clone(filt_full_x_tensor[int(self.val_factor*len_full):,...])
    train_y_tensor = torch.clone(full_y_tensor[int(self.val_factor*len_full):,...])

    # Build dataloaders
    train_dataloader = torch.utils.data.DataLoader((train_x_tensor, train_filt_x_tensor ,train_y_tensor),
                                        batch_size=self.batch_size,
                                        shuffle=False, num_workers=0)

    test_dataloader = torch.utils.data.DataLoader((test_x_tensor, test_filt_x_tensor, test_y_tensor), 
                                        batch_size=self.batch_size,shuffle=False,num_workers=0)

    val_dataloader = torch.utils.data.DataLoader((val_x_tensor, val_filt_x_tensor, val_y_tensor),
                                      batch_size=self.batch_size,
                                      shuffle=False, num_workers=0)

    # Dictionary reshape
    self.dataloaders = {'train':train_dataloader,        
                        'val': val_dataloader,
                        'test':test_dataloader}

  def mask_datasets(self, full_sinograms, dataset_size):
    '''
    Mask datasets in order to undersample sinograms, obtaining undersampled and full_y reconstruction datasets for training.
    Params:
        - full_sinograms (ndarray): full_y sampled volume of sinograms, with size (n_projections, detector_number, z-slices)
        - num_beams (int): Number of beams to undersample the dataset. The function builds an masking array clamping to zero the values
        that are not sampled in the sinogram.
        - dataset_size (int): number to slices to take from the original sinogram's volume 
        - img_size (int): Size of reconstructed images, in pixels.
        - rand_angle (int): Starting angle to subsample evenly spaced
    '''    
    
    # List for tensor formation (this is pretty memory intensive)
    undersampled = []
    undersampled_filtered = []
    desired = []

    # Assert if dataset_size requested is larger than full_sinograms
    assert(dataset_size <= full_sinograms.shape[2])

    # Using boolean mask, keep values sampled and clamp to zero others
    # Masking dataset has to be on its own
    print('Masking with {} method'.format(self.sampling_method))
    undersampled_sinograms = self.subsample_sinogram(full_sinograms, self.sampling_method)
    
    if self.use_rand == True:
        rand = np.random.choice(range(full_sinograms.shape[2]), dataset_size, replace=False)
    else:
        rand = np.arange(full_sinograms.shape[2], dataset_size)
    
    # Grab random slices and roll axis so to sample slices
    undersampled_sinograms = torch.FloatTensor(np.rollaxis(undersampled_sinograms[:,:,rand], 2)).to(device)
    full_sinograms = torch.FloatTensor(np.rollaxis(full_sinograms[:,:,rand], 2)).to(device)
    
    # Inputs
    for us_sinogram, full_sinogram in tqdm(zip(undersampled_sinograms, full_sinograms)):
        
        # Normalization of input sinogram - Undersampled
        img = self.radon.backward(self.radon.filter_sinogram(us_sinogram))
        img = self.normalize_image(img)
        
        # Undersampled filtered
        undersampled_filtered.append(img)
        
        # Normalize 0-1 under sampled sinogram
        us_sinogram = self.normalize_image(us_sinogram)

        img = self.radon.backward(us_sinogram)*np.pi/self.number_projections_undersample
        img = self.normalize_image(img)

        # Undersampled raw backprojection
        undersampled.append(img)
        
        # Normalization of output sinogram - Fully sampled
        img = self.radon.backward(self.radon.filter_sinogram(full_sinogram))
        img = self.normalize_image(img)

        # Fully sampled filtered backprojection
        desired.append(img)
    
    # Format dataset to feed network
    desired = torch.unsqueeze(torch.stack(desired), 1)
    undersampled = torch.unsqueeze(torch.stack(undersampled), 1)
    undersampled_filtered = torch.unsqueeze(torch.stack(undersampled_filtered), 1)

    return desired, undersampled, undersampled_filtered

  def subsample_sinogram(self, sinogram, method = 'equispaced-linear'):
    '''
    Subsamples sinogram by masking images with zeros with a particular method.
    Params: 
      sinogram (ndarray): sinogram to mask with defined method
      method (string): Sampling meethod to mask sinogram
        Avalaible methods:
          * linear-equispaced: from a random angle seed, samples equispaced the angular space.
    '''

    # Seed angle for data augmentation
    if method == 'equispaced-linear':
      
      undersampled_sinogram = np.copy(sinogram)
      rand_angle = np.random.randint(0, self.number_projections_total)

      # Zeros Masking
      zeros_idx = np.linspace(0, self.number_projections_total, self.number_projections_undersample, endpoint = False).astype(int)
      zeros_idx = (zeros_idx+rand_angle)%self.number_projections_total
      zeros_mask = np.full(self.number_projections_total, True, dtype = bool)
      zeros_mask[zeros_idx] = False
      undersampled_sinogram[zeros_mask, :, :] = 0
    
    return undersampled_sinogram

  def _get_next_from_dataloader(self, set_name):
    '''
    Gets next item in dataloader.
    Params:
      - set_name (string): Refers to set of data where image comes from (train, val, test)
    '''

    return next(iter(self.dataloaders[set_name]))

  def _create_radon(self):
    '''
    Creates Torch-Radon method for the desired sampling and image size of the dataloader
    '''
    # Grab number of angles
    self.angles = np.linspace(0, 2*np.pi, self.number_projections_total, endpoint = False)
    
    self.radon = Radon(self.img_resize, self.angles, clip_to_circle = False, det_count = self.det_count)

  @staticmethod
  def normalize_image(image):

    return (image - image.min())/(image.max()-image.min())
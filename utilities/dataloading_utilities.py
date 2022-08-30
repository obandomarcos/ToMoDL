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
from torch_radon import Radon
import pickle
from pathlib import Path
import torch
import cv2
import scipy.ndimage as ndi
from torch.utils.data import Dataset

# Modify for multi-gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasetProcessor:
  '''
  Zebra dataset 
  Params:
    - folder_path (string): full path folder 
  '''
  def __init__(self, kw_dictionary):
    '''
    Initialises Zebrafish dataset according to Bassi format.
    Folder
    '''

    self.process_kwdictionary(kw_dictionary)

    for sample in self.fish_parts_available:
      
      self.load_images(sample)
      self.correct_rotation_axis(sample = sample, max_shift = 200, shift_step = 1)
      self.dataset_resize(sample)
      self.write_dataset_reconstruction(sample)

      del self.registered_volume[sample]
  
  def process_kwdictionary(self, kw_dictionary): 
    '''
    Load keyword arguments for dataloader
    Params:
     - kw_dictionary (dict): Dictionary containing keywords
    '''

    self.folder_path = pathlib.Path(kw_dictionary['folder_path'])
    self.folder_name = pathlib.PurePath(self.folder_path).name
    
    self.objective = 10
    self.dataset_folder = kw_dictionary['dataset_folder']
    self.experiment_name =kw_dictionary['experiment_name']

    self.image_volume = {}
    self.registered_volume = {}
    self.shifts_path = self.folder_path

    self.img_resize = kw_dictionary['img_resize']
    self.det_count = int((self.img_resize+0.5)*np.sqrt(2))
    
    self.load_shifts = kw_dictionary['load_shifts']
    self.save_shifts = kw_dictionary['save_shifts']

    # Define number of angles and radon transform to undersample  
    self.number_projections_total = kw_dictionary['number_projections_total']
    self.number_projections_undersampled = kw_dictionary['number_projections_undersampled']
    self.acceleration_factor = self.number_projections_total//self.number_projections_undersampled

    self._create_radon()

    self.sampling_method = kw_dictionary['sampling_method']
    

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

    return 

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
  
  def correct_rotation_axis(self,  max_shift = 200, shift_step = 4, center_shift_top = 0, center_shift_bottom = 0, sample = 'head'):
    '''
    Corrects rotation axis by finding optimal registration via maximising reconstructed image's intensity variance.

    Based on 'Walls, J. R., Sled, J. G., Sharpe, J., & Henkelman, R. M. (2005). Correction of artefacts in optical projection tomography. Physics in Medicine & Biology, 50(19), 4645.'
    '''
    
    if self.load_shifts == True:
      
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

    if self.save_shifts == True:
      
      with open(str(self.shifts_path)+"_{}".format(sample)+".pickle", 'wb') as f:

        pickle.dump(self.shifts, f)

    # Create Registered volume[sample] with the shifts
    self.register_volume(sample)

  def register_volume(self, sample):
    """
    Register volume with interpolated shifts.
    Params:
      - sample (string): Name of sample
    """
    assert(self.shifts is not None)

    # Shift according to shifts
    for idx, shift in enumerate(tqdm(self.shifts[sample], 'Registering in progress')):
      
      self.registered_volume[sample][:,:,idx] = ndi.shift(self.registered_volume[sample][:,:,idx], (0, shift), mode = 'nearest')

  def dataset_resize(self, sample):
    '''
    Resizes sinograms according to reconstruction image size
    '''
    
    print('Resizing in progress...')
    # Move axis to (N_projections, n_detector, n_slices)
    self.registered_volume[sample] = np.rollaxis(self.registered_volume[sample], 2)
    # Resize projection number % 16
  
    self.registered_volume[sample] = np.array([cv2.resize(img, (self.det_count, self.number_projections_total)) for img in self.registered_volume[sample]])
    print('Finished')
    
    self.registered_volume[sample] = np.moveaxis(self.registered_volume[sample], 0,-1)

  def write_dataset_reconstruction(self, sample):
    '''
    Mask datasets in order to undersample sinograms, obtaining undersampled and full_y reconstruction datasets for training. Then saves images in folder 
    Params:
        - full_sinogram (ndarray): full_y sampled sinogram, with size (n_projections, detector_number, z-slices)
        - num_beams (int): Number of beams to undersample the dataset. The function builds an masking array clamping to zero the values
        that are not sampled in the sinogram.
        - dataset_size (int): number to slices to take from the original sinogram's volume 
        - img_size (int): Size of reconstructed images, in pixels.
        - rand_angle (int): Starting angle to subsample evenly spaced
    '''    
    # Create dataset folder and subfolders for acceleration folders
    reconstructed_dataset_folder = self.dataset_folder+'/x'+str(self.acceleration_factor)+'/'+self.folder_name+'_'+sample+'_'+str(self.acceleration_factor)+'/'

    # Try to build datafolder if it doesn't exist, otherwise assume that it is already available.
    try:
      Path(reconstructed_dataset_folder).mkdir(parents=True)
    except FileExistsError:
      return 

    us_unfiltered_dataset_folder = reconstructed_dataset_folder+'us_{}_unfiltered/'.format(self.acceleration_factor)
    fs_filtered_dataset_folder = reconstructed_dataset_folder+'fs_filtered/'
    us_filtered_dataset_folder = reconstructed_dataset_folder+'us_{}_filtered/'.format(self.acceleration_factor)

    Path(us_unfiltered_dataset_folder).mkdir(parents=True, exist_ok=True)
    Path(fs_filtered_dataset_folder).mkdir(parents=True, exist_ok=True)
    Path(us_filtered_dataset_folder).mkdir(parents=True, exist_ok=True)

    write_us_unfiltered = os.path.isdir(us_unfiltered_dataset_folder)
    write_fs_filtered = os.path.isdir(fs_filtered_dataset_folder)
    write_us_filtered = os.path.isdir(us_filtered_dataset_folder)
    
    # Grab full sinogram
    full_sinogram = self.registered_volume[sample].astype(float)

    # Using boolean mask, keep values sampled and clamp to zero others
    # Masking dataset has to be on its own
    print('Masking with {} method'.format(self.sampling_method))
    undersampled_sinograms = self.subsample_sinogram(full_sinogram, self.sampling_method)
    
    # Grab random slices and roll axis so to sample slices
    undersampled_sinograms = torch.FloatTensor(np.rollaxis(undersampled_sinograms, 2)).to(device)
    full_sinogram = torch.FloatTensor(np.rollaxis(full_sinogram, 2)).to(device)
    
    # Inputs
    for sinogram_slice, (us_sinogram, full_sinogram) in tqdm(enumerate(zip(undersampled_sinograms, full_sinogram))):
        
        sinogram_slice = str(sinogram_slice)
        
        if write_us_filtered == True:
          # Undersampled filtered reconstructed image path
          us_filtered_img_path = us_filtered_dataset_folder+sinogram_slice+'.jpg'
          
          # Normalization of input sinogram - Undersampled
          us_filtered_img = self.radon.backward(self.radon.filter_sinogram(us_sinogram))
          us_filtered_img = self.normalize_image(us_filtered_img)
          
          # Write undersampled filtered
          thumbs =cv2.imwrite(us_filtered_img_path, 255.0*us_filtered_img.cpu().detach().numpy())
          print(thumbs)
        
        if write_us_unfiltered == True: 
          print('Slice {sinogram_slice} escrita\n') 
          # Undersampled unfiltered reconstructed image path
          us_unfiltered_img_path = us_unfiltered_dataset_folder+sinogram_slice+'.jpg'

          # Normalize 0-1 under sampled sinogram
          us_sinogram = self.normalize_image(us_sinogram)

          us_unfiltered_img = self.radon.backward(us_sinogram)*np.pi/self.number_projections_undersampled
          us_unfiltered_img = self.normalize_image(us_unfiltered_img)

          # Write undersampled filtered
          thumbs = cv2.imwrite(us_unfiltered_img_path,255.0*us_unfiltered_img.cpu().detach().numpy())
          print(thumbs)
        
        if write_fs_filtered == True:
          
          print('Slice {sinogram_slice} escrita\n')
          # Fully sampled filtered reconstructed image path
          fs_filtered_img_path = fs_filtered_dataset_folder+sinogram_slice+'.jpg'
          # Normalization of output sinogram - Fully sampled
          fs_filtered_img = self.radon.backward(self.radon.filter_sinogram(full_sinogram))
          fs_filtered_img = self.normalize_image(fs_filtered_img)
          # Write fully sampled filtered
          thumbs = cv2.imwrite(fs_filtered_img_path, 255.0*fs_filtered_img.cpu().detach().numpy())
          print(thumbs)

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
  
  def delete_registered_volume(self, sample):
    
    del self.registered_volume[sample]
  
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
      zeros_idx = np.linspace(0, self.number_projections_total, self.number_projections_undersampled, endpoint = False).astype(int)
      zeros_idx = (zeros_idx+rand_angle)%self.number_projections_total
      zeros_mask = np.full(self.number_projections_total, True, dtype = bool)
      zeros_mask[zeros_idx] = False
      undersampled_sinogram[zeros_mask, :, :] = 0
    
    return undersampled_sinogram
  
  @staticmethod
  def normalize_image(img):
    '''
    Normalizes images between 0 and 1.
    Params: 
      - img (ndarray): Image to normalize
    '''
    img = (img-img.min())/(img.max()-img.min())

    return img
  
  def _create_radon(self):
    '''
    Creates Torch-Radon method for the desired sampling and image size of the dataloader
    '''
    # Grab number of angles
    self.angles = np.linspace(0, 2*np.pi, self.number_projections_total, endpoint = False)
    
    self.radon = Radon(self.img_resize, self.angles, clip_to_circle = False, det_count = self.det_count)

# Multi-dataset to dataloader
class ZebraDataloader:
  '''
  List of tasks:
    1 - Load ZebraDatasets (all reconstructed)
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

    self.process_kwdictionary(kw_dictionary)
    
    for dataset_num, folder_path in enumerate(self.folder_paths):                                     
       
      # Loads dataset registered
      self.zebra_datasets[folder_path] = ZebraDataset(self.kw_dictionary_dataset)
  
  def __getitem__(self, folder_path):
      
    if folder_path not in self.zebra_datasets.keys():
      raise KeyError

    return self.zebra_datasets[folder_path]

  def process_kwdictionary(self, kw_dictionary):
    '''
    Load keyword arguments for dataloader
    Params:
     - kw_dictionary (dict): Dictionary containing keywords
    '''

    self.folder_paths = kw_dictionary['folder_paths']

    self.total_size = kw_dictionary['total_size']

    self.train_factor = kw_dictionary['train_factor']
    self.val_factor = kw_dictionary['val_factor']
    self.test_factor = kw_dictionary['test_factor']
    self.augment_factor = kw_dictionary['augment_factor']
    
    self.use_rand = kw_dictionary['use_rand']
    self.k_fold_datasets = kw_dictionary['k_fold_datasets']

    self.batch_size = kw_dictionary['batch_size']

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
                print('Dataset ya registrado')
                self.datasets_registered.append(registered_dataset_path)

            else:

                # Load corresponding registrations
                dataset.correct_rotation_axis(sample = sample, max_shift = 200, shift_step = 1)
                
                # Append volumes        
                print("Dataset {}/{} loaded - {} {}".format(dataset_num+1, len(self.folder_paths), str(dataset.folder_name), sample))
                
                # Resize registered volume to desired
                dataset.dataset_resize(sample)

                dataset.dataset_reconstruction()

                with open(registered_dataset_path, 'wb') as f:
                    
                  for image in dataset.registered_volume[sample]: 
                    
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

          if k_dataset < len(self.datasets_registered)- self.k_fold_datasets:
              print('Processing training/validation volumes\n')
              full_x.append(tX)
              full_y.append(tY)
              filt_full_x.append(filtX)

          else:
              print('Processing testing volumes\n')
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
                                      shuffle=False, 
                                      num_workers=0)

    # Form Datasets with folders



    # Dictionary reshape
    self.dataloaders = {'train':train_dataloader,        
                        'val': val_dataloader,
                        'test':test_dataloader}

  def _get_next_from_dataloader(self, set_name):
    '''
    Gets next item in dataloader.
    Params:
      - set_name (string): Refers to set of data where image comes from (train, val, test)
    '''

    return next(iter(self.dataloaders[set_name]))

  @staticmethod
  def normalize_image(image):

    return (image - image.min())/(image.max()-image.min())

class ReconstructionDataset(Dataset):
  
  def __init__(self, root_folder, acceleration_factor, transform = None):
    '''
    Params:
      - root_folder (string): root folder contains code for dataset + sample
      - acceleration_factor (int): acceleration factor 
    '''
    self.root_folder = root_folder+'/'
    self.acceleration_factor = str(acceleration_factor)
    self.transform = transform

    self.fs_filt_folder = 'fs_filtered/'
    self.us_filt_folder = 'us_{}_filtered/'.format(self.acceleration_factor)
    self.us_unfilt_folder = 'us_{}_unfiltered/'.format(self.acceleration_factor)

    self.unfiltered_us_recs_len = len([f for f in os.listdir(self.root_folder+self.us_unfilt_folder) if '.pt' in f])
    self.filtered_us_recs_len = len([f for f in os.listdir(self.root_folder+self.us_filt_folder) if '.pt' in f])
    self.filtered_fs_recs_len = len([f for f in os.listdir(self.root_folder+self.fs_filt_folder) if '.pt' in f])

    self.unfiltered_us_recs = torch.stack([torch.load(self.root_folder+self.us_unfilt_folder+str(index)+'.pt') for index in range(self.unfiltered_us_recs_len)], 0)
    self.filtered_us_recs = torch.stack([torch.load(self.root_folder+self.us_filt_folder+str(index)+'.pt') for index in range(self.filtered_us_recs_len)], 0)
    self.filtered_fs_recs = torch.stack([torch.load(self.root_folder+self.fs_filt_folder+str(index)+'.pt') for index in range(self.filtered_fs_recs_len)], 0)

  def __len__(self):

      return self.filtered_us_recs_len

  def __getitem__(self, index):
    '''
    Retrieves undersampled unfiltered reconstruction (unfiltered_us_rec), undersampled filtered reconstruction (filtered_us_rec) and fully sampled filtered reconstruction (filtered_fs_rec), used as Input, FBP benchmark and Output respectively. 
    '''

    unfiltered_us_rec = self.normalize_image(self.unfiltered_us_recs[index, ...])
    filtered_us_rec = self.normalize_image(self.filtered_us_recs[index, ...])
    filtered_fs_rec = self.normalize_image(self.filtered_fs_recs[index, ...])

    return (unfiltered_us_rec, filtered_us_rec, filtered_fs_rec)

  @staticmethod
  def normalize_image(image):

    return (image - image.min())/(image.max()-image.min())

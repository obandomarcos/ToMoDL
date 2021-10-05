"""
Utility functions for model parameter saving and loading
"""
import numpy as np
import torch
import time
import copy 
import datetime
import sys, os
from torch_radon import Radon, RadonFanbeam
import torchvision
import DataLoading as DL
import math
import matplotlib.pyplot as plt
import cv2 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Torch dataset tidying
def formRegDatasets(folder_paths, umbral, img_resize = 100, sample = 'head', experiment = 'Bassi'):
    
    train_dataset = []
    test_dataset = []
    disp_reg = []
    
    for dataset in folder_paths:    
      
      df = DL.ZebraDataset(dataset, 'Datasets', 'Bassi')
      # Cargo las registraciones correspondientes                                                      
      df.loadRegTransforms()
      # Agarro el valor de la registración para ese volumen 
      disp_reg.append(df.meanDisplacement)
      del df
    
    # Agarro el mayor valor de desplazamiento para cortar el volumen
    disp_reg = math.ceil(max(np.array([abs(d) for d in disp_reg])))
    print(disp_reg) 
    for dataset_num, dataset in enumerate(folder_paths):                                                   
      df = DL.ZebraDataset(dataset, 'Datasets', 'Bassi')
      print('Loading image for dataset {}'.format(df.folderName))                                      
      # Cargo el dataset
      df.loadImages(sample = 'head')
      # Cargo las registraciones correspondientes                                                      
      df.loadRegTransforms()
      # Aplico las transformaciones para este dataset                                               
      df.applyRegistration(sample = 'head')                                                         

      # Agarro los datos de la registración
      if abs(df.Tparams['Ty'].mean()) < umbral:                                                        
        
        print('Registration transformation {}'.format(df.Tparams['Ty'].mean()))
        # apendeo los volumenes (es lo unico que me importan, ya estan registrados)                        
        if dataset_num <= 2: 
            print("Loaded train dataset")
            
            dataset = df.getRegisteredVolume('head', margin = disp_reg//2, saveDataset = False, useSegmented = True)
            det_range = np.linspace(0, dataset.shape[1], int((img_resize+0.5)*np.sqrt(2)), endpoint = False).astype(int)
            dataset = dataset[:,det_range,:]
            train_dataset.append(dataset)                                                

        # Borro el dataframe                                                                               
        else:
            print("Loaded test dataset")
            dataset = df.getRegisteredVolume('head', margin = disp_reg//2, saveDataset = False, useSegmented = True)
            det_range = np.linspace(0, dataset.shape[1], int((img_resize+0.5)*np.sqrt(2)), endpoint = False).astype(int)
            dataset = dataset[:,det_range,:]
            test_dataset.append(dataset)

      del df

    return train_dataset, test_dataset

def formMaskDataset(sino_dataset, dataset_size, slice_idx, num_beams):
    '''
    Forms a dataset of images out of different subsamplings applied to a volume
    params:
        sino_dataset (ndarray): sinogram volume
        dataset_size (int): number of images
        slice_num (int): slice index to be grabbed
        projection_num (int): 
    ''' 
    target_img = sino_dataset[:,:,slice_idx]
    train_dataset = np.repeat(target_img[...,None], dataset_size, axis = 2)

    for i in range(dataset_size):
        
        # choose random projections 
        rand = np.random.choice(range(sino_dataset.shape[0]), sino_dataset.shape[0]-num_beams,replace=False)
        # clamp to zero non desired lines
        train_dataset[rand, :, i] = 0
    
    return train_dataset, target_img

def formUniqueDataset(sino_datasets, dataset_size, num_beams, slice_idx):
    """
    Routine for training with a unique image. Returns a training dataset with random subsampling (num_beams used for reconstruction), a target image and the maximum number of projections taken.
    params:
        sino_dataset (ndarray): sinogram volume
        dataset_size (int): number of images in training dataset
        num_beams (int): number of beams
        slice_idx (int): slice index 
        img_resize (int): image resizing, squared
    """
    # Choose one of the datasets
    sino_dataset = sino_datasets[np.random.choice(range(len(sino_datasets)), 1).astype(int)[0]]
    
    # Obtain masked datasets
    train_dataset, target_img = formMaskDataset(sino_dataset, dataset_size, slice_idx, num_beams)

    # Resize sinogram
    n_angles = sino_dataset.shape[0]
    angles_full = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    
    det_count = train_dataset.shape[1]
    image_size = int(det_count/np.sqrt(2)+0.5)
        
    radon = Radon(image_size, angles_full, clip_to_circle = False, det_count = det_count)
    
    training_Atb = []

    for img in np.rollaxis(train_dataset, 2):
    
        # Normalization
        #img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img = radon.backward(radon.filter_sinogram(torch.FloatTensor(img).to(device)))
        mn = torch.min(img)
        mx = torch.max(img)
        norm = (img-mn)*(1.0/(mx-mn))
        
        training_Atb.append(norm)

    #target_img = cv2.normalize(target_img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    
    target_img = radon.backward(radon.filter_sinogram(torch.FloatTensor(target_img).to(device)))
    mn = torch.min(target_img)
    mx = torch.max(target_img)
    target_img = (target_img-mn)*(1.0/(mx-mn)) 

    training_Atb = torch.unsqueeze(torch.stack(training_Atb), 1)
    
    # Image resize
#    transform = torch.nn.Sequential(torchvision.transforms.Normalize([0.5], [0.5]))
    
    #with torch.no_grad():
    #    
    #    train_Atb_sub = training_Atb - training_Atb.min(0, keepdim = True)[0]
    #    train_Atb_sub /= training_Atb.max(0, keepdim = True)[0]
    #    target_img_sub = target_img - target_img.min(0, keepdim = True)[0]
    #    target_img_sub /= target_img.max(0, keepdim = True)[0]
    #
    #transform = torch.jit.script(transform)
    #training_Atb = transform.forward(train_Atb_sub)
    #target_img = transform.forward(torch.unsqueeze(target_img_sub, 0))
    
    return training_Atb, target_img, n_angles

def maskDatasets(full_sino, num_beams, dataset_size, img_size, angle_seed = 0):
    '''
    Evolution of form datasets, masking them fully 
    '''
    # Subsample detector range RESIZE IN REGDATASETS
    det_count = int((img_size+0.5)*np.sqrt(2))
    #det_range = np.linspace(0, full_sino.shape[1], det_count, endpoint = False).astype(int)
    #full_sino = full_sino[:, det_range, :]
    undersampled_sino = np.copy(full_sino)

    # Clamp to zero projections, 
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
    desired = []
 
    # Grab random slices
    assert(dataset_size <= full_sino.shape[2])
    rand = np.random.choice(range(full_sino.shape[2]), dataset_size, replace=False)
    
    # Normalize
    for img in np.rollaxis(undersampled_sino[:,:,rand], 2):
        
        img = radon.backward(radon.filter_sinogram(torch.FloatTensor(img).to(device)))
        mn = torch.min(img)
        mx = torch.max(img)
        norm = (img-mn)*(1.0/(mx-mn))
        
        undersampled.append(norm)

    for img in np.rollaxis(full_sino[:,:,rand], 2):

        img = radon.backward(radon.filter_sinogram(torch.FloatTensor(img).to(device)))
        mn = torch.min(img)
        mx = torch.max(img)
        norm = (img-mn)*(1.0/(mx-mn))
        
        desired.append(norm)

    desired = torch.unsqueeze(torch.stack(desired), 1)
    undersampled = torch.unsqueeze(torch.stack(undersampled), 1)
    
    return desired, undersampled


def formDataloaders(train_dataset, test_dataset, number_projections, train_size, val_size, test_size, batch_size, img_size, augment_factor = 1):
    """
    Form torch dataloaders for training and testing, full and undersampled
    params:

        - train_dataset and test dataset are a list of volumes containing the sinograms to be processed into images
        - number_projections is the number of projections the sinogram is reconstructed with for undersampled FBP
        - train size (test_size) is the number of training (test) images to be taken from the volumes. They are taken randomly using formDatasets
        - batch_size is the number of images a batch contains in the dataloader
        -img_size is the size of the new images to be reconstructed
    """
    
    trainX = []
    trainY = []
    testX = []
    testY = []
    valX = []
    valY = []
    
    # Augment factor iterates over the datasets for data augmentation
    for i in range(augment_factor):

        rand_angle = np.random.randint(0, number_projections)

        # Dataset train
        for dataset in train_dataset:
    
            l = len(train_dataset)
            tY, tX = maskDatasets(dataset, number_projections, (train_size+val_size)//l, img_size, rand_angle)
            trainX.append(tX)
            trainY.append(tY)
    
        for dataset in test_dataset:
            
            l = len(test_dataset)
            tY, tX = maskDatasets(dataset, number_projections, test_size//l, img_size, rand_angle)
            testX.append(tX)
            testY.append(tY)

    # Build dataloaders
    trainX = torch.vstack(trainX)
    trainY = torch.vstack(trainY)
    testX = torch.vstack(testX)
    testY = torch.vstack(testY) 

    valX = torch.clone(trainX[:int(augment_factor*val_size),...])
    valY = torch.clone(trainY[:int(augment_factor*val_size),...])
    
    trainX = torch.clone(trainX[int(augment_factor*val_size):,...])
    trainY = torch.clone(trainY[int(augment_factor*val_size):,...])
    
    trainX = torch.utils.data.DataLoader(trainX,
                                          batch_size=batch_size,
                                          shuffle=False, num_workers=0)
    trainY = torch.utils.data.DataLoader(trainY,                                                    
                                      batch_size=batch_size,                                           
                                      shuffle=False, num_workers=0)                                    

    testX = torch.utils.data.DataLoader(testX, batch_size=1,
                                                shuffle=False, num_workers=0)
    testY = torch.utils.data.DataLoader(testY, batch_size=1,
                                                shuffle=False, num_workers=0)
    
    valX = torch.utils.data.DataLoader(valX, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
    valY = torch.utils.data.DataLoader(valY, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)

    dataloaders = {'train':{'x':trainX, 'y':trainY}, 'val':{'x':valX, 'y':valY}, 'test':{'x':testX,'y':testY}}

    return dataloaders

def unique_model_training(model, criterion, criterion_fbp, optimizer, dataloaders, target_image, num_epochs, device, batch_size, disp = True):
    
    # configure labels 
    labels = torch.unsqueeze(target_image, 0).repeat(batch_size, 1, 1, 1)
    since = time.time()
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    train_info = {}
    train_info['train'] = []
    train_info['train_fbp'] = []
    
    train_info['val'] = []
    train_info['val_fbp'] = []
    
    for epoch in range(num_epochs):
        
        prev_time = time.time()
                                        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
                                        
            running_loss = 0.0
            running_std_loss = 0.0
                                        
            fbp_loss = 0.0
            fbp_std_loss = 0.0

            for batch_i, inputs in enumerate(dataloaders[phase]):
                                                                                           
               inputs.to(device)
               labels.to(device)
                                                                                           
               optimizer.zero_grad() #zero the parameter gradients
                
               #forward pass
               # Track history in training only
               with torch.set_grad_enabled(phase=='train'):
                   
                   outputs = model(inputs)

                   loss = criterion(outputs['dc'+str(model.K)], labels) 
                   loss_fbp = criterion_fbp(inputs, labels)
              
                   if phase == 'train':
                    
                       loss.backward()
                       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type =2.0)
                       optimizer.step()

               running_loss += loss.item()*inputs.size(0) 
               fbp_loss += loss_fbp.item()*inputs.size(0)
                                                                                                    
               if torch.cuda.is_available():
                   torch.cuda.empty_cache()
                                                                                                    
               if disp:
                                                                                                    
                   batches_done = epoch * len(dataloaders[phase]) + batch_i
                   batches_left = num_epochs * len(dataloaders[phase]) - batches_done
                   time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
               #    print(time.time()-prev_time)
                   prev_time = time.time()
                   
                   sys.stdout.write(
                           "\r[%s] [Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s "
                           % (
                               phase,
                               epoch+1,
                               num_epochs,
                               batch_i+1,
                               len(dataloaders[phase]),
                               loss.item(),
                               time_left,
                           )
                       )
            epoch_loss = running_loss/len(dataloaders[phase])
            epoch_loss_fbp = fbp_loss/len(dataloaders[phase])

            train_info[phase].append(epoch_loss)
            train_info[phase+'_fbp'].append(epoch_loss_fbp)

            if disp:
                print('')
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                print('{} Loss FBP: {:.4f} '.format(phase, epoch_loss_fbp))

            if phase == 'val' and epoch_loss < best_loss:
                
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
#        checkpoint_plot(outputs, root, epoch)
    
    time_elapsed = time.time()-since                                                                              
    if disp:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model , train_info

# Training function
def model_training(model, criterion, crit_fbp, optimizer, dataloaders, device, root, num_epochs = 25, disp=False, do_checkpoint = 0):
    """
    Trains pytorch model
    """
 
    since = time.time()
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
 
    train_x = dataloaders['train']['x']   
    train_y = dataloaders['train']['y']
    val_x = dataloaders['val']['x']
    val_y = dataloaders['val']['y']
    
    train_info = {}
    train_info['train'] = []
    train_info['train_fbp'] = []
    
    train_info['val'] = []
    train_info['val_fbp'] = []

    loss_std = []
    loss_std_fbp = []

    for epoch in range(num_epochs):
        prev_time = time.time()
        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
 
            running_loss = 0.0
            running_std_loss = 0.0

            fbp_loss = 0.0
            fbp_std_loss = 0.0
 
            for batch_i, (inputs, labels) in enumerate(zip(*dataloaders[phase].values())):

               inputs.to(device)
               labels.to(device) 

               optimizer.zero_grad() #zero the parameter gradients 

               #forward pass
               # Track history in training only
               with torch.set_grad_enabled(phase=='train'):
                   
                   outputs = model(inputs)
                   loss = criterion(outputs['dc'+str(model.K)], labels)
                   loss_fbp = crit_fbp(inputs, labels)

                   if phase == 'train':
                       
                       loss.backward()
                       #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type =2.0)
                       optimizer.step()
               
               if (epoch in [0, num_epochs-1]) and (batch_i in [0, 10, 20]):
                                                                                                         
                   print('Plotted {}'.format(phase))
                   plot_outputs(labels, outputs, root+'{}_images_epoch{}_proj{}_batch{}_K{}.pdf'.format(phase, epoch, model.proj_num, batch_i, model.K))

               # los desvios se pueden sumar en cuadratura
               running_loss += loss.item()*inputs.size(0)
               #running_std_loss += loss_std*inputs.size(0)

               fbp_loss += loss_fbp.item()*inputs.size(0)
               #fbp_std_loss += loss_std_fbp*inputs.size(0)      

               if torch.cuda.is_available():
                   torch.cuda.empty_cache()
                   del inputs, outputs

               if disp:

                   batches_done = epoch * len(dataloaders[phase]['x']) + batch_i
                   batches_left = num_epochs * len(dataloaders[phase]['x']) - batches_done
                   time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
               #    print(time.time()-prev_time)
                   prev_time = time.time()
                   
                   sys.stdout.write(
                           "\r[%s] [Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s "
                           % (
                               phase,
                               epoch+1,
                               num_epochs,
                               batch_i+1,
                               len(dataloaders[phase]['x']),
                               loss.item(),
                               time_left,
                           )
                       )
                
            epoch_loss = running_loss/len(dataloaders[phase]['x'])
            epoch_loss_fbp = fbp_loss/len(dataloaders[phase]['x'])

            train_info[phase].append(epoch_loss)

            train_info[phase+'_fbp'].append(epoch_loss_fbp)

            if disp:
                print('')
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                print('{} Loss FBP: {:.4f} '.format(phase, epoch_loss_fbp))

            if phase == 'val' and epoch_loss < best_loss:
                
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
#        checkpoint_plot(outputs, root, epoch)

        if do_checkpoint>0:
            if epoch%do_checkpoint==0:
                 checkpoint(root, epoch, model)        

    time_elapsed = time.time()-since
    
    if disp:
        
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model , train_info

def plot_data(x, y, root):

    fig, ax = plt.subplots(1,2)

    ax[0].imshow(x[0,0,:,:].detach().cpu().numpy())
    ax[1].imshow(y[0,0,:,:].detach().cpu().numpy())
    ax[0].set_title('X')
    ax[1].set_title('Y')

    fig.savefig(root)

def checkpoint_plot(outputs, root, epoch):
    # outputs

    fig, axs = plt.subplots(1,len(outputs))
    
    for out, key, ax in zip(outputs.values(), outputs.keys(), axs):
        
        ax.set_title(key)
        ax.imshow(out[0,0,:,:].detach().cpu().numpy())
        ax.axis('off')
        
    print('Plot saved in {}'.format(root)) 
    
    fig.savefig(root+'test_modl_internal_epoch{}.pdf'.format(epoch), bbox_inches = 'tight', pad_inches = 0)

def checkpoint(root, epoch, model):
    """ Saves the dictionaries of a given pytorch model for
        the right epoch
        """
    model_out_path = "model_epoch_{}.pth".format(epoch)
    model_out_path = root + model_out_path;
    torch.save(model.state_dict() , model_out_path);
    print("Checkpoint saved to {}".format(model_out_path))

def save_net(title, model):
    """Saves dictionaries of a given pytorch model in the place defined by
        title
        """
    model_out_path = "{}.pth".format(title)
    model_out_path = model_out_path;
    torch.save(model.state_dict(), model_out_path);
    print("Model Saved")


def load_net(title, model, device):
    """Loads net defined by title """
    model_out_path = "{}.pth".format(title)
    model.load_state_dict(torch.load(model_out_path, map_location=torch.device(device)))
    print("Model Loaded: {}".format(title))

def plot_outputs(target, prediction, path):
    
    fig, ax = plt.subplots(1, len(prediction.keys())+1, figsize = (16,6))
    
    ax[0].imshow(target.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
    ax[0].set_title('Target')
    ax[0].axis('off')

    for a, (key, image) in zip(ax[1:], prediction.items()):

        im = a.imshow(image.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
        a.set_title(key)
        a.axis('off')
    
    cax = fig.add_axes([a.get_position().x1+0.01,a.get_position().y0,0.02,a.get_position().height])
    plt.colorbar(im, cax = cax)
    fig.savefig(path, bbox_inches = 'tight')

def psnr(img_size, mse, batch):
    
    mse = np.array(mse)
    return 10*np.log10(1.0*batch/(mse/(img_size*img_size)))

def plot_histogram(dictionary, img_size, path):
    """
    Plot histograms 
    """
    fig, ax = plt.subplots(1, 2, figsize = (6,6))

    ax[0].hist(dictionary['loss_net'], color = 'orange')
    ax[0].grid(True)
    ax[0].set_xlabel('MSE Network')

    ax[1].hist(dictionary['loss_fbp'])
    ax[1].grid(True)
    ax[1].set_xlabel('MSE FBP')
    
    fig.savefig(path)

# Deprecated
def formDatasets(sino_dataset, num_beams, size, img_size):
    
    """
    This function receives a sinogram dataset and returns two FBP reconstructed datasets,
    ocan't multiply sequence by non-int of type 'floatne with the full span of angles and one reconstructed with num_beams
    """ 
    # Angles
    n_angles = sino_dataset.shape[0]
    angles_full = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles_under = np.linspace(0, 2*np.pi, num_beams, endpoint=False)
  
    projs = np.linspace(0, n_angles, num_beams, endpoint=False).astype(int)
  
    det_count = sino_dataset.shape[1]
    image_size = int(det_count/np.sqrt(2)-0.5)
    # Datasets undersampled with projs
    under_sino = sino_dataset[projs,:,:]
  
    rad_full = Radon(image_size, angles_full, clip_to_circle=False, det_count=det_count)
    rad_under = Radon(image_size, angles_under, clip_to_circle=False, det_count=det_count)
  
    undersampled = []
    desired = []
    # retrieve random z-slices to train
    rand = np.random.choice(range(under_sino.shape[2]), size, replace=False)
  
    # Undersampled
    for img in np.rollaxis(under_sino[:,:,rand], 2):
  
      undersampled.append(rad_under.backward(rad_under.filter_sinogram(torch.FloatTensor(img).to(device))))
  
    # Full
    for img in np.rollaxis(sino_dataset[:,:,rand], 2):
  
      desired.append(rad_full.backward(rad_full.filter_sinogram(torch.FloatTensor(img).to(device))))
  
    desired = torch.unsqueeze(torch.stack(desired), 1)
    undersampled = torch.unsqueeze(torch.stack(undersampled), 1)
    
    # Image resize
    transform = torch.nn.Sequential(torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.Normalize([0.5], [0.5]))
    
    transform = torch.jit.script(transform)
    desired = transform.forward(desired)
    undersampled = transform.forward(undersampled)
  
    return desired, undersampled

def model_training_unet(model, criterion, crit_fbp, optimizer, dataloaders, device, root, num_epochs = 25, disp=False, do_checkpoint = 0):
    """
    Trains pytorch model
    """
 
    since = time.time()
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
 
    train_x = dataloaders['train']['x']   
    train_y = dataloaders['train']['y']
    val_x = dataloaders['val']['x']
    val_y = dataloaders['val']['y']
    
    train_info = {}
    train_info['train'] = []
    train_info['train_fbp'] = []
    
    train_info['val'] = []
    train_info['val_fbp'] = []

    loss_std = []
    loss_std_fbp = []

    for epoch in range(num_epochs):
        prev_time = time.time()
        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
 
            running_loss = 0.0
            running_std_loss = 0.0

            fbp_loss = 0.0
            fbp_std_loss = 0.0
 
            for batch_i, (inputs, labels) in enumerate(zip(*dataloaders[phase].values())):

               inputs.to(device)
               labels.to(device) 

               optimizer.zero_grad() #zero the parameter gradients 

               #forward pass
               # Track history in training only
               with torch.set_grad_enabled(phase=='train'):
                   
                   outputs = model(inputs)
                   loss = criterion(outputs, labels)
                   loss_fbp = crit_fbp(inputs, labels)

                   if phase == 'train':
                       
                       loss.backward()
                       #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type =2.0)
                       optimizer.step()

               # los desvios se pueden sumar en cuadratura
               running_loss += loss.item()*inputs.size(0)
               #running_std_loss += loss_std*inputs.size(0)

               fbp_loss += loss_fbp.item()*inputs.size(0)
               #fbp_std_loss += loss_std_fbp*inputs.size(0)      

               if torch.cuda.is_available():
                   torch.cuda.empty_cache()
                   del inputs, outputs

               if disp:

                   batches_done = epoch * len(dataloaders[phase]['x']) + batch_i
                   batches_left = num_epochs * len(dataloaders[phase]['x']) - batches_done
                   time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
               #    print(time.time()-prev_time)
                   prev_time = time.time()
                   
                   sys.stdout.write(
                           "\r[%s] [Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s "
                           % (
                               phase,
                               epoch+1,
                               num_epochs,
                               batch_i+1,
                               len(dataloaders[phase]['x']),
                               loss.item(),
                               time_left,
                           )
                       )
                
            epoch_loss = running_loss/len(dataloaders[phase]['x'])
            epoch_loss_fbp = fbp_loss/len(dataloaders[phase]['x'])

            train_info[phase].append(epoch_loss)

            train_info[phase+'_fbp'].append(epoch_loss_fbp)

            if disp:
                print('')
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                print('{} Loss FBP: {:.4f} '.format(phase, epoch_loss_fbp))

            if phase == 'val' and epoch_loss < best_loss:
                
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
#        checkpoint_plot(outputs, root, epoch)

        if do_checkpoint>0:
            if epoch%do_checkpoint==0:
                 checkpoint(root, epoch, model)        

    time_elapsed = time.time()-since
    
    if disp:
        
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model , train_info

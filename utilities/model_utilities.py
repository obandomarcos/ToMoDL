"""
Utility functions for model parameter saving and loading

author: obanmarcos
"""
import numpy as np
import torch
import time
import copy 
import datetime
import sys, os
from torch_radon import Radon, RadonFanbeam
import torchvision
import utilities.dataloading_utilities as DL
import math
import matplotlib.pyplot as plt
import cv2 
import torchvision.transforms as T
import pickle
import os.path
import os
from tqdm import tqdm
# import albumentations
import torch.nn.functional as nnf

os.chdir('/home/obanmarcos/Balseiro/DeepOPT/')
from utilities.folders import *

def model_training_unet(model, criterion, crit_fbp, optimizer, dataloaders, device, root, num_epochs = 25, disp=False, do_checkpoint = 0, plot_title = False,title = '', compute_mse = True, monai = True):
    """
    Training routine for raw Unet
    Params:
        - model (nn.Module): PyTorch model to be trained
        - criterion (nn.loss): Objective function to minimize. Compares network output with target image
        - optimizer (nn.optimizer): Optimizer to train the network. currently using Adam optimizer
        - dataloaders (dict): Dictionary of dataloaders, with train and val datasets
        - device: GPU device currenly being used
        - root (string): Folder path to save results
        - num_epochs (int): number of epochs to run the training
        - disp (bool): display training progress
        - do_checkpoint (int): checkpoint every do_checkpoint epoch
        - title (string): plot title
    """
    since = time.time()
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    if compute_mse == True:
        mse_loss = torch.nn.MSELoss(reduction = 'sum')

    train_x = dataloaders['train']['filtX']
    train_y = dataloaders['train']['y']
    
    val_x = dataloaders['val']['filtX']
    val_y = dataloaders['val']['y']

    train_info = {}
    train_info['train'] = []
    train_info['train_fbp'] = []
    train_info['train_mse'] = []

    train_info['val'] = []
    train_info['val_fbp'] = []
    train_info['val_mse'] = []

    for epoch in range(num_epochs):
        
        prev_time = time.time()
        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
 
            running_loss = 0.0    
            fbp_loss = 0.0
            mse_loss_count = 0.0

            for batch_i, (_, inputs, labels) in enumerate(zip(*dataloaders[phase].values())):
                
                inputs.to(device)
                labels.to(device) 

                optimizer.zero_grad() #zero the parameter gradients 
                
                if monai == True:
                    inputs = nnf.interpolate(inputs, size=(model.imageSize, model.imageSize), mode='bicubic', align_corners=False)           
                    labels = nnf.interpolate(labels, size=(model.imageSize, model.imageSize), mode='bicubic', align_corners=False)           
                
                # Track history in training only
                with torch.set_grad_enabled(phase=='train'):
                     
                    output = model(inputs)
                    
                    loss = criterion(output, labels)
                    loss_fbp = crit_fbp(inputs, labels)
                    
                    if compute_mse == True:
                        mse = mse_loss(output, labels) 

                    if phase == 'train':
                            
                        if (np.isnan(loss.item())):
                            # Solución provisoria
                            print('FOUND A NAN IN BATCH {}'.format(batch_i))
                            continue

                        loss.backward()
                        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type =2.0)
                        optimizer.step()
                
                running_loss += loss.item()*inputs.size(0)
                fbp_loss += loss_fbp.item()*inputs.size(0)

                if compute_mse == True:
                    mse_loss_count += mse.item()*inputs.size(0)        

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if disp:

                    batches_done = epoch * len(dataloaders[phase]['x']) + batch_i
                    batches_left = num_epochs * len(dataloaders[phase]['x']) - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                
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
            
            epoch_loss = running_loss/(len(dataloaders[phase]['x'])*inputs.size(0))
            epoch_loss_fbp = fbp_loss/(len(dataloaders[phase]['x'])*inputs.size(0))

            train_info[phase].append(epoch_loss)
            train_info[phase+'_fbp'].append(epoch_loss_fbp)
 
            if compute_mse == True:
                epoch_loss_mse = mse_loss_count/(len(dataloaders[phase]['x'])*inputs.size(0))
                train_info[phase+'_mse'].append(epoch_loss_mse)

            if disp:
            
                print('')
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                print('{} Loss FBP: {:.4f} '.format(phase, epoch_loss_fbp))
            
            if phase == 'val' and epoch_loss < best_loss:
                
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
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

# Training function
def model_training(model, criterion, crit_backproj, crit_fbp, optimizer, dataloaders, device, root, num_epochs = 25, disp=False, do_checkpoint = 0, plot_title = False,title = '', compute_mse = True, monai = False, compute_ssim = False):
    """
    Training routine for model
    Params:
        - model (nn.Module): PyTorch model to be trained
        - criterion (nn.loss): Objective function to minimize. Compares network output with target image
        - crit_fbp (nn.loss): Objective function between inputs of the network (A^tb) and output
        - optimizer (nn.optimizer): Optimizer to train the network. currently using Adam optimizer
        - dataloaders (dict): Dictionary of dataloaders, with train and val datasets
        - device: GPU device currenly being used
        - root (string): Folder path to save results
        - num_epochs (int): number of epochs to run the training
        - disp (bool): display training progress
        - do_checkpoint (int): checkpoint every do_checkpoint epoch
        - title (string): plot title
    """
 
    since = time.time()
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    if compute_mse == True:
        mse_loss = torch.nn.MSELoss(reduction = 'sum')

    train_x = dataloaders['train']['x']   
    filtTrain_x = dataloaders['train']['filtX']
    train_y = dataloaders['train']['y']
    
    val_x = dataloaders['val']['x']
    filtVal_x = dataloaders['val']['filtX']
    val_y = dataloaders['val']['y']
    
    train_info = {}
    train_info['train'] = []
    train_info['train_backproj'] = []
    train_info['train_fbp'] = []
    train_info['train_norm'] = []
    train_info['train_mse'] = []

    train_info['val'] = []
    train_info['val_backproj'] = []
    train_info['val_fbp'] = []
    train_info['val_norm'] = []
    train_info['val_mse'] = []

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
            fbp_loss = 0.0
            backproj_loss = 0.0
            norm_loss = 0.0
            mse_loss_count = 0.0

            for batch_i, (inputs, inputs_fbp, labels) in enumerate(zip(*dataloaders[phase].values())):

               inputs.to(device)
               inputs_fbp.to(device)
               labels.to(device) 

               if monai == True:
                   inputs = nnf.interpolate(inputs, size=(model.imageSize, model.imageSize), mode='bicubic', align_corners=False)           
                   labels = nnf.interpolate(labels, size=(model.imageSize, model.imageSize), mode='bicubic', align_corners=False)           
                   inputs_fbp = nnf.interpolate(inputs_fbp, size=(model.imageSize, model.imageSize), mode='bicubic', align_corners=False)           
               optimizer.zero_grad() #zero the parameter gradients 
               
               # Track history in training only
               with torch.set_grad_enabled(phase=='train'):
                    
                   outputs = model(inputs)

                   if compute_ssim == True: 
            
                       loss = 1-criterion(outputs['dc'+str(model.K)], labels)
                       loss_backproj = 1-crit_backproj(inputs, labels)
                       loss_fbp = 1-crit_fbp(inputs_fbp, labels)
                   
                   else:

                       loss = criterion(outputs['dc'+str(model.K)], labels)
                       loss_backproj = crit_backproj(inputs, labels)
                       loss_fbp = crit_fbp(inputs_fbp, labels)


                   if compute_mse == True:

                       mse = mse_loss(outputs['dc'+str(model.K)], labels) 

                   # Output normalization
                   loss_norm = psnr_normalize(outputs['dc'+str(model.K)], labels)
                    
                   if phase == 'train':
                        
                       if (np.isnan(loss.item())):
                           # Solución provisoria
                           print('FOUND A NAN IN BATCH {}'.format(batch_i))
                           continue

                       loss.backward()
                       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type =2.0)
                       optimizer.step()
               
               # Plot images
               if (plot_title == True) and (epoch in [0, num_epochs-1]) and (batch_i in [0, 10, 20]):                                                                                           
                   print('Plotted {}'.format(phase))
                   path_plot = '{}_images_epoch{}_proj{}_batch{}_K{}_lam{}.pdf'.format(phase, epoch, model.proj_num, batch_i, model.K, model.lam.data)
                   title_plot = title+'{} images epoch{} proj{} batch{} K{} lam{}'.format(phase, epoch, model.proj_num, batch_i, model.K, model.lam.data)
                   plot_outputs(labels, outputs, root+path_plot, title_plot)
                

               running_loss += loss.item()*inputs.size(0)
               backproj_loss += loss_backproj.item()*inputs.size(0)
               fbp_loss += loss_fbp.item()*inputs.size(0)
               norm_loss += loss_norm.item()*inputs.size(0)

               if compute_mse == True:
                   mse_loss_count += mse.item()*inputs.size(0)        
    
               if torch.cuda.is_available():
                   torch.cuda.empty_cache()

               if disp:

                   batches_done = epoch * len(dataloaders[phase]['x']) + batch_i
                   batches_left = num_epochs * len(dataloaders[phase]['x']) - batches_done
                   time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
               
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
                
            epoch_loss = running_loss/(len(dataloaders[phase]['x'])*inputs.size(0))
            epoch_loss_fbp = fbp_loss/(len(dataloaders[phase]['x'])*inputs.size(0))
            epoch_loss_backproj = backproj_loss/(len(dataloaders[phase]['x'])*inputs.size(0))
            epoch_loss_norm = norm_loss/(len(dataloaders[phase]['x'])*inputs.size(0))
            
            train_info[phase].append(epoch_loss)
            train_info[phase+'_fbp'].append(epoch_loss_fbp)
            train_info[phase+'_backproj'].append(epoch_loss_backproj)
            train_info[phase+'_norm'].append(epoch_loss_norm)
            
            if compute_mse == True:
                epoch_loss_mse = mse_loss_count/(len(dataloaders[phase]['x'])*inputs.size(0))
                train_info[phase+'_mse'].append(epoch_loss_mse)

            if disp:
                print('')
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                print('{} Loss FBP: {:.4f} '.format(phase, epoch_loss_fbp))
                print('{} Loss Backprojection: {:.4f} '.format(phase, epoch_loss_backproj))
                print('{} Loss Norm: {:.4f}'.format(phase, epoch_loss_norm))

            if phase == 'val' and epoch_loss < best_loss:
                
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

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


def psnr_normalize(outputs, target):
    """
    Adjusts outputs mean to target mean and calculates PSNR
    """
    norm_loss = torch.nn.MSELoss(reduction = 'sum')
    
    norm_output = torch.zeros_like(outputs)
    norm_target = torch.zeros_like(target)
    
    for i, (out, tar) in enumerate(zip(outputs, target)):

        norm_output[i,...] = (out-out.min())/(out.max()-out.min())
        norm_target[i,...] = (tar-tar.min())/(tar.max()-tar.min())
    
    #print('norm output max', norm_output.max(), ' min ', norm_output.min())
    #print('norm output min', norm_target.max(), ' min ', norm_target.min())
    loss = norm_loss(norm_output, norm_target)
    
    return loss

def plot_data(x, y, root):

    fig, ax = plt.subplots(1,2)

    ax[0].imshow(x[0,0,:,:].detach().cpu().numpy())
    ax[1].imshow(y[0,0,:,:].detach().cpu().numpy())
    ax[0].set_title('X')
    ax[1].set_title('Y')

    fig.savefig(root)

def plot_sample(X, title):

    fig, axs = plt.subplots(1,3)
    
    rand = np.random.choice(range(X.shape[0]), 3, replace=False)

    for i, ax in zip(rand, axs):

        ax.imshow(X.detach().cpu().numpy()[i,0,:,:], cmap = 'gray')

    fig.savefig(results_folder+title+'.pdf')

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

def plot_outputs(target, prediction, path, title = ''):
    
    fig, ax = plt.subplots(1, len(prediction.keys())+1, figsize = (16,6))
    
    im = ax[0].imshow(target.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
    ax[0].set_title('Target')
    ax[0].axis('off') 
    plt.suptitle(title)
    #cax = fig.add_axes([ax[0].get_position().x1+0.01,ax[0].get_position().y0,0.02,ax[0].get_position().height])
    #plt.colorbar(im, cax = cax)

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
                       #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type =2.0)
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

# For unique dataset sampling
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

# Deprecated
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

def mse(img1, img2):

    return ((img1 - img2)**2).sum()

def k_fold_list(l, k_elems):
    
    for _ in range(k_elems):
        l = [l.pop()]+l
    return l


"""
Function for training Unet to remove streaks from undersampled OPT reconstructions
    
train_input: path to folder with sinograms for training

test_input: path to folder with sinograms for testing

savename: file name for network to save to

number_angles: number of angles to be used for undersampled sinogram
"""
import torch
import torch.nn as nn
from OPT_dataset import OPT_dataset
import numpy as np
from unet_model_original import UNet
import copy
import matplotlib.pyplot as plt
import os

def train_unet_OPT(train_input,test_input,savename,number_angles):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # OPT dataset
    train_dataset = OPT_dataset(input_folder = train_input, crop_size = [128,128], augment = True, number_angles = number_angles)
    test_dataset = OPT_dataset(input_folder = test_input, crop_size = [512,512], augment = True, number_angles = number_angles)    
    
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=8, 
                                              shuffle=False)
    # create network    
    model = UNet(1,1).to(device)
    
    # Hyper-parameters
    num_epochs = 60
    learning_rate = 0.001    
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    
    # track train error
    epoch_steps = total_step*num_epochs
    train_error = np.zeros(epoch_steps)
    ii = 0
    
    # track validation error
    validate_steps = len(test_loader)*num_epochs
    validate_error = np.zeros(validate_steps)
    iii = 0
    
    # set starting point for early stopping
    last_error = 0.5
    fail_count = 0
    
    # training loop
    model.train()
    for epoch in range(num_epochs):
        # loop over all sinograms
        for i, (images, labels, I_max, I_min, I_mean) in enumerate(train_loader):
            # Move images to GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)            
            loss =  criterion(outputs, labels)
            
            # Backward and optimize           
            loss.backward()
            optimizer.step()
            train_error[ii] = loss.item()
            ii = ii + 1
            
            
            if (i) % 10 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        # plot training error
        plt.plot(train_error[:ii])
        plt.xlabel('steps')
        plt.ylabel('training l1 error')
        plt.show()
                
        # evaluate current network on validation set for early stopping
        model.eval()
        with torch.no_grad():
            error = 0
            count = 0
            for images, labels, I_max, I_min, I_mean in test_loader:
                count = count + 1
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                error = error + loss.item()
                validate_error[iii] = loss.item()
                iii = iii + 1
                
            error = error/count
            print('Current error: {}, Last error: {}'.format(error, last_error))
        
        # plot validation error
        plt.plot(validate_error[:iii])
        plt.xlabel('steps')
        plt.ylabel('validation l1 error')
        plt.show()
        
        # check if network hasn't been getting better
        if epoch > 0 and error > last_error:
            fail_count = fail_count + 1
            if fail_count == 3:
                break
        else:
            # reset early stopping clock and save current network
            fail_count = 0
            last_error = copy.copy(error)
            torch.save(model.state_dict(), savename)
        
        model.train() 
        
        # Decay learning rate
        if (epoch+1) % 10 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

if __name__ == "__main__":
    
    dirname = os.path.dirname(__file__)
    train_input = os.path.join(dirname, 'trainingData/train')
    test_input = os.path.join(dirname, 'trainingData/validate')
    save_name = os.path.join(dirname, 'trainedNetworks/Unet_OPT_40_angles')
    number_angles = 40
    
    train_unet_OPT(train_input, test_input, save_name, number_angles)    

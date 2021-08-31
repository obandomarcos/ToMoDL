'''
Test: Model training 
author: obanmarcos 
'''
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import phantominator as ph
import torchvision

f140114_5dpf = "/home/marcos/DeepOPT/DataOPT/140114_5dpf"  # 5 days post-fertilization
f140117_3dpf = "/home/marcos/DeepOPT/DataOPT/140117_3dpf"  # 3 days post-fertilization
f140115_1dpf = "/home/marcos/DeepOPT/DataOPT/140315_1dpf"  # 1 days post-fertilization

f140315_3dpf = "/home/marcos/DeepOPT/DataOPT/140315_3dpf"     # 3 days post-fertilization
f140415_5dpf_4X = "/home/marcos/DeepOPT/DataOPT/140415_5dpf_4X"  # 5 days post-fertilization
f140419_5dpf = "/home/marcos/DeepOPT/DataOPT/140519_5dpf"     # 5 days post-fertilization

f140714_5dpf = "/home/marcos/DeepOPT/DataOPT/140714_5dpf"
f140827_3dpf_4X = "/home/marcos/DeepOPT/DataOPT/140827_3dpf_4X"
f140827_5dpf_4X = '/home/marcos/DeepOPT/DataOPT/140827_5dpf_4X'

folder_paths = [f140114_5dpf, f140117_3dpf, f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf]
results_folder = '/home/marcos/DeepOPT/Resultados/'
model_folder = '/home/marcos/DeepOPT/Models/'

#%% Load registered datasets
umbral_reg = 50
train_dataset = []
test_dataset = []

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
  if df.Tparams['Ty'].mean() < umbral_reg:
    
    print('Registration transformation {}'.format(df.Tparams['Ty'].mean()))
    # apendeo los volumenes (es lo unico que me importan, ya estan registrados)
    if dataset_num <= 4: 
        train_dataset.append(df.getRegisteredVolume('head', saveDataset = False))
    # Borro el dataframe
    else:
        test_dataset.append(df.getRegisteredVolume('head', saveDataset = False))

  del df

#%% Datasets 
# Training with more than one dataset
number_projections = 90
train_size = 200
test_size = 200
batch_size = 5

trainX = []
trainY = []
# Datasets settings
for dataset in train_dataset:
  
  l = len(train_dataset)
  tY, tX = DL.formDatasets(dataset, number_projections, train_size//l)
  trainX.append(tX)
  trainY.append(tY)

testX, testY = DL.formDatasets(test_dataset, number_projections, test_size)

# Image resizing
resizer = torchvision.transforms.Resize((size, size))
trainX = [resizer.forward(tX) for tX in trainX]
trainY = [resizer.forward(tY) for tY in trainY]

testX = resizer.forward(testX)
testY = resizer.forward(testY)

#### AQUI FALTA PONER LOS DATA LOADERS
# # Build dataloaders
# trainX = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(trainX), 
#                                      batch_size=batch_size, 
#                                      shuffle=False, num_workers=0)
# trainY = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(trainY), 
#                                      batch_size=batch_size, 
#                                      shuffle=False, num_workers=0)

# testX = torch.utils.data.DataLoader(testX, batch_size=batch_size, 
#                                            shuffle=False, num_workers=0)
# testY = torch.utils.data.DataLoader(testY, batch_size=batch_size, 
#                                            shuffle=False, num_workers=0)

#%% Model Settings

nLayer = 5
K = 10
epochs = 100
lam = 0.01
maxAngle = 360

model = modl.OPTmodl(nLayer, K, maxAngle, number_projections, size, None, lam)
loss_fn = torch.nn.MSELoss(reduction = 'sum')
lr = 1e-3
optimizer = torch.optim.RMSprop(model.parameters())

#%% Model training
num_epochs_save = 10
loss_full = []

for epoch in range(epochs+1):

  for des, under in zip(desired_dataloader, under_dataloader):
    
    predicted = model(under)
    loss = loss_fn(predicted['dc'+str(K-1)], des)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  if (epoch%num_epochs_save) == 0:
    
    loss, current = loss.item(), epoch
    print(f"loss: {loss:>7f}  [{current:>5d}/{epochs:>5d}]")
    loss_full.append(loss)

    model.epochs_save = epoch
    model.state_dict()['epoch_save'] = epoch
    torch.save({'state_dict' : model.state_dict(),
                'epoch_save' : epoch},
               model_folder+'MODL_size{}_nLayer{}_K{}_lam{}_projnum{}.pth'.format(size, nLayer, K, lam, number_projections))


#%% Plot results

fig, ax = plt.subplots(loss)


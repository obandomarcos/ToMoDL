import torch 
from torchsummary import summary

model_path_old = '/home/obanmarcos/Balseiro/DeepOPT/SavedModels/KFolding_PSNR_MODL_Test71K_8_lam_0.05_nlay_8_proj_72__Kfold0.pth'

model = torch.load(model_path_old)

print(model.keys())
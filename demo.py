import piq
import numpy as np
import torch


def _to_torch_pair(ref: np.ndarray, img: np.ndarray, device: str = "cpu"):
    """
    Convert numpy arrays to torch tensors with shape (1, 1, H, W).

    Convention:
        ref = reference / target
        img = test / prediction / reconstruction
    """
    assert ref.shape == img.shape

    # # Normalize using reference scale

    if len(img.shape) == 2:
        ref_min, ref_max = ref.min(), ref.max()
        denom = ref_max - ref_min

        ref_n = (ref - ref_min) / denom
        img_n = (img - ref_min) / denom
        img_n = np.clip(img_n, 0.0, 1.0)
        ref_t = torch.tensor(ref_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        img_t = torch.tensor(img_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    elif len(img.shape) == 3:
        ref_t = torch.zeros((img.shape[0], 1, img.shape[1], img.shape[2])).to(device)
        img_t = torch.zeros((img.shape[0], 1, img.shape[1], img.shape[2])).to(device)
        for i in range(img.shape[0]):
            ref_n = (ref[i] - ref[i].min()) / (ref[i].max() - ref[i].min())
            img_n = (img[i] - ref[i].min()) / (ref[i].max() - ref[i].min())
            img_n = np.clip(img_n, 0.0, 1.0)
            # plt.figure(figsize=(20, 5))
            # plt.subplot(1, 2, 1)
            # plt.axis("off")
            # plt.imshow(img_n, cmap='gray')
            # # add label
            # plt.title("QBI FBP")
            # plt.subplot(1, 2, 2)
            # plt.imshow(ref_n, cmap='gray')
            # plt.title("Tomopy Astra SART")
            # plt.axis("off")
            # plt.colorbar()
            # ref_t[i,0] = torch.tensor(ref_n, dtype=torch.float32)
            # img_t[i,0] = torch.tensor(img_n, dtype=torch.float32)
    return ref_t, img_t


def ssim_numpy(ref: np.ndarray, img: np.ndarray, data_range=1, device: str = "cpu") -> float:
    ref_t, img_t = _to_torch_pair(ref, img, device)
    return piq.ssim(img_t, ref_t, data_range=data_range)


def fsim_numpy(ref: np.ndarray, img: np.ndarray, data_range=1, device: str = "cpu") -> float:
    ref_t, img_t = _to_torch_pair(ref, img, device)
    return piq.fsim(img_t, ref_t, data_range=data_range, chromatic=False)


def msssim_numpy(ref: np.ndarray, img: np.ndarray, data_range=1, device: str = "cpu") -> float:
    ref_t, img_t = _to_torch_pair(ref, img, device)
    return piq.multi_scale_ssim(img_t, ref_t, data_range=data_range)


def vif_numpy(ref: np.ndarray, img: np.ndarray, device: str = "cpu") -> float:
    ref_t, img_t = _to_torch_pair(ref, img, device)
    # IMPORTANT: VIF is not symmetric
    return piq.vif_p(img_t, ref_t)


def psnr_numpy(ref: np.ndarray, img: np.ndarray, data_range=1, device: str = "cpu") -> float:
    ref_t, img_t = _to_torch_pair(ref, img, device)
    return piq.psnr(img_t, ref_t, data_range=data_range)


import matplotlib.pyplot as plt
from pathlib import Path
import ToMoDL.utilities.dataloading_utilities as dlutils
from config import *
from torch.utils.data import DataLoader
import torch

# import wandb
import sys, os

# from tomopari.src.tomopari.processors.OPTProcessor import ToMoDL

from ToMoDL.models.modl import ToMoDL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# run = wandb.init(project="deepopt")

# Load model
# artifact_tomodl = run.use_artifact('/path/to/artifact', type='model')   # '/datasets/x20/140114_5dpf_body_20/'
# artifact_tomodl_dir = artifact_tomodl.download()
# artifact_tomodl_dir = 'datasets/x20/140114_5dpf_body_20/'

# model_tomodl = MoDLReconstructor.load_from_checkpoint(Path(artifact_tomodl_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict)
resnet_options_dict = {
    "number_layers": 8,
    "kernel_size": 3,
    "features": 64,
    "in_channels": 1,
    "out_channels": 1,
    "stride": 1,
    "use_batch_norm": True,
    "init_method": "xavier",
    "device": device,
}

tomodl_dictionary = {
    "use_torch_radon": True,
    "K_iterations": 6,
    "number_projections": 720,
    "acceleration_factor": 20,
    "lambda": 0.5,
    "is_half_rotation": False,
    "use_shared_weights": True,
    "denoiser_method": "U-Net",
    "resnet_options": resnet_options_dict,
    "in_channels": 1,
    "out_channels": 1,
    "device": device,
    "iter_conjugate": 5,
}

model_tomodl = ToMoDL(tomodl_dictionary)
model_tomodl.to(device)
# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# artifact_path = os.path.join(__location__, "model.ckpt")
artifact_path = "/home/nhattm/ToMoDL/ToMoDL/gh1j7acb/checkpoints/tomodl128.ckpt"
tomodl_checkpoint = torch.load(artifact_path, map_location=torch.device("cuda"), weights_only=False)

tomodl_checkpoint["state_dict"] = {k.replace("model.", ""): v for k, v in tomodl_checkpoint["state_dict"].items()}
# tomodl_checkpoint["state_dict"] = dict(filter(my_filtering_function, tomodl_checkpoint["state_dict"].items()))
model_tomodl.load_state_dict(tomodl_checkpoint["state_dict"], strict=True)
model_tomodl.eval()
# model_tomodl.lam = torch.nn.Parameter(torch.tensor([labmda], requires_grad=True, device=device))
print(model_tomodl.lam)
# Load dataset
dataset_dict = {
    "root_folder": "/home/nhattm/ToMoDL/datasets/full_fish_128/x20/140827_3dpf_4x_head_20",  # In our case, datasets/x20/140114_5dpf_body_20
    "acceleration_factor": 20,
    "transform": None,
}

test_dataset = dlutils.ReconstructionDataset(**dataset_dict)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Extract image from dataloader and move it to CPU after processing
us_unfil_im, us_fil_im, fs_fil_im = list(iter(test_dataloader))[
    300
]  # Unfiltered undersampled and filtered undersampled and fully sampled FBP

# us_unfil_im = torch.load("test_input.pt", map_location="cpu", weights_only=False)
print(us_unfil_im.min(), us_unfil_im.max(), us_unfil_im.mean(), us_unfil_im.std(), us_unfil_im.shape)
unfil_im = us_unfil_im.numpy().squeeze()
fil_im = us_fil_im.numpy().squeeze()
fs_im = fs_fil_im.numpy().squeeze()
# image_tomodl = model_tomodl(us_unfil_im.to(device))["dc" + str(model_tomodl.model.K)][0, 0].cpu().detach().numpy()  # Model Output
image_tomodl = (
    model_tomodl(us_unfil_im.to(device))["dc" + str(tomodl_dictionary["K_iterations"])][0, 0].cpu().detach().numpy()
)  # Model Output
image_tomodl = (image_tomodl - image_tomodl.mean()) / image_tomodl.std()

# print("PSNR tomodl:", psnr_numpy(fs_im, image_tomodl, device="cpu"))
# print("SSIM tomodl:", ssim_numpy(fs_im, image_tomodl, device="cpu"))
# print("PSNR unfiltered:", psnr_numpy(fs_im, fil_im, device="cpu"))
# print("SSIM unfiltered:", ssim_numpy(fs_im, fil_im, device="cpu"))

# Plot comparison
plt.figure(figsize=(25, 3))
plt.subplot(151)
plt.imshow(unfil_im, cmap="gray")
plt.colorbar()
plt.title("Unfiltered")

plt.subplot(152)
plt.imshow(fil_im, cmap="gray")
plt.colorbar()
plt.title("Filtered")

plt.subplot(153)
plt.imshow(fs_im, cmap="gray")
plt.colorbar()
plt.title("Full image")

plt.subplot(154)
plt.imshow(image_tomodl, cmap="gray")
plt.colorbar()
plt.title("dc final")
plt.tight_layout()
plt.show()

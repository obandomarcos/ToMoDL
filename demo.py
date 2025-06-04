from ToMoDL.models.models_system import MoDLReconstructor
import matplotlib.pyplot as plt
from pathlib import Path
import ToMoDL.utilities.dataloading_utilities as dlutils
from config import *
from torch.utils.data import DataLoader
import torch

# import wandb
import sys, os

sys.path.append("./napari-tomodl/src/napari_tomodl/")

from processors import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
labmda = 0.5
tomodl_dictionary = {
    "use_torch_radon": True,
    "metric": "psnr",
    "K_iterations": 6,
    "number_projections_total": 100,
    "acceleration_factor": 20,
    "image_size": 256,
    "lambda": labmda,
    "use_shared_weights": True,
    "denoiser_method": "resnet",
    "resnet_options": resnet_options_dict,
    "in_channels": 1,
    "out_channels": 1,
    "device": device,
    "iter_conjugate": 10,
}

model_tomodl = ToMoDL(tomodl_dictionary)

# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# artifact_path = os.path.join(__location__, "model.ckpt")
artifact_path = "napari-tomodl/src/napari_tomodl/processors/model256_lambda0.4.ckpt"
tomodl_checkpoint = torch.load(artifact_path, map_location=torch.device("cuda:0"))

tomodl_checkpoint["state_dict"] = {k.replace("model.", ""): v for k, v in tomodl_checkpoint["state_dict"].items()}
# tomodl_checkpoint["state_dict"] = dict(filter(my_filtering_function, tomodl_checkpoint["state_dict"].items()))
model_tomodl.load_state_dict(tomodl_checkpoint["state_dict"], strict=False)
# model_tomodl.lam = torch.nn.Parameter(torch.tensor([labmda], requires_grad=True, device=device))
print(model_tomodl.lam)
# Load dataset
dataset_dict = {
    "root_folder": "datasets/full_fish_256/x20/140114_5dpf_body_20",  # In our case, datasets/x20/140114_5dpf_body_20
    "acceleration_factor": 20,
    "transform": None,
}

test_dataset = dlutils.ReconstructionDataset(**dataset_dict)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Extract image from dataloader and move it to CPU after processing
us_unfil_im, us_fil_im, fs_fil_im = next(
    iter(test_dataloader)
)  # Unfiltered undersampled and filtered undersampled and fully sampled FBP
unfil_im = us_unfil_im.numpy().squeeze()
fil_im = us_fil_im.numpy().squeeze()
fs_im = fs_fil_im.numpy().squeeze()

# image_tomodl = model_tomodl(us_unfil_im.to(device))["dc" + str(model_tomodl.model.K)][0, 0].cpu().detach().numpy()  # Model Output
image_tomodl = (
    model_tomodl(us_unfil_im.to(device))["dc" + str(tomodl_dictionary["K_iterations"])][0, 0].cpu().detach().numpy()
)  # Model Output

# Plot comparison
plt.figure(figsize=(15, 3))

plt.subplot(141)
plt.imshow(unfil_im)
plt.colorbar()
plt.title("Unfiltered")

plt.subplot(142)
plt.imshow(fil_im)
plt.colorbar()
plt.title("Filtered")

plt.subplot(143)
plt.imshow(fs_im)
plt.colorbar()
plt.title("Full image")

plt.subplot(144)
plt.imshow(image_tomodl)
plt.colorbar()
plt.title("Inference")

plt.tight_layout()
plt.show()

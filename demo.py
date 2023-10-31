from ToMoDL.models.models_system import MoDLReconstructor
import matplotlib.pyplot as plt
from pathlib import Path
import ToMoDL.utilities.dataloading_utilities as dlutils
from config import model_system_dict, trainer_system_dict, dataloader_system_dict
from torch.utils.data import Dataloader
import torch
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = wandb.init(project="deepopt")

# Load model
artifact_tomodl = run.use_artifact('path/to/artifact', type='model')
artifact_tomodl_dir = artifact_tomodl.download()
model_tomodl = MoDLReconstructor.load_from_checkpoint(Path(artifact_tomodl_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict)

# Load dataset
dataset_dict = {'root_folder' : 'datasets/x20/140315_3dpf_body_20', # In our case, datasets/x20/140315_3dpf_body_20
                'acceleration_factor' : 20,
                'transform' : None}
test_dataset = dlutils.ReconstructionDataset(**dataset_dict)
test_dataloader = DataLoader(test_dataset, 
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)

# Extract image from dataloader and move it to CPU after processing
us_unfil_im, us_fil_im, fs_fil_im = next(iter(test_dataloader)) # Unfiltered undersampled and filtered undersampled and fully sampled FBP
image_tomodl = model_tomodl(us_unfil_im.to(device))['dc'+str(model_tomodl.model.K)][0,0,...].detach().cpu().numpy() #
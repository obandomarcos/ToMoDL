<h1 align="center">🔬<ins>ToMoDL</ins>🔬<br>Model-Based Deep Learning Architecture for Optical Tomography Projection 3D Reconstruction</h1>
<p align="center">
  <a href="https://www.linkedin.com/in/marcos-obando-22a816170">Marcos Obando</a>
  ·
  <a href="https://www.creatis.insa-lyon.fr/~ducros/WebPage/">Nicolas Ducros</a>
  ·
  <a href="https://scholar.google.com/citations?user=8ZHx3j8AAAAJ&hl=fr">Andrea Bassi</a>
  ·
  <a href="https://scholar.google.com/citations?hl=es&user=LUH06dgAAAAJ&view_op=list_works&sortby=pubdate">Germán Mato</a>
  ·
  <a href="https://scholar.google.com/citations?user=-xtye-QAAAAJ&hl=en">Teresa Correia</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/marcoso96/ToMoDL/main/images/ComparativeQualitative.png" >
  <br>
  <em>ToMoDL is a model-based neural network for tomographic reconstruction with a low number of samples<br>Introducing a physics informed reconstruction algorithm, high quality images can be recovered from more than 20 times less acquisition data.</em>
</p>

##

Welcome to the ToMoDL GitHub repository! This repository contains code that implements a technique for reconstructing OPT (Optical Projection Tomography) images by solving an optimization problem using a Model-Based Deep Learning (MoDL) architecture. 🌟

### 📝 Description

This code is based on the MoDL architecture developed in [1]. It provides a powerful method to reconstruct OPT images by solving the following optimization problem:

$$ \mathbf{x_{rec}} = \arg\min_{\mathbf{x}} ||\mathbf{A}\mathbf{x} - \mathbf{b}||^2_2 + \lambda ||\mathbf{x}-\mathcal{D}_{\mathbf{\theta}}(\mathbf{x})||^2_2 $$

Here, $\mathbf{A}$ can be any measurement operator, and in our case, we considered the Radon transform operator combined with an undersampling mask. $\mathcal{D}_{\mathbf{\theta}}(\mathbf{x})$ represents the denoiser, which uses a residual learning CNN.
<p align="center">
  <img src="https://raw.githubusercontent.com/marcoso96/ToMoDL/main/images/Algorithm.png" >
  <br>
  <em>ToMoDL solves the proposed inverse problem via a proximal gradient algorithm, where data consistency and denoising are alternatingly enforced. </em>
</p>
<!-- 
### 📦 Dependencies

The code relies on the Torch Radon library, developed by Ronchetti [2], for implementing the Radon forward and backward operators. To install it, you can use the following command:

```shell script
wget -qO- https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/master/auto_install.py  | python -
```

Please note that Torch Radon has some compatibility issues with PyTorch versions above 1.7. We recommend creating a virtual environment with PyTorch version between 1.5 and 1.7. Python 3.8 should work fine. -->

## 📂 Installation

We encourage to create a custom virtual environment for running ToMoDL with the following instructions:

```
conda create --name tomodl python=3.8
```

Install required packages using pip inside venv:

```
pip install -r requirements.txt
```

### 📚 OPT Datasets

The ToMoDL architecture can be trained using the OPT datasets stored in the `DataOPT` folder. These datasets consist of transmitted projection images of live zebrafish at different days post fertilization (dpf). Each dataset comprises a series of 888 sinograms `y` of shape `D x θ`, sampled at 720 angle steps (0.5 degrees per step) with an 880-pixel detector, where each pixel is evenly spaced by 1.3 μm.

To ensure compatibility with Torch Radon operators, sinograms are resized to 640 angle steps, making them a multiple of 16 projections. The image preprocessing involves registering opposite pairs to correct the axis of rotation's shift with respect to the detector. 

NOTE: In order to test out the reconstruction algorithm itself, we provide a preprocessed dataset at the folder `datasets`, which contains fully sampled reconstructions for ground truth (FBP) and x20 undersampled reconstructions (FBP for comparison and unfiltered reconstruction for input).

## Demo

Under the Lightning framework, using ToMoDL consists in three main modules for dataloading, model and training setting. Minimal configuration are provided within dictionaries at `config.py`. We'll directly import a default configuration for L=8 layers, K=8 iterations ToMoDL and train the model. We can configure [Weights and Biases](https://wandb.ai/site) to monitor the training:

```python
from tomodl.training import train_utilities as trutils
from tomodl.config import model_system_dict, trainer_system_dict, dataloader_system_dict
import wandb

wandb.init()
default_configs = {'trainer_kwdict': trainer_dict,
                    'dataloader_kwdict' : dataloader_dict,
                    'model_system_kwdict': model_system_dict}

trainer = trutils.TrainerSystem(**default_configs)
trainer.train_model()
```
With the saved model, we can easily perform reconstructions loading the artifacts from W&B or the corresponding checkpoint file. We'll load the test dataset to asess its performance:

```python
from models.models_system import MoDLReconstructor
import matplotlib.pyplot as plt

# Load model
artifact_tomodl = run.use_artifact('path/to/artifact', type='model')
artifact_tomodl_dir = artifact_tomodl.download()
model_tomodl = MoDLReconstructor.load_from_checkpoint(Path(artifact_tomodl_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict)

# Load dataset
dataset_dict = {'root_folder' : 'path/to/test/dataset/', # In our case, datasets/x20/140315_3dpf_body_20
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
```

## Further configuration

With modules built on top of [PyTorch Lightning ⚡](https://lightning.ai/docs/pytorch/stable/), `config.py` dictionaries enable us to explore useful strategies for iterative networks such as stochastic weights averaging, custom batch accumulation, parallelization across multiple devices among others. Here we show the default configuration for our training, where we extended its trainer in order to perform K-folding and resume training from previous folds. A further exploration of these features can be found at the `scripts` folder.

```python
lightning_trainer_dict = {'max_epochs': 40,
                            'log_every_n_steps': 10,
                            'check_val_every_n_epoch': 1,
                            'gradient_clip_val' : 0.5,
                            'accelerator' : 'gpu', 
                            'devices' : 1,
                            'fast_dev_run' : False,
                            'default_root_dir': model_folder}

trainer_dict = {'lightning_trainer_dict': lightning_trainer_dict,
                'use_k_folding': True, 
                'track_checkpoints': True,
                'epoch_number_checkpoint': 10,
                'use_swa' : False,
                'use_accumulate_batches': False,
                'k_fold_number_datasets': 3,
                'use_logger' : True,
                'logger_dict': logger_dict,
                'track_default_checkpoints'  : True,
                'use_auto_lr_find': False,
                'batch_accumulate_number': 3,
                'use_mixed_precision': False,
                'batch_accumulation_start_epoch': 0, 
                'profiler': profiler, 
                'restore_fold': False,
                'resume': False}

```

## 📚 References

[1] MoDL: Model-Based Deep Learning Architecture for Inverse Problems by H.K. Aggarwal, M.P Mani, and Mathews Jacob in IEEE Transactions on Medical Imaging, 2018.

[2] Ronchetti, M. (2020). Torchradon: Fast differentiable routines for computed tomography. arXiv preprint arXiv:2009.14788.

Explore the code and have fun reconstructing optical tomography projections with ToMoDL! If you have any questions or suggestions, feel free to reach out. 🤗🚀

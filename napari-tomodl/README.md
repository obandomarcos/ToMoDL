# napari-tomodl

[![License MIT](https://img.shields.io/pypi/l/napari-tomodl.svg?color=green)](https://github.com/marcoso96/napari-tomodl/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-tomodl.svg?color=green)](https://pypi.org/project/napari-tomodl)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-tomodl.svg?color=green)](https://python.org)
<!-- [![tests](https://github.com/marcoso96/napari-tomodl/workflows/tests/badge.svg)](https://github.com/marcoso96/napari-tomodl/actions) -->
[![codecov](https://codecov.io/gh/marcoso96/napari-tomodl/branch/main/graph/badge.svg)](https://codecov.io/gh/marcoso96/napari-tomodl)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-tomodl)](https://napari-hub.org/plugins/napari-tomodl)

A plugin for optical projection tomography reconstruction with model-based neural networks.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->
## 🔬 Introduction

**napari-tomodl** is a [napari](https://napari.org/) plugin that enables users to easily reconstruct tomography images directly from raw projection data. Simply load an ordered stack of projection files into the napari viewer, and the plugin takes care of reconstructing the corresponding tomographic volume.  

## 🚀 Usage

1. **Load ordered stack**  
![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/napari-tomodl/figures/stack_image.png)

   Go to **File → Open Files as Stack...** and load the angular projections for parallel beam optical tomography reconstruction.

2. **Select image layer**  
![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/napari-tomodl/figures/select_layer.png) 

   In the dropdown menu, click **Select image layer** and choose the loaded volume.  

<!--  make this line be bigger and bold -->
<h3>From here you can choose between two reconstruction modes: Basic and Advanced.</h3>

### 🔹 Basic Mode
![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/napari-tomodl/figures/basic_mode.png)  

3. **Half-rotation**  
   - Click **Half rotation** if your projection data was acquired from 0° to 180°.  
   - Leave it unchecked if data was acquired from 0° to 360°.

4. **Automatic axis alignment**  
   If the rotation axis is not correctly aligned during acquisition, enable **Automatic axis alignment**.  This aligns the sinogram to the detector center using the [Wall-method].

5. **Compression**  
   Projection images are assumed to have shape **(Theta, Detector size, Z)** in vertical axis mode.  
   You can compress along the Z-axis:  
   - **HIGH** → resize Z to 100  
   - **MEDIUM** → resize Z to 256  
   - **LOW** → resize Z to 512  
   - **NO** → no compression  

6. **Reconstruction method**  
   - **FBP CPU / FBP GPU** → from the [QBI_radon] library  
   - **TOMODL CPU / TOMODL GPU / UNET CPU / UNET GPU** → proposed in our [ToMoDL-paper]  

7. **Smoothing level**  
   Select smoothing strength (only applies to **TOMODL** methods). Can be adjusted in the **Advanced mode**.
    - **LOW** → 2  
    - **MEDIUM** → 4  
    - **HIGH** → 6 

8. **Rotation axis**  
   - **Vertical** → for data shape (Theta, Detector size, Z)  
   - **Horizontal** → for data shape (Theta, Z, Detector size)
---

### 🔹 Advanced Mode
![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/52d849b3a488d28d6fcaef3c199dad167ade45b3/napari-tomodl/figures/advanced_mode.png)

9. **Manual axis alignment**  
   Shift the object along the detector axis (Z-axis).  
   - Negative values → shift left  
   - Positive values → shift right  

10. **Reshape volume**  
    Select a reconstruction size (alternative to compression levels from Basic mode).

11. **Flat-field correction**  
    Apply flat-field correction to projection data before reconstruction.

12. **Clip to circle**  
    Constrain the reconstructed object inside a circular region.

13. **Filter (FBP only)**  
    Choose the filter to apply when using FBP methods. 

14. **Full volume**  
    - Enabled → reconstruct the whole volume.  
    - Disabled → reconstruct only a subset of slices along the detector axis (faster for testing).

15. **Batch size**  
    Number of slices processed simultaneously:  
    - Higher values → faster reconstruction but greater GPU memory usage.  
    - On CPU → limited to processing **1 slice at a time**.  

16. **One Slice**  
    - Enabled → reconstruct only a single slice at the specified index.  
    - Disabled → reconstruct from index 0 up to the chosen slice index in the **# of slices to reconstruct** field.  

17. **Invert colors**  
    Invert grayscale values in the reconstructed volume.

18. **16-bit conversion**  
    Convert the reconstructed volume to **16-bit** for faster 3D rendering.  
    Leave unchecked to keep **32-bit float** output.
---

19. **Reconstruct!** 

![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/napari-tomodl/figures/reconstruct_button.png)

   A new layer will appear on top of the projections stack with the reconstructed volume.


## 💻 Installation Guide

### 1️⃣ Install **napari** as a bundled app

Please visit the official napari documentation and follow the instructions here:  
👉 [Napari Installation Guide (bundled app)](https://napari.org/stable/tutorials/fundamentals/installation_bundle_conda.html)

You can also directly download the installer for your operating system:
- **Windows (.exe):**  
  [napari-0.6.4-Windows-x86_64.exe](https://github.com/napari/napari/releases/download/v0.6.4/napari-0.6.4-Windows-x86_64.exe)
- **Ubuntu (.sh):**  
  [napari-0.6.4-Linux-x86_64.sh](https://github.com/napari/napari/releases/download/v0.6.4/napari-0.6.4-Linux-x86_64.sh)

---
### 2️⃣ Install **PyTorch** inside Napari’s bundled environment

This guide provides detailed instructions for installing **PyTorch** within the Napari bundled environment. Follow the steps carefully to ensure compatibility and a smooth installation process.

> **💡 Tip:** Ensure Napari is closed before proceeding with the installation to avoid conflicts.
Before installing PyTorch, verify your system's CUDA version if you plan to use GPU support. This ensures compatibility with the PyTorch version and CUDA toolkit.

**Check CUDA Version**:
   - Open a **Command Prompt** (Windows) or **terminal** (Linux).
   - Run the following command:
     ```bash
     nvidia-smi
     ```
   - Choose a `pytorch-cuda` version that is **less than or equal to** your CUDA version. Supported versions include 11.8, 12.1, or 12.4.
   - For more details on compatible PyTorch versions, visit the [PyTorch Previous Versions page](https://pytorch.org/get-started/previous-versions).

#### 🪟 **For Windows Users**

1. Open **Command Prompt** (do **not** use PowerShell, as it may cause path resolution issues).
2. Run the appropriate command based on your hardware setup:

   #### 🔹 For GPU Support
   Ensure your CUDA version matches the `pytorch-cuda` version specified (e.g., 12.1). Replace `2.5.0` and `12.1` with versions compatible with your system if needed.

   ```bash
   "%LOCALAPPDATA%\napari-0.6.4\envs\napari-0.6.4\Scripts\conda.exe" install -y pytorch==2.5.0 pytorch-cuda=12.1 -c conda-forge -c pytorch -c nvidia --override-channels
   ```

   #### 🔹 For CPU Only
   If you do not have a compatible NVIDIA GPU or prefer to use CPU, run:

   ```bash
   "%LOCALAPPDATA%\napari-0.6.4\envs\napari-0.6.4\Scripts\conda.exe" install -y pytorch==2.5.0 cpuonly -c pytorch -c nvidia -c conda-forge --override-channels
   ```

#### 🐧 **For Linux Users**
1. Open a **terminal**.
2. Run the appropriate command based on your hardware setup:

   #### 🔹 For GPU Support
   Ensure your CUDA version matches the `pytorch-cuda` version specified (e.g., 12.1). Replace `2.5.0` and `12.1` with versions compatible with your system if needed.

   ```bash
   ~/.local/napari-0.6.4/bin/conda install -y pytorch==2.5.0 pytorch-cuda=12.1 -c conda-forge -c pytorch -c nvidia --override-channels
   ```

   #### 🔹 For CPU Only
   If you do not have a compatible NVIDIA GPU or prefer to use CPU, run:

   ```bash
   ~/.local/napari-0.6.4/bin/conda install -y pytorch==2.5.0 cpuonly -c pytorch -c nvidia -c conda-forge --override-channels
   ```
---
### 3️⃣ Install our plugin — **napari-tomodl**

Our plugin is available on the [napari-hub](https://napari-hub.org/plugins/napari-tomodl.html).
---
🔹 Option 1: Install directly from napari
1. Open **napari**.  
2. Go to **Plugins → Install/Uninstall Plugins**.  
3. Search for **napari-tomodl** and click **Install**.
---
🔹 Option 2: Install via pip (from Napari Console)
Open napari’s **Python Console** and type:

```bash
pip install napari-tomodl
```

> After installation, **restart napari** to apply the changes. 😊

## 🤝 Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## 📜 License

Distributed under the terms of the [MIT] license,
"napari-tomodl" is free and open source software

## 🐛Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[QBI_radon]: https://github.com/QBioImaging/QBI_radon
[Wall-method]: https://doi.org/10.1088/0031-9155/50/19/015
[ToMoDL-paper]: https://doi.org/10.1038/s41598-023-47650-3
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

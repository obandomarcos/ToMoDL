# tomopari

[![License MIT](https://img.shields.io/pypi/l/tomopari.svg?color=green)](https://github.com/obandomarcos/ToMoDL/blob/nhattm/tomopari/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tomopari.svg?color=green)](https://pypi.org/project/tomopari)
[![Python Version](https://img.shields.io/pypi/pyversions/tomopari.svg?color=green)](https://python.org)
<!-- [![tests](https://github.com/marcoso96/tomopari/workflows/tests/badge.svg)](https://github.com/marcoso96/tomopari/actions) -->
<!-- [![codecov](https://codecov.io/gh/marcoso96/tomopari/branch/main/graph/badge.svg)](https://codecov.io/gh/marcoso96/tomopari) -->
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/tomopari)](https://napari-hub.org/plugins/tomopari)

A plugin for accelerated tomographic reconstruction.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->
## 🔬 Introduction

**tomopari** is a [napari](https://napari.org/) plugin that enables users to easily reconstruct tomography images directly from raw projection data. Simply load an ordered stack of projection files into the napari viewer, and the plugin takes care of reconstructing the corresponding tomographic volume.  

## 🚀 Usage

1. **Load ordered stack**  
![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/tomopari/figures/stack_image.png)

   Go to **File → Open Files as Stack...** and load the angular projections for parallel beam optical tomography reconstruction.
   
   After loading, the stack of θ-angular projection images should have shape of $N_{angles} × det_h × det_w$, where:

   - $N_{angles}$ is the number of projection views (one image per rotation angle θ),
   - $det_h$ is the detector height (vertical pixel dimension),
   - $det_w$ is the detector width (horizontal pixel dimension).
2. **Select image layer**  
![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/tomopari/figures/select_layer.png) 

   In the dropdown menu, click **Select image layer** and choose the loaded volume.  

<!--  make this line be bigger and bold -->
<h3>From here you can choose between two reconstruction modes: Basic and Advanced.</h3>

### 🔹 Basic Mode
![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/tomopari/figures/basic_mode.png)  

3. **Half-rotation**  
   - Click **Half rotation** if your projection data were acquired from 0° to 180°.  
   - Leave it unchecked if data were acquired from 0° to 360°.

4. **Automatic axis alignment**  
   If the rotation axis is not correctly aligned during acquisition, enable **Automatic axis alignment**.  This aligns the sinogram to the detector center using the [Wall-method].

5. **Compression**  
   Compression affects the detector dimension differently depending on the acquisition mode:

   - Vertical-axis mode → the $det_w$ will be resized
   - Horizontal-axis mode → the $det_h$ will be resized

   Available compression levels:
   - **HIGH** → resize to 100  
   - **MEDIUM** → resize to 256  
   - **LOW** → resize to 512  
   - **NO** → no compression  
  


6. **Reconstruction method**  
   - **FBP CPU / FBP GPU** → from the [QBI_radon] library  
   - **TOMODL CPU / TOMODL GPU / UNET CPU / UNET GPU** → proposed in our [ToMoDL-paper]  

7. **Smoothing level**  
   Select smoothing strength (only applies to **TOMODL** methods). Can be more fine tuned in the **Advanced mode**.
    - **LOW** → 2  
    - **MEDIUM** → 4  
    - **HIGH** → 6 

8. **Rotation axis**  
   Select how your data is organized with respect to the rotation axis:
   - **Vertical** → rotation axis parallel to $det_h$
   - **Horizontal** → rotation axis parallel to $det_w$
---

### 🔹 Advanced Mode
![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/tomopari/figures/advanced_mode.png)

9. **Manual axis alignment**  
   Shift the object along $det_w$ in vertical mode, and along $det_h$ in horizontal mode. 
   - Negative values → shift left (toward lower pixel indices)
   - Positive values → shift right (toward higher pixel indices)

10. **Reshape volume**  
    Select a reconstruction size (alternative to compression levels from Basic mode).

11. **Flat-field correction**  
    Apply flat-field correction to projection data before reconstruction.

12. **Clip to circle**  
    Constrain the reconstructed object inside a circular region.

13. **Filter (FBP only)**  
    Choose the filter to apply when using FBP methods. 

14. **Full volume mode**  
    - Enabled → reconstruct the whole volume.  

16. **One Slice mode**  
    - Enabled → reconstruct only a single slice at **Slice #** index. 

17. **Slices mode**
    - Enabled → reconstruct from index 0  up to the chosen slice index defined in **Slice #**.

18. **Batch size**  
    Number of slices processed simultaneously:  
    - Higher values → faster reconstruction but greater GPU memory usage.  
    - On CPU → limited to processing **1 slice at a time**.  

19. **Invert colors**  
    Invert grayscale values in the reconstructed volume.

20. **16-bit conversion**  
   The reconstructed volume is always generated in **32-bit float** precision. Enable this option to convert the final volume to **16-bit**, which significantly improves 3D rendering performance in napari.
   Leave it unchecked if you prefer to keep the full 32-bit float output.
---

21. **Reconstruct!** 

![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/tomopari/figures/reconstruct_button.png)

   A new layer will appear on top of the projections stack with the reconstructed volume.


## 💻 Installation Guide *(No Code — Highly Recommended)*

### 🧩 **Step 1: Install Napari (Bundled App)**

> 💡 *Skip this step if you already installed Napari via `pip`.*

You can directly download the official Napari **bundled installer** for your operating system:

* 🪟 **Windows (.exe):**
  👉 [napari-0.6.5-Windows-x86_64.exe](https://github.com/napari/napari/releases/download/v0.6.5/napari-0.6.5-Windows-x86_64.exe)

* 🐧 **Ubuntu (.sh):**
  👉 [napari-0.6.5-Linux-x86_64.sh](https://github.com/napari/napari/releases/download/v0.6.5/napari-0.6.5-Linux-x86_64.sh)

📘 **Official Guide:**
Follow the Napari documentation for detailed installation steps:
🔗 [Napari Installation Guide (Bundled App)](https://napari.org/0.6.5/tutorials/fundamentals/installation_bundle_conda.html)

---

### ⚙️ **Step 2: Install PyTorch Inside Napari’s Bundled Environment**

> 💡 *Skip this step if PyTorch is already installed in your Napari environment.*

This step ensures **PyTorch** is properly installed within Napari’s internal Conda environment for full compatibility.

#### 🪟 **Windows Users**

1. Download the installer:
   🔗 [install_torch2napari_windows.bat](https://github.com/obandomarcos/ToMoDL/releases/download/v0.1.0/install_torch2napari_windows.bat)
2. Double-click the `.bat` file to run it.
   *(It will automatically detect Napari’s environment and install PyTorch.)*

#### 🐧 **Linux Users**

1. Download the installer:
   🔗 [install_torch2napari_linux.sh](https://github.com/obandomarcos/ToMoDL/releases/download/v0.1.0/install_torch2napari_linux.sh)
2. Run it in your terminal:

   ```bash
   bash install_torch2napari_linux.sh
   ```

---

### 3️⃣ Install our plugin — **tomopari**

Our plugin is available on the [napari-hub](https://napari-hub.org/plugins/tomopari.html).
---
🔹 Option 1: Install directly from napari
1. Open **napari**.  
2. Go to **Plugins → Install/Uninstall Plugins**.  
3. Search for **tomopari** and click **Install**.
---
🔹 Option 2: Install via pip (from Napari Console)
Open napari’s **Python Console** and type:

```bash
pip install tomopari
```

> After installation, **restart napari** to apply the changes. 😊

## 🤝 Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## 📜 License

Distributed under the terms of the [MIT] license,
"tomopari" is free and open source software

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

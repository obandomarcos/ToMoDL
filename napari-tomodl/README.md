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
![plot](https://raw.githubusercontent.com/obandomarcos/ToMoDL/refs/heads/nhattm/napari-tomodl/figures/advanced_model.png)  

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

14. **Reconstruct full volume**  
    - Enabled → reconstruct the whole volume.  
    - Disabled → reconstruct only a subset of slices along the detector axis (faster for testing).

15. **Batch size**  
    Number of slices processed simultaneously:  
    - Higher values → faster reconstruction but greater GPU memory usage.  
    - On CPU → limited to processing **1 slice at a time**.  

16. **Reconstruct only slices**  
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


## 💻 Installation

This package requires [QBI_radon] for optimized GPU tomographic reconstruction:
<!-- add code block -->
```bash
pip install QBI-radon
```

You can install `napari-tomodl` via [pip]:
<!-- add code block -->
```bash
pip install napari-tomodl
```


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

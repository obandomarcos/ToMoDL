---
title: 'tomopari: A plugin for accelerated tomographic reconstruction'
tags:
authors:
  - name: Marcos Obando
    equal-contrib: true
    affiliation: "1,2,3" # (Multiple affiliations must be quoted)
  - name: Minh Nhat Trinh
    affiliation: 4 # (Multiple affiliations must be quoted)
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
  - name: David Paleček
    affiliation: 4 # (Multiple affiliations must be quoted)
  - name: Germán Mato
    equal-contrib: true 
    affiliation: "1,5"
  - name: Teresa M Correia
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "4,6"
affiliations:
 - name: Instituto Balseiro, Bariloche, Argentina
   index: 1
 - name: Centrum Wiskunde & Informatica, Amsterdam, the Netherlands
   index: 2
 - name: University of Eastern Finland, Kuopio, Finland
   index: 3
 - name: Centro de Ciências do Mar do Algarve (CCMAR/CIMAR LA), University of Algarve, Faro, Portugal
   index: 4   
 - name: Medical Physics Department, Centro Atómico Bariloche, Bariloche, Argentina
   index: 5
 - name: School of Biomedical Engineering and Imaging Sciences, King’s College London, London, United Kingdom
   index: 6
date: 
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
---

# Summary

Recent advances in various tomographic methodologies have contributed to a broad range of applications and research areas, from medical [@ginat2014advances] and industrial X-ray imaging [@de2014industrial] to optical sectioning techniques, which provide a mesoscopic framework for visualizing translucent samples [@sharpe2004optical]. Tomographic imaging involves acquiring raw data (2D projections) from various angles around the object of interest. After acquisition, data are typically preprocessed to correct for artefacts. Then, an appropriate reconstruction algorithm is used to generate a series of 2D cross-sectional images (slices) from the corrected raw projections. Finally, these slices can be visualized individually or combined into 3D renderings, providing a complete and interactive view of the internal structure of the sample.  

# Statement of need

Tomography is essential in biomedical imaging, but 3D reconstruction using conventional methods, such as filtered backprojection (FBP), typically requires hundreds of 2D projections, which can lead to long acquisition times, excessive radiation exposure, potential sample damage, and motion-related artefacts. Compressed sensing (CS) can accelerate acquisitions by reducing the number of projections (undersampling data), but it depends on computationally expensive iterative algorithms [@bioucas2007new] [@correia2015accelerated], while recent deep learning-based methods enable fast, high-quality reconstructions from undersampled data [@davis2019convolutional] [@obando2023model]. However, most existing open-source software focuses on classical methods, requires specialized knowledge, and lacks 3D visualization tools [@scikit-image],[@van2016fast].

Napari [@chiu2022napari] provides a fast, flexible, and user-friendly viewer for 2D and 3D large-scale images, and has rapidly emerged as a hub for high-performance applications, including microscopy, medical imaging, and material sciences, supported by an active open-source community. Despite the availability of various tomographic reconstruction algorithms, there is a lack of integrated software to perform reconstructions alongside other common tasks such as 3D visualization, segmentation [@ronneberger2015u], and tracking [@wu2016deep], within a single, interactive environment. This gap motivated the development of a napari plugin for 3D tomographic reconstruction.

Here, we introduce *tomopari*, a napari plugin that integrates four tomographic reconstruction methods (that often require separate software): FBP [@kak2001principles], CS (Two-step Iterative Shrinkage/Thresholding, TwIST)  [@bioucas2007new] [@correia2015accelerated], U-Net [@ronneberger2015u] [@davis2019convolutional], and Tomographic Model-based Deep Learning (ToMoDL) [@obando2023model], our recently developed physics-informed deep learning approach for accelerated Optical Projection Tomography (OPT). This user-friendly open-source tool enables both classical and state-of-the-art 3D tomographic reconstruction and fast interactive 3D rendering, within a single interactive environment, while providing a flexible framework for future extensions.


# State of the field

Several open-source tools exist for tomographic reconstruction. Scikit-image [@scikit-image] provides accessible implementations of classical analytical methods and is widely used for educational purposes and simple Python-based tomography workflows, but lacks GPU acceleration, a graphical user interface (GUI), and native support for deep learning–based reconstruction. ASTRA [@van2016fast] offers highly efficient CPU and GPU implementations of several common algorithms (e.g., FBP, SART, SIRT) and is widely adopted in X-ray computed tomography (CT) research, where computational performance is critical. However, ASTRA is primarily code-based, has no integrated GUI, and requires external dependencies for visualization and expertise to implement end-to-end workflows. Other libraries, such as TorchRadon [@ronchetti2020torchradon], provide differentiable operators for learning-based reconstruction but primarily serve as low-level modular components rather than complete end-user solutions.

*tomopari* was developed rather than contributing directly to existing projects for several reasons. First, existing libraries typically require substantial programming expertise to assemble full pipelines for preprocessing, reconstruction, visualization, and analysis. Second, they focus mainly on classical analytical or iterative reconstruction methods and do not natively integrate learning-based approaches capable of producing high-quality reconstructions from undersampled data, which is essential for accelerated acquisitions and real-time imaging. By embedding conventional, CS, and deep learning reconstruction methods directly into napari, *tomopari* provides a unified, interactive environment that supports large datasets and complete end-to-end workflows, from raw projections to 3D visualization and analysis. Its compatibility with community-built napari plugins for segmentation and quantitative analysis allows users to work within a single, easy-to-install application, accessible across expertise levels. *tomopari*'s modular design makes it easy to add new features (e.g., add conventional methods, training on user-provided data) and adapt to evolving tomographic imaging workflows.   


# Software design

*tomopari* is a Python-based napari plugin designed to integrate advanced tomographic reconstruction with 3D rendering and image analysis. A central design goal was balancing computational performance and methodological flexibility with accessibility for non-expert users. This motivated a layered architecture in which computationally intensive operations (e.g., Radon transforms, iterative solvers, and deep learning inference) are handled by optimized numerical backends (NumPy, PyTorch, QBI-radon), while user-facing functionality is implemented as a modular napari plugin.

*tomopari* builds on established open-source libraries including NumPy [@harris2020array], SciPy [@virtanen2020scipy], and scikit-image [@scikit-image]. Neural network–based methods (U-Net and ToMoDL) are implemented using PyTorch [@NEURIPS2019_9015]. To reduce the computational cost of iteratively applying the Radon transform (forward model), *tomopari* employs an adapted version of QBI-radon, our fast, differentiable tomography backend implemented as a PyTorch 2.0 extension, enabling efficient execution on both CPU and GPU across major operating systems [@trinh2025radon].

To avoid a programming-heavy API, *tomopari* provides a widget-based interface integrated with napari’s layer model. Reconstruction methods share a common internal interface, allowing analytical, CS, and learning-based approaches to be selected interchangeably while reusing preprocessing, batching, and device-selection logic. This design simplifies maintenance and supports future extension to additional geometries and reconstruction algorithms.

*tomopari* was developed as a standalone plugin supporting interactive reconstruction workflows. Users provide projection stacks and minimal acquisition parameters, with optional controls for alignment [@walls2005correction], compression, filtering, and CPU/GPU execution. The framework is easy to install, user-friendly (with basic and advanced modes), and modular, enabling integration of additional analysis tools and learning workflows.

# Research Impact Statement

*tomopari* introduces a novel, user-oriented capability: fast tomographic reconstruction, from classical analytical methods to state-of-the-art model-based deep learning, directly within an interactive N-dimensional image viewer. It enables rapid hypothesis testing, parameter exploration, and analysis without requiring users to switch between multiple software environments or write custom scripts.

The plugin provides recently published reconstruction methods, including ToMoDL, in a reproducible and reusable form, facilitating their adoption beyond the original developers. It supports multiple real-world imaging modalities (e.g., OPT, X-ray CT, synchrotron-based high-throughput tomography), demonstrating versatility across scales and experimental contexts. Moreover, it is fully open source, released under a permissive license, and structured to support community contributions, with documented workflows, configurable parameters, and hardware-agnostic CPU/GPU execution.

By embedding advanced reconstruction directly into napari, *tomopari* bridges the gap between reconstruction, segmentation, visualization, and quantitative analysis. This shortens the analysis loop for experimental scientists and lowers the expertise required to apply modern reconstruction techniques, positioning *tomopari* as a versatile, reproducible, and high-performance tool for accelerated tomographic imaging research.

# Methods and Workflow

The reconstruction methods implemented in the *tomopari* package are:

- **FBP** (Filtered backprojection) is a classical analytical reconstruction method that applies frequency-domain filtering followed by backprojection. It is computationally efficient and well-suited for simple geometries such as parallel-beam tomography.
- **TwIST** is an iterative compressed sensing method adapted for tomography [@correia2015accelerated], solving a non-convex optimisation problem with total variation regularisation. It produces high-quality reconstructions from undersampled data but is computationally demanding and requires parameter tuning.
- **U\-Net** is a deep learning architecture with skip connections for reconstructing streak-free images from undersampled FBP inputs [@ronneberger2015u] [@davis2019convolutional]. It enables fast inference but requires large training datasets.
- **ToMoDL** is a model-based deep learning framework that alternates between data consistency and CNN-based artefact removal [@obando2023model]. By explicitly incorporating the forward model, it reduces the number of learnable parameters and performs well when training data are limited.

In \autoref{fig:Figura1}, a complete pipeline describing the usage of *tomopari* is presented. The input is a single channel raw data acquired in a parallel beam tomography, loaded as an ordered stack of files. Two user modes are provided: a basic mode for users without deep learning expertise, and an advanced mode for fine control over smoothing, alignment, flat-field correction, and compression trade-offs. Processing steps labeled 1-10 in \autoref{fig:Figura1} are:

![\textbf{tomopari basic and advanced mode pipelines.} Step-by-step from a stack of raw projection acquisition to reconstruction of a single specific slice or full volume. \label{fig:Figura1}](./tomopari/figures/Figure1.pdf)

1. **Load stack** – The workflow begins by importing the ordered stack of raw projection images (sinograms) into napari using its file manager. This generates a new 3D image layer representing the raw data to be reconstructed.

2. **Select image layer** – Within the *tomopari* plugin interface, the user selects the desired input layer by clicking on *Select image layer*. This step defines which dataset will be used for reconstruction.

3. **Half-rotation scans** – If the acquisition corresponds to a 180° half-rotation instead of a full 360° dataset, this option needs to be enabled to correctly interpret the projection geometry during reconstruction.

4. **Manual or automatic axis alignment** – The rotation axis alignment can be adjusted automatically or manually.

   * *Automatic alignment* applies an efficient implementation of the variance maximisation method [@walls2005correction] to estimate the correct center of rotation.
   * *Manual alignment* allows the user to specify the pixel offset corresponding to the rotation axis shift. For fine-tuning, a single-slice reconstruction is recommended for iterative manual adjustment.

5. **Data pre-processing** – Optional pre-processing steps such as flat-field correction and image resizing can be applied to normalise projection intensities and adapt image dimensions before reconstruction. As an alternative, in the basic mode, users can select: 

* *Compression/projection image resizing* - Resize the volume dimension corresponding to the detector’s width/height to reduce memory usage or accelerate computation. The plugin provides four compression levels: **HIGH**, **MEDIUM**, **LOW**, or **NO**, producing resolutions of 100, 256, 512, or the full uncompressed size, respectively

6. **Reconstruction methods** – Users can select between different reconstruction algorithms according to their application:

   * *FBP* (Filtered Back Projection) — classical analytical approach;
   * *TwIST* — iterative algorithm for sparse or undersampled data;
   * *U-Net* — deep learning-based reconstruction using a convolutional network;
   * *ToMoDL* — hybrid deep learning approach combining model-based priors and learned representations.

**CPU/GPU selection** Users can choose whether to perform reconstruction on the CPU or accelerate computations using the GPU, depending on available hardware and the selected algorithm. When running on the GPU, *tomopari* supports **batch reconstruction**, allowing multiple slices to be reconstructed **in parallel** to significantly improve processing speed. The batch size can be adjusted depending on available GPU memory in the advanced mode.

7. **Reconstruction settings** – Parameters controlling reconstruction quality can be adjusted, including the choice of filter (for FBP), smoothing level, and whether to clip the reconstruction to a circular field of view. 
 
 * *Clip to circle* Restricts reconstruction to a circular field of view (FOV), i.e., the reconstructed image is masked so that everything outside the FOV is set to zero, removing background noise and improving visual quality.
 * *Filter selection (for FBP methods)* Users can choose the desired filtering kernel (e.g., *Ram-Lak*, *Shepp-Logan*, etc.) for the filtered backprojection algorithm, balancing noise suppression and edge preservation.
 * *Smoothing level* corresponds to the number of ToMoDL iterations, which controls the sharpness/smoothness of the reconstructed images. As an alternative, in the basic mode, users can select the smoothing strength as **HIGH**, **MEDIUM**, **LOW**, which correspond to 6, 4 and 2 iterations of ToMoDL, respectively.

8. **Reconstruct the full volume or selected slices** – Users can choose to reconstruct the entire volume, a single slice, or a specific range of slices. For large datasets, reconstructing only a few slices is useful for quick testing. Reconstruction can also be performed in multiple batches to optimize memory usage and improve computational efficiency.

9. **Choose rotation axis** – Depending on the experimental setup, the user defines whether the rotation axis is vertical or horizontal to ensure that projections are correctly aligned during reconstruction.

10. **Volume post-processing** – Finally, reconstructed volumes can optionally be post-processed, for example, by applying color inversion or converting the data to 16-bit depth for faster rendering. The reconstructed volume is always produced in 32-bit float precision. However, the plugin provides an option to convert the final volume to 16-bit, which can significantly improve 3D rendering performance in napari without affecting the reconstruction step itself.

* *Full or partial volume reconstruction* Enables fast testing or memory-efficient reconstruction by limiting computation to a subset of slices along the detector axis. 
* *Intensity inversion* Inverts grayscale values in the reconstructed image volume, which can be useful when projection data were acquired with inverted intensity mapping.

Once these steps are completed, the 'Reconstruction' button executes the desired specifications for image recovery from projections. In napari, outputs are written as image layers, which can be analyzed by other plugins and saved in different formats. One special feature that napari offers on top of 3D images is volume rendering, useful once a full volume is computed with the presented plugin. Normalisation of intensity and contrast can also be applied to specific layers using napari's built-in tools in the top-left bar. 

# Use cases

We present three parallel beam tomography use cases for the *tomopari* plugin:

1. \textbf{Optical projection tomography} (OPT)
Projection data of wild-type zebrafish (*Danio rerio*) at 5 days post fertilisation were obtained using a 4$\times$ magnification objective. Using a rotatory cylinder, transmitted projection images were acquired with an angle step of 1 degree. The acquired projections have 2506 $\times$ 768 pixels with a resolution of 1.3 μm per pixel [@bassi2015optical]. These projections were downsampled to 627 $\times$ 192 pixels in order to reduce the computational complexity. Note that deep learning-based reconstruction methods were only trained with this OPT dataset. 
2. \textbf{High-resolution X-ray parallel tomography} (X-ray CT).
Projection data from a foramnifera were obtained using 20 KeV X rays and a high-resolution detector with 1024 $\times$ 1280 pixels (5 μm per pixel). A rotatory support was used to acquire 360 projections with 1-degree interval. The projections were downsampled to 256 $\times$ 320 to reduce computational complexity. The raw data were pre-processed using phase contrast techniques to improve contrast [@Paganin2002]. 
3. \textbf{High-Throughput Tomography} (HiTT).
Synchrotron X-ray projection data from an ant, fixed in a mixture of PFA (paraformaldehyde) and GA (glutaraldehyde), dehydrated and mounted in ethanol, were obtained using a phase-contrast imaging platform for life-science samples on the EMBL beamline P14 [@albers2024high]. The HiTT dataset contains 1800 projections over 180 degrees, acquired at 0.1-degree intervals, each composed of 3 tiles with a size of 2048 $\times$ 2048 pixels each (0.65 μm per pixel). The projections were downsampled by a factor of 2.

In \autoref{fig:Figura2} we show representative examples of the 2D reconstructions obtained with FBP and ToMoDL, and 3D volumes obtained using the plugin with the ToMoDL option. The volumes were fully rendered using the built-in napari capabilities, allowing for full integration of the data analysis workflow in napari. 

![\textbf{Reconstruction use cases}. Left panels: 2D slices reconstructed from undersampled data using FBP and ToMoDL methods (OPT, X-ray CT and synchrotron X-ray HiTT). For each case, the acceleration factor, degrees per step and rotation range are indicated. Right panels: 3D renderings of ToMoDL reconstructions.\label{fig:Figura2}](./tomopari/figures/Figure2.pdf)

# AI usage disclosure

Generative AI tools were not used in the creation of this software, the authorship of this manuscript, or the preparation of any supporting materials.

# Acknowledgements

This study received Portuguese national funds from FCT - Foundation for Science and Technology through contracts UID/04326/2025, UID/PRR/04326/2025 and LA/P/0101/2020 (DOI:10.54499/LA/P/0101/2020), ‘la Caixa’ Foundation and FCT, I P under the Project code LCF/PR/HR22/00533, European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie OPTIMAR grant with agreement no 867450 (DOI:10.3030/867450), European Union’s Horizon Europe Programme IMAGINE under grant agreement no. 101094250 (DOI:10.3030/101094250) and NVIDIA GPU hardware grant. M.O was supported by the European Union (GA 101072354) and UKRI (grant number EP/X030733/1). Views and opinions expressed are however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. M.O. would like to thank Dr Ezgi Demircan-Tureyen and Dr Vladyslav Andriiashen (Centrum Wiskunde & Informatica) for useful discussions. The authors would like to express sincere gratitude to: Dr. Ksenia Denisova, Dr. Elizabeth Duke and Dr. Yannick Schwab from EMBL for providing HiTT data and support in data preparation; Dr. Andrea Bassi from Politecnico di Milano for providing access to the OPT data; and Dr. Jose Lipovetzky and Damian Leonel Corsi (Centro Atomico Bariloche) for providing access to the high-resolution X-ray data.   

# References

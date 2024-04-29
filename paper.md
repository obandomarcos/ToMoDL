---
title: 'napari-ToMoDL'
tags:
authors:
  - name: Marcos Obando
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Germán Mato 
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Teresa Correia
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary


# Statement of need

Recent advances in different tomographic methodologies have contributed to a broad range of research areas, from x-ray dealing with medical and industrial application to optical sectioning, which provides a mesoscopic framework to visualise translucent samples, just to name a few. Once data is acquired, experimental face several challenges: first, raw projections should be reconstructed via a mathematical algorithm and, if needed, correct artifacts in a preprocessing step. For this purpose, several libraries have been introduced within the Python language to aleviate this burden, with scikit-image [@scikit-image], the ASTRA toolbox \cite{} and TorchRadon \cite{} as an example. Nevertheless, the usage of these tools requires specific knowledge of image processing that still refrains experimentalists from an accessible data analysis and visualization framework to handle raw tomographic data.

napari \cite{}, a fast and practical bioimaging visualiser for multidimensional data, has rapidly emerged as a hub for high performance applications spanning a broad range of imaging data, including microscopy, medical imaging, etcetera. Given a context with an extensive offer of tomographic reconstruction algorithms and a lack of software integration for image analysts that enables them to access to other complex tasks such as segmentation \cite{} and  tracking \cite{}, we stood for developing a ready-to-use widget that offers state-of-the-art methods for tomographic reconstruction as well as an extensive framework for future methodologies.

In this work we present napari-ToMoDL, a napari plugin that contains four main methods for tomographic reconstruction: filtered backprojection (FBP), Two-step Iterative Shrinkage/Thresholding (TwIST), U-Net and ToMoDL,  being the last our recent introduced method for optical projection tomography reconstruction. With an ordered stack of raw projections as input for parallel beam reconstruction, the plugin also offers the capability of axis alignment via variance maximization \cite{walls}.  


# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

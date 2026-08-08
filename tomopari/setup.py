from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="tomopari",
    version="0.2.15",
    description="A plugin for accelerated tomographic reconstruction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Marcos Antonio Obando, Minh Nhat Trinh, David Palecek, Germán Mato, Teresa Correia",
    author_email="marcos.obando@ib.edu.ar",
    license="MIT",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Framework :: napari",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.9",
    install_requires=[
        "magicgui",
        "qtpy",
        # "napari",
        # "pyqt5",
        "scikit-image",
        "scipy",
        "qbi-radon",
        "opencv-python-headless",
        # "pyopengl==3.1.6"
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    entry_points={
        "napari.manifest": [
            "tomopari = tomopari:napari.yaml",
        ]
    },
    package_data={"": ["*.yaml"], "tomopari.processors": ["*.ckpt"]},
)

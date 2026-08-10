"""
Graphical reconstruction widget for OPT/CT tomography inside Napari.

This module defines a full-featured Qt-based reconstruction interface for
Optical Projection Tomography (OPT), including:

• Basic and advanced reconstruction modes
• GPU/CPU Filtered Backprojection (FBP)
• TwIST, UNet, and ToMoDL deep-learning reconstruction
• Automatic and manual rotation-axis alignment
• Flat-field correction
• Slice-based, volume-based, and batch reconstruction
• Real-time progress updates via QThread
• Napari layer management for visualization

The widget integrates with `OPTProcessor` to perform reconstruction and with
the Napari viewer for displaying 2D/3D volumes.

Authors:
    Marcos Obando
    Minh Nhat Trinh
    David Palecek
    Germán Mato
    Teresa Correia

Created:
    Feb 2, 2023
"""

# %%
import os
from .processors.OPTProcessor import OPTProcessor
from .processors.functions_utils import *
from .widget_settings import Settings, Combo_box
import gc
import torch
import datetime
import dask.array as da
from magicgui import magic_factory
import napari
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSplitter,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QTabWidget,
    QFormLayout,
    QComboBox,
    QLabel,
    QProgressBar,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QFrame,
)
from qtpy.QtCore import Qt, QThread, Signal
from napari.layers import Image
import numpy as np
from napari.qt.threading import thread_worker
from time import time
import scipy.ndimage as ndi
from enum import Enum

import zarr
import tempfile
from pathlib import Path
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# this thread is used to update the progress bar
class BarThread(QThread):
    """Thread used to update a progress bar during reconstruction.

    Computes a percent completion based on:
        value, min, max → emits progressChanged(int)

    Signals:
        progressChanged (int): Percentage (0–100).

    Attributes:
        min (int): Lower bound of the progress range.
        max (int): Upper bound of the progress range.
        value (int): Current progress position.
    """

    progressChanged = Signal(int)

    def __init__(self, parent=None):
        super(BarThread, self).__init__(parent)
        self.max = 1
        self.min = 0
        self.value = 1

    def run(self):
        percent = (self.value - self.min) / (self.max - self.min) * 100
        self.progressChanged.emit(int(percent))


class Rec_modes(Enum):
    """Available reconstruction algorithms.

    Options:
        FBP_CPU: CPU filtered backprojection
        FBP_GPU: GPU filtered backprojection
        TWIST_CPU: TwIST iterative reconstruction on CPU
        TOMODL_CPU: Model-based deep learning reconstruction (CPU)
        TOMODL_GPU: Model-based deep learning reconstruction (GPU)
        TOMODL fast mode uses the optimized UNet-backed implementation.
    """

    FBP_GPU = 0
    FBP_CPU = 1
    TOMODL_GPU = 2
    TOMODL_CPU = 3
    TWIST_CPU = 6


# OPTProcessor values retained for the former UNET modes. They are now
# selected through the TOMODL "FAST mode" checkbox instead of separate methods.
TOMODL_FAST_GPU_PROCESS = 4
TOMODL_FAST_CPU_PROCESS = 5


class Compression_modes(Enum):
    """Detector-size compression settings for memory/performance trade-offs.

    Values represent target detector width used during resizing.
        HIGH → 128 pixels
        MEDIUM → 256 pixels
        LOW → 512 pixels
        NO → 1024 pixels (no compression / full resolution)
    """

    HIGH = 128
    MEDIUM = 256
    LOW = 512
    NO = 1024


class Smoothing_modes(Enum):
    """Smoothing level for CNN or iterative reconstruction filters.

    Values indicate number of smoothing iterations or diffusion strength.
    """

    LOW = 2
    MEDIUM = 4
    HIGH = 6


class Filter_modes(Enum):
    """Available filters for filtered backprojection (FBP).

    Options:
        RAMP (Ram-Lak)
        SHEPPLOGAN
        COSINE
        HAMMING
        HANN
        NO → use no filter
    """

    RAMP = "ramp"
    SHEPPLOGAN = "shepp-logan"
    COSINE = "cosine"
    HAMMING = "hamming"
    HANN = "hann"  # TODO: add hann
    NO = "NO"


class Order_Modes(Enum):
    """Sinogram axis ordering.

    Vertical   → (theta, detector, z)
    Horizontal → (detector, theta, z)
    """

    Vertical = 0
    Horizontal = 1


class ReconstructionWidget(QTabWidget):
    """Napari widget providing a full tomographic reconstruction interface.

    Supports:
        • Basic and advanced reconstruction workflows
        • Selection of image layers
        • Resizing, axis alignment, flat-field correction
        • Multiple reconstruction algorithms (FBP, TwIST, MoDL, UNet)
        • Slice-wise or full-volume reconstruction
        • Progress reporting via QThread
        • Automatic updating of Napari layers

    Args:
        viewer (napari.Viewer): Active Napari viewer instance.
    """

    name = "Reconstructor"

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.setup_ui_basic()
        self.setup_ui_advanced()

        self.bar_thread_basic = BarThread(self)
        self.bar_thread_basic.progressChanged.connect(self.progressBar_basic.setValue)
        self.bar_thread_advanced = BarThread(self)
        self.bar_thread_advanced.progressChanged.connect(self.progressBar_advanced.setValue)
        self.set_tab_style()

    def setup_ui_basic(self):
        """Initialize the Basic Mode reconstruction tab.

        Creates widgets for:
            • Layer selection
            • Compression
            • Reconstruction method (CPU/GPU, FBP/TWIST/etc.)
            • Rotation-axis mode
            • Smoothing level
            • Basic reconstruction button + progress bar
        """

        def add_section(_layout, _title):
            _layout.addWidget(QLabel(_title))
            _layout.addWidget(QSplitter(Qt.Vertical))

        # Tab 1 - Basic settings and reconstruction

        # i) add a tab widget
        self.acquisition_params_widget_basic = QWidget()
        self.addTab(self.acquisition_params_widget_basic, "Basic Mode")

        # ii) layout
        self.basic_reconstruction_layout = QVBoxLayout()
        self.basic_reconstruction_widget = QWidget()
        # self.basic_reconstruction_layout.addWidget(QLabel("Basic reconstruction"))
        self.basic_reconstruction_layout.addWidget(self.basic_reconstruction_widget)

        self.choose_layer_widget_basic = choose_layer()
        self.choose_layer_widget_basic.call_button.visible = False
        self.add_magic_function(self.choose_layer_widget_basic, self.basic_reconstruction_layout)
        select_button = QPushButton("Select image layer")
        select_button.clicked.connect(self.select_layer_basic)
        self.basic_reconstruction_layout.addWidget(select_button)

        settings_layout = QVBoxLayout()
        add_section(settings_layout, "Settings")
        self.basic_reconstruction_layout.addLayout(settings_layout)
        # remove space between Select image layer and settings
        self.createSettingsBasic(settings_layout)
        self.acquisition_params_widget_basic.setLayout(self.basic_reconstruction_layout)

    def setup_ui_advanced(self):
        """Initialize the Advanced Mode reconstruction tab.

        Adds advanced controls including:
            • Manual & automatic axis alignment
            • Flat-field correction
            • Reconstruction size control
            • Filter selection
            • Slice selection (single, multiple, full volume)
            • Batch size configuration
            • 16-bit output conversion toggle
        """

        def add_section(_layout, _title):
            _layout.addWidget(QLabel(_title))
            _layout.addWidget(QSplitter(Qt.Vertical))

        # Tab 1 - Basic settings and reconstruction

        # i) add a tab widget
        self.acquisition_params_widget_advanced = QWidget()
        self.addTab(self.acquisition_params_widget_advanced, "Advanced Mode")

        # ii) layout
        self.advanced_reconstruction_layout = QVBoxLayout()
        self.advanced_reconstruction_widget = QWidget()
        # self.advanced_reconstruction_layout.addWidget(QLabel("Advanced reconstruction"))
        self.advanced_reconstruction_layout.addWidget(self.advanced_reconstruction_widget)

        self.choose_layer_widget_advanced = choose_layer()
        self.choose_layer_widget_advanced.call_button.visible = False
        self.add_magic_function(self.choose_layer_widget_advanced, self.advanced_reconstruction_layout)
        select_button = QPushButton("Select image layer")
        select_button.clicked.connect(self.select_layer_advanced)
        self.advanced_reconstruction_layout.addWidget(select_button)

        settings_layout = QVBoxLayout()
        add_section(settings_layout, "Settings")
        self.advanced_reconstruction_layout.addLayout(settings_layout)
        self.createSettingsAdvanced(settings_layout)
        self.acquisition_params_widget_advanced.setLayout(self.advanced_reconstruction_layout)

    def createSettingsBasic(self, slayout):
        """Create all configurable basic-mode controls.

        Includes widgets for:
            • Half rotation
            • Automatic axis alignment
            • Compression level
            • Reconstruction method
            • Smoothing mode
            • Axis order

        Also adds:
            • Basic reconstruction start button
            • Progress bar
        """
        self.is_half_rotation_basic = Settings(
            "Half-rotation (angles 0-180)",
            dtype=bool,
            initial=False,
            layout=slayout,
            write_function=self.set_opt_processor_basic,
        )

        self.registerbox_basic = Settings(
            "Automatic axis alignment",
            dtype=bool,
            initial=False,
            layout=slayout,
            write_function=self.set_opt_processor_basic,
        )

        self.compression_basic = Combo_box(
            name="Compression",
            initial=Compression_modes.LOW.value,
            choices=Compression_modes,
            layout=slayout,
            write_function=self.set_opt_processor_basic,
        )

        # create combobox for reconstruction method
        self.reconbox_basic = Combo_box(
            name="Reconstruction method",
            initial=Rec_modes.FBP_GPU.value,
            choices=Rec_modes,
            layout=slayout,
            write_function=self.set_opt_processor_basic,
        )
        self.fast_mode_basic = QCheckBox("FAST mode")
        self.fast_mode_basic.setVisible(False)
        self.fast_mode_basic.toggled.connect(self.set_opt_processor_basic)
        self.reconbox_basic.combo.currentIndexChanged.connect(self.update_basic_fast_mode_visibility)
        slayout.addWidget(self.fast_mode_basic)

        self.smoothingbox_basic = Combo_box(
            name="Smoothing",
            initial=Smoothing_modes.MEDIUM.value,
            choices=Smoothing_modes,
            layout=slayout,
            write_function=self.set_opt_processor_basic,
        )

        self.orderbox_basic = Combo_box(
            name="Rotation axis",
            initial=Order_Modes.Vertical.value,
            choices=Order_Modes,
            layout=slayout,
            write_function=self.set_opt_processor_basic,
        )
        # add space and the end of the layout
        slayout.addSpacing(300)
        # add calculate psf button
        calculate_btn = QPushButton("Basic reconstruct")
        calculate_btn.clicked.connect(self.stack_reconstruction_basic)
        slayout.addWidget(calculate_btn)

        self.progressBar_basic = QProgressBar()
        slayout.addWidget(self.progressBar_basic)

    def createSettingsAdvanced(self, slayout):
        """Create all advanced-mode UI controls.

        Adds widgets for:
            • Flat-field correction
            • Manual / automatic axis alignment
            • Volume resizing
            • Reconstruction filter selection
            • Slice selection mode (full, one slice, multiple)
            • Batch size
            • Rotation axis ordering
            • Color inversion
            • 16-bit conversion
            • Progress bar and reconstruction trigger
        """
        self.is_half_rotation_advanced = Settings(
            "Half-rotation (angles 0-180)",
            dtype=bool,
            initial=False,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )

        self.registerbox_advanced = Settings(
            "Automatic axis alignment",
            dtype=bool,
            initial=False,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )

        self.manualalignbox_advanced = Settings(
            "Manual axis alignment",
            dtype=bool,
            initial=False,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )

        self.alignbox_advanced = Settings(
            "Axis shift",
            dtype=float,
            vmin=-500,
            vmax=500,
            initial=0,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )

        self.reshapebox_advanced = Settings(
            "Reshape volume", dtype=bool, initial=True, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        self.resizebox_advanced = Settings(
            "Reconstruction size",
            dtype=int,
            initial=128,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )
        self.flat_correction_advanced = Settings(
            "Flat-field correction",
            dtype=bool,
            initial=False,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )

        # create combobox for reconstruction method
        self.reconbox_advanced = Combo_box(
            name="Reconstruction method",
            initial=Rec_modes.FBP_GPU.value,
            choices=Rec_modes,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )
        self.fast_mode_advanced = QCheckBox("FAST mode")
        self.fast_mode_advanced.setVisible(False)
        self.fast_mode_advanced.toggled.connect(self.set_opt_processor_advanced)
        self.reconbox_advanced.combo.currentIndexChanged.connect(self.update_fast_mode_visibility)
        slayout.addWidget(self.fast_mode_advanced)
        self.clipcirclebox_advanced = Settings(
            "Clip to circle", dtype=bool, initial=True, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        # self.filterbox_advanced = Settings(
        #     "Use filtering", dtype=bool, initial=True, layout=slayout, write_function=self.set_opt_processor_advanced
        # )
        self.filterbox_advanced = Combo_box(
            name="Filter",
            initial=Filter_modes.RAMP.value,
            choices=Filter_modes,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )
        self.iterations_advanced = Settings(
            "Smoothing Level", dtype=int, initial=2, layout=slayout, write_function=self.set_opt_processor_advanced
        )
        radio_layout = QHBoxLayout()

        # self.fullvolume_advanced = Settings(
        #     "Full volume", dtype=bool, initial=True, layout=radio_layout, write_function=self.set_opt_processor_advanced
        # )
        # self.is_reconstruct_one_advanced = Settings(
        #     "One slice", dtype=bool, initial=False, layout=radio_layout, write_function=self.set_opt_processor_advanced
        # )
        self.fullvolume_advanced_mode = QRadioButton("Full volume")
        self.one_slice_advanced_mode = QRadioButton("One slice")
        self.multiple_slices_advanced = QRadioButton("Slices")
        self.fullvolume_advanced_mode.setChecked(True)  # Full volume = True
        self.one_slice_advanced_mode.setChecked(False)  # One slice = False
        self.multiple_slices_advanced.setChecked(False)  # Multiple slices = False
        # Add to button group for mutual exclusivity
        self.radio_group_advanced = QButtonGroup()
        self.radio_group_advanced.addButton(self.fullvolume_advanced_mode, 0)  # id 0 -> Full volume
        self.radio_group_advanced.addButton(self.one_slice_advanced_mode, 1)  # id 1 -> One slice
        self.radio_group_advanced.addButton(self.multiple_slices_advanced, 2)  # id 2 -> Multiple slices
        self.radio_group_advanced.setExclusive(True)

        # Connect signal to update function
        self.radio_group_advanced.idClicked.connect(self.set_opt_processor_advanced)

        # Add to layout
        radio_layout.addWidget(self.fullvolume_advanced_mode)
        radio_layout.addWidget(self.one_slice_advanced_mode)
        radio_layout.addWidget(self.multiple_slices_advanced)
        # Add layout to slayout
        slayout.addLayout(radio_layout)
        self.slices_advanced = Settings(
            "Slice #", dtype=int, initial=0, layout=slayout, write_function=self.set_opt_processor_advanced
        )
        self.batch_size_advanced = Settings(
            "Batch size", dtype=int, initial=32, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        self.orderbox_advanced = Combo_box(
            name="Rotation axis",
            initial=Order_Modes.Vertical.value,
            choices=Order_Modes,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )
        self.invert_color_advanced = Settings(
            "Invert colors", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor_advanced
        )
        self.output_conversion_advanced = Settings(
            "16-bit conversion",
            dtype=bool,
            initial=True,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )

        # add calculate psf button
        calculate_btn = QPushButton("Advanced reconstruct")
        calculate_btn.clicked.connect(self.stack_reconstruction_advanced)
        slayout.addWidget(calculate_btn)

        self.progressBar_advanced = QProgressBar()
        slayout.addWidget(self.progressBar_advanced)

    def show_image(self, image_values, fullname, min_value=0, max_value=65535, **kwargs):

        if "scale" in kwargs.keys():
            scale = kwargs["scale"]
        else:
            scale = [1.0] * image_values.ndim

        if "hold" in kwargs.keys() and fullname in self.viewer.layers:

            self.viewer.layers[fullname].data = image_values
            self.viewer.layers[fullname].scale = scale

        else:
            layer = self.viewer.add_image(
                image_values,
                name=fullname,
                scale=scale,
                cache=False,
                multiscale=False,
                contrast_limits=[min_value, max_value],
            )
            return layer

    def select_layer_basic(self, sinos: Image):
        """Select input sinogram for basic reconstruction.

        Determines whether the input is 2D or 3D and initializes the OPTProcessor.

        Args:
            sinos (Image): Napari image layer selected by the user.
        """
        sinos = self.choose_layer_widget_basic.image.value

        if sinos.data.ndim == 3 and sinos.data.shape[2] > 1:
            self.flat_field_advanced = flat_field_estimate(sinos.data[0])
            print(f"Flat-field estimate: {self.flat_field_advanced}")
            self.input_type = "3D"
            self.imageRaw_name = sinos.name
            sz, sy, sx = sinos.data.shape
            print(sz, sy, sx)
            if not hasattr(self, "h_basic"):
                self.start_opt_processor_basic()
            print(f"Selected image layer: {sinos.name}")
        else:
            self.input_type = "2D"
            self.imageRaw_name = sinos.name
            # add dim to the image
            # sinos.data = np.expand_dims(sinos.data, axis=0)
            sy, sx = sinos.data.shape
            print(sy, sx)
            if not hasattr(self, "h_basic"):
                self.start_opt_processor_basic()
            print(f"Selected image layer: {sinos.name}")

    def select_layer_advanced(self, sinos: Image):
        """Select input sinogram for advanced-mode reconstruction.

        Also computes flat-field estimate for 3D data.

        Args:
            sinos (Image): Selected Napari image layer.
        """
        sinos = self.choose_layer_widget_advanced.image.value

        if sinos.data.ndim == 3 and sinos.data.shape[2] > 1:
            self.flat_field_advanced = flat_field_estimate(sinos.data[0])
            print(f"Flat-field estimate: {self.flat_field_advanced}")
            self.input_type = "3D"
            self.imageRaw_name = sinos.name
            sz, sy, sx = sinos.data.shape
            print(sz, sy, sx)
            if not hasattr(self, "h_advanced"):
                self.start_opt_processor_advanced()
            print(f"Selected image layer: {sinos.name}")
        else:
            self.input_type = "2D"
            self.imageRaw_name = sinos.name
            # add dim to the image
            # sinos.data = np.expand_dims(sinos.data, axis=0)
            sy, sx = sinos.data.shape
            print(sy, sx)
            if not hasattr(self, "h_advanced"):
                self.start_opt_processor_advanced()
            print(f"Selected image layer: {sinos.name}")

    def stack_reconstruction_basic(self):
        """Run full-volume or single-slice reconstruction in Basic Mode.

        Performs:
            • Preprocessing of sinogram (axis swap, resizing)
            • Optional axis alignment
            • Reconstruction using selected algorithm
            • GPU batching when available
            • Scaling and conversion to uint16
            • Automatic display in Napari

        Runs in a background thread to keep UI responsive.
        """
        self.scale_image_basic = self.viewer.layers[self.imageRaw_name].scale

        def update_opt_image_basic(stack):
            # stack = next((arg for arg in reversed(args) if isinstance(arg, np.ndarray)), args[-1])

            imname = "basic_" + self.imageRaw_name
            self.show_image(
                stack,
                fullname=imname,
                min_value=0,
                max_value=65535,
                scale=self.scale_image_basic,
            )
            print("Stack reconstruction completed")
            gc.collect()
            torch.cuda.empty_cache()

        @thread_worker
        def _reconstruct_basic():
            """
            ToDO: Link projections
            """

            raw_sinos = self.get_sinos()
            if self.input_type == "3D":

                if self.orderbox_basic.val == 0:
                    # This should preferably remain a view, not a copy
                    sinos = np.moveaxis(raw_sinos, 1, 2)
                    self.h_basic.theta, self.h_basic.Q, self.h_basic.Z = sinos.shape

                else:
                    sinos = np.moveaxis(raw_sinos, 0, 1)
                    self.h_basic.Q, self.h_basic.theta, self.h_basic.Z = sinos.shape
                    print(f"Reconstruction sinogram shape: {sinos.shape}")
            else:
                # Avoid repeatedly calling get_sinos()
                sinos = raw_sinos.T[..., None]

                if self.orderbox_basic.val == 0:
                    self.h_basic.theta, self.h_basic.Q, self.h_basic.Z = sinos.shape
                else:
                    self.h_basic.Q, self.h_basic.theta, self.h_basic.Z = sinos.shape

            original_size = self.h_basic.Q

            if self.compression_basic.text in {"HIGH", "MEDIUM", "LOW"}:
                size_compression = Compression_modes[self.compression_basic.text].value
            else:
                size_compression = original_size
            # optVolume = np.empty(
            #     (
            #         size_compression,
            #         size_compression,
            #         self.h_basic.Z,
            #     ),
            #     dtype=np.float32,
            # )
            optVolume = da.empty(
                (size_compression, size_compression, self.h_basic.Z),
                dtype=np.float32,
                chunks=(size_compression, size_compression, 32),
            )
            # -------------------------------------------------------------------------
            # Automatic centre-shift correction
            # -------------------------------------------------------------------------
            if self.registerbox_basic.val:
                rotation_factor = 2 if self.is_half_rotation_basic.val else 1

                sinos, shift_value = find_center_shift(
                    sinos,
                    resize_shape=size_compression,
                    bar_thread=self.bar_thread_basic,
                    type_sino=self.input_type,
                    order_mode=self.orderbox_basic.val,
                    clip_to_circle=True,
                    device=device,
                    rotation_factor=rotation_factor,
                )
                self.alignbox_advanced.val = shift_value

            # -------------------------------------------------------------------------
            # Determine the slices to reconstruct
            # -------------------------------------------------------------------------
            if self.input_type == "3D":
                slice_start = 0
                slice_stop = self.h_basic.Z
            else:
                # A 2D input is stored as one slice.
                slice_start = 0
                slice_stop = 1

            # -------------------------------------------------------------------------
            # Determine batch size
            # -------------------------------------------------------------------------
            gpu_modes = {
                Rec_modes.FBP_GPU.value,
                Rec_modes.TOMODL_GPU.value,
            }

            if self.reconbox_basic.val in gpu_modes:
                if size_compression <= 128:
                    batch_size = 32
                elif size_compression <= 256:
                    batch_size = 16
                elif size_compression <= 512:
                    batch_size = 8
                else:
                    batch_size = 4
            else:
                batch_size = 1

            # Avoid using a batch larger than the number of available slices.
            batch_size = min(
                batch_size,
                max(1, slice_stop - slice_start),
            )

            # -------------------------------------------------------------------------
            # Reconstruction
            # -------------------------------------------------------------------------
            self.bar_thread_basic.start()
            self.bar_thread_basic.max = slice_stop
            time_in = datetime.datetime.now()
            self.global_min, self.global_max = np.inf, -np.inf
            for batch_start in range(slice_start, slice_stop, batch_size):
                batch_end = min(batch_start + batch_size, slice_stop)
                zidx = slice(batch_start, batch_end)

                print(
                    f"Reconstructing slices {batch_start} to {batch_end - 1}",
                    end="\r",
                )

                # Input sinogram layout:
                #     (theta, detector_pixels, Z)
                source = sinos[:, :, zidx]

                # Compute only the current batch when sinos is backed by Dask.
                # This prevents loading the whole sinogram volume into RAM.
                if isinstance(source, da.Array):
                    source = source.compute()

                source = source.astype(
                    np.float32,
                    copy=False,
                )

                # ---------------------------------------------------------------------
                # Resize only the current batch when necessary.
                #
                # Remove this block if the basic-mode sinograms have already been
                # resized before entering this reconstruction section.
                #
                # Expected resize_batch input/output:
                #     (theta, detector_pixels, batch)
                # ---------------------------------------------------------------------
                if source.shape[1] != size_compression:
                    source = self.h_basic.resize_batch(source)

                # Reconstruction expects:
                #     (detector_pixels, theta, batch)
                source_t = source.transpose((1, 0, 2)) if self.orderbox_basic.val == 0 else source
                if self.registerbox_basic.val:
                    source_t = shift_volume_one_axis_cv2(
                        source_t,
                        shift_value,
                        axis=0,
                    )
                # Ensure that the reconstruction backend receives a contiguous
                # float32 array. This is especially useful for GPU reconstruction.
                stack = np.ascontiguousarray(
                    source_t,
                    dtype=np.float32,
                )

                reconstructed_batch = self.h_basic.reconstruct(stack)
                self.global_min = min(self.global_min, reconstructed_batch.min())
                self.global_max = max(self.global_max, reconstructed_batch.max())
                optVolume[:, :, zidx] = reconstructed_batch

                # Update progress using the actual end of the current batch.
                self.bar_thread_basic.value = batch_end
                self.bar_thread_basic.run()

            total_time = datetime.datetime.now() - time_in

            print("\nComputation time total: {:.3f} s".format(total_time.total_seconds()))

            # -------------------------------------------------------------------------
            # Convert layout:
            #     (height, width, Z) -> (Z, height, width)
            # -------------------------------------------------------------------------
            optVolume = np.moveaxis(optVolume, -1, 0)
            # -------------------------------------------------------------------------
            # Update image scale after compression
            # -------------------------------------------------------------------------
            if self.compression_basic.text in {"HIGH", "MEDIUM", "LOW"}:
                resize_ratio = original_size / size_compression

                if self.input_type == "3D":
                    self.scale_image_basic = [
                        self.scale_image_basic[0],
                        self.scale_image_basic[1] * resize_ratio,
                        self.scale_image_basic[2] * resize_ratio,
                    ]
                else:
                    self.scale_image_basic = [
                        self.scale_image_basic[0] * resize_ratio,
                        self.scale_image_basic[1],
                    ]

            # -------------------------------------------------------------------------
            # Stop progress thread
            # -------------------------------------------------------------------------
            self.bar_thread_basic.value = 0
            self.bar_thread_basic.run()
            self.bar_thread_basic.quit()

            # -------------------------------------------------------------------------
            # Convert reconstructed data to uint16
            # -------------------------------------------------------------------------
            # print("min and max values of the reconstruction:", optVolume.min().compute(), optVolume.max().compute())
            print("min and max values of the reconstruction:", optVolume.min(), optVolume.max())
            optVolume = (optVolume - self.global_min) / (self.global_max - self.global_min) * 65535
            optVolume = optVolume.astype(np.uint16, copy=False)
            self.global_min, self.global_max = 0, 65535
            print("reconstruction shape:", optVolume.shape)
            # convert to dask array

            # -------------------------------------------------------------------------
            # Return result
            # -------------------------------------------------------------------------
            if self.input_type == "3D":
                return optVolume

            return optVolume[0]

        self._reconstruction_worker_basic = _reconstruct_basic()
        self._reconstruction_worker_basic.returned.connect(update_opt_image_basic)
        self._reconstruction_worker_basic.start()

    def stack_reconstruction_advanced(self):
        """Run reconstruction using Advanced Mode settings.

        Supports:
            • Flat-field correction
            • Manual or automatic rotational alignment
            • Custom reconstruction size
            • Full, single, or multiple-slice reconstruction
            • GPU batch processing
            • Optional output conversion to 16-bit

        Output volume is automatically pushed to Napari.
        """
        self.scale_image_advanced = self.viewer.layers[self.imageRaw_name].scale

        def update_opt_image_advanced(stack):
            # stack = next((arg for arg in reversed(args) if isinstance(arg, np.ndarray)), args[-1])

            imname = "advanced_" + self.imageRaw_name
            self.show_image(
                stack,
                fullname=imname,
                min_value=self.global_min,
                max_value=self.global_max,
                scale=self.scale_image_advanced,
            )
            print("Stack reconstruction completed")
            gc.collect()
            torch.cuda.empty_cache()

        @thread_worker
        def _reconstruct_advanced():
            """
            ToDO: Link projections
            """

            # Load the sinogram only once
            # -------------------------------------------------------------------------
            raw_sinos = self.get_sinos()

            if self.input_type == "3D":
                if self.orderbox_advanced.val == 0:
                    # This should preferably remain a view, not a copy
                    sinos = np.moveaxis(raw_sinos, 1, 2)
                    self.h_advanced.theta, self.h_advanced.Q, self.h_advanced.Z = sinos.shape

                else:
                    sinos = np.moveaxis(raw_sinos, 0, 1)
                    self.h_advanced.Q, self.h_advanced.theta, self.h_advanced.Z = sinos.shape

            else:
                # Avoid repeatedly calling get_sinos()
                sinos = raw_sinos.T[..., None]

                if self.orderbox_advanced.val == 0:
                    self.h_advanced.theta, self.h_advanced.Q, self.h_advanced.Z = sinos.shape
                else:
                    self.h_advanced.Q, self.h_advanced.theta, self.h_advanced.Z = sinos.shape

            original_size = self.h_advanced.Q

            # -------------------------------------------------------------------------
            # Resize only if required
            # Warning: this operation may still create a large full-volume allocation.
            # -------------------------------------------------------------------------
            if self.reshapebox_advanced.val:
                reconstruction_size = self.resizebox_advanced.val
            else:
                reconstruction_size = original_size
            optVolume = da.empty(
                (
                    reconstruction_size,
                    reconstruction_size,
                    self.h_advanced.Z,
                ),
                dtype=np.float32,
                chunks=(reconstruction_size, reconstruction_size, 32),
            )

            if self.registerbox_advanced.val:
                rotation_factor = 2 if self.is_half_rotation_advanced.val else 1

                sinos, shift_value = find_center_shift(
                    sinos,
                    resize_shape=self.resizebox_advanced.val if self.reshapebox_advanced.val else None,
                    bar_thread=self.bar_thread_advanced,
                    type_sino=self.input_type,
                    order_mode=self.orderbox_basic.val,
                    clip_to_circle=True,
                    device=device,
                    rotation_factor=rotation_factor,
                )
                self.alignbox_advanced.val = shift_value

            elif self.manualalignbox_advanced.val:
                shift_value = self.alignbox_advanced.val

            # -----------------------------------------------------------------------
            # Determine the slices to reconstruct
            # -------------------------------------------------------------------------
            if self.input_type == "3D" and self.one_slice_advanced_mode.isChecked():
                slice_start = int(self.slices_advanced.val)
                slice_stop = slice_start + 1

            elif self.input_type == "3D" and self.multiple_slices_advanced.isChecked():
                slice_start = 0
                slice_stop = min(
                    self.slices_advanced.val + 1,
                    self.h_advanced.Z,
                )

            else:
                slice_start = 0
                slice_stop = self.h_advanced.Z

            # -------------------------------------------------------------------------
            # Determine batch size
            # -------------------------------------------------------------------------
            gpu_modes = {
                Rec_modes.FBP_GPU.value,
                Rec_modes.TOMODL_GPU.value,
            }

            if self.reconbox_advanced.val in gpu_modes:
                batch_size = max(1, int(self.batch_size_advanced.val))
            else:
                batch_size = 1

            batch_size = min(batch_size, slice_stop - slice_start)
            if self.flat_correction_advanced.val:
                print("Using flat-field correction")
                flat_field = self.flat_field_advanced
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # -------------------------------------------------------------------------
            # Reconstruction
            # -------------------------------------------------------------------------
            self.bar_thread_advanced.start()
            self.bar_thread_advanced.max = slice_stop

            time_in = time()
            self.global_min, self.global_max = np.inf, -np.inf
            for batch_start in range(slice_start, slice_stop, batch_size):
                batch_end = min(batch_start + batch_size, slice_stop)
                zidx = slice(batch_start, batch_end)

                print(
                    f"Reconstructing slices {batch_start} to {batch_end - 1}",
                    end="\r",
                )

                source = sinos[:, :, zidx]

                # Compute only this batch when sinos is a Dask array.
                if isinstance(source, da.Array):
                    source = source.compute()

                # -------------------------------------------------------------
                # 2. Convert/correct the batch before resizing
                # -------------------------------------------------------------

                if self.flat_correction_advanced.val:
                    for image in range(source.shape[2]):
                        image_clahe = np.copy(source[:, :, image])
                        image_clahe = -np.log(image_clahe / flat_field)

                        image_clahe = (image_clahe - np.min(image_clahe)) / (np.max(image_clahe) - np.min(image_clahe))
                        image_clahe = clahe.apply((image_clahe * 65535).astype(np.uint16, copy=False))
                        source[:, :, image] = image_clahe.astype(np.float32, copy=False)

                source = source.astype(np.float32, copy=False)

                # -------------------------------------------------------------
                # 3. Resize only the current batch
                #
                # Input:
                #     (theta, Q, batch)
                #
                # Output:
                #     (theta, resized_Q, batch)
                # -------------------------------------------------------------
                if self.reshapebox_advanced.val:
                    source = self.h_advanced.resize_batch(source)

                # -------------------------------------------------------------
                # 4. Reorder for the reconstruction function
                #
                # (theta, Q, batch) -> (Q, theta, batch)
                # -------------------------------------------------------------
                source_t = source.transpose((1, 0, 2)) if self.orderbox_advanced.val == 0 else source

                if self.manualalignbox_advanced.val or self.registerbox_advanced.val:
                    source_t = shift_volume_one_axis_cv2(
                        source_t,
                        shift_value,
                        axis=0,
                    )
                # Ensure the array passed to the reconstruction function is
                # contiguous in memory.
                stack = np.ascontiguousarray(
                    source_t,
                    dtype=np.float32,
                )

                # -------------------------------------------------------------
                # 5. Reconstruct the current batch
                # -------------------------------------------------------------
                reconstructed_batch = self.h_advanced.reconstruct(stack)
                self.global_min = min(self.global_min, reconstructed_batch.min())
                self.global_max = max(self.global_max, reconstructed_batch.max())
                optVolume[:, :, zidx] = reconstructed_batch

                # -------------------------------------------------------------
                # 6. Update progress
                # -------------------------------------------------------------
                self.bar_thread_advanced.value = batch_end
                self.bar_thread_advanced.run()

            print("Computation time total: {} s".format(round(time() - time_in, 3)))

            self.bar_thread_advanced.value = 0
            self.bar_thread_advanced.run()  # (Q, Q, Z)
            optVolume = da.moveaxis(optVolume, -1, 0)  # (Q, Q, Z) -> (Z, Q, Q)

            if self.reshapebox_advanced.val:
                if (
                    self.fullvolume_advanced_mode.isChecked() or self.multiple_slices_advanced.isChecked()
                ) and self.input_type == "3D":
                    # self.scale_image_advanced = [
                    #     self.scale_image_advanced[0] / original_size * self.resizebox_advanced.val,
                    #     self.scale_image_advanced[1],
                    #     self.scale_image_advanced[2],
                    # ]
                    self.scale_image_advanced = [
                        self.scale_image_advanced[0],
                        self.scale_image_advanced[1] * original_size / self.resizebox_advanced.val,
                        self.scale_image_advanced[2] * original_size / self.resizebox_advanced.val,
                    ]
                else:
                    self.scale_image_advanced = [
                        self.scale_image_advanced[0] * original_size / self.resizebox_advanced.val,
                        self.scale_image_advanced[1] * original_size / self.resizebox_advanced.val,
                    ]
            else:
                if self.one_slice_advanced_mode.isChecked() and self.input_type == "3D":
                    self.scale_image_advanced = [1.0, 1.0]

            self.bar_thread_advanced.value = 0
            self.bar_thread_advanced.run()
            self.bar_thread_advanced.quit()

            # convert to uint16
            if self.output_conversion_advanced.val == True:
                print("min: ", self.global_min, "max: ", self.global_max)
                optVolume = (optVolume - self.global_min) / (self.global_max - self.global_min) * 65535
                optVolume = optVolume.astype(np.uint16, copy=False)
                self.global_min, self.global_max = 0, 65535
                print("done converting to uint16")
            if self.invert_color_advanced.val:
                optVolume = self.global_max - optVolume

            if self.one_slice_advanced_mode.isChecked() and self.input_type == "3D":
                # print("scale image advanced: ", self.scale_image_advanced)
                return optVolume[self.slices_advanced.val]
            elif self.fullvolume_advanced_mode.isChecked() and self.input_type == "3D":
                return optVolume
            elif self.multiple_slices_advanced.isChecked() and self.input_type == "3D":
                number_of_slices = min(
                    int(self.slices_advanced.val) + 1,
                    int(optVolume.shape[0]),
                )
                return optVolume[:number_of_slices]
            else:
                return optVolume[0]

        self._reconstruction_worker_advanced = _reconstruct_advanced()
        self._reconstruction_worker_advanced.returned.connect(update_opt_image_advanced)
        self._reconstruction_worker_advanced.start()

    def get_sinos(self):
        """Return the currently selected sinogram layer.

        Returns:
            ndarray: Sinogram data.
        Raises:
            KeyError: If no valid sinogram layer is selected.
        """
        try:
            return self.viewer.layers[self.imageRaw_name].data
        except:
            raise (KeyError(r"Please select a valid 3D image ($\theta$, q, z)"))

    def set_opt_processor_basic(self, *args):
        """Update OPTProcessor parameters for Basic Mode.

        Applies UI-selected settings to:
            • Reconstruction method
            • Compression level
            • Rotation axis mode
            • Smoothing iterations
            • Batch size
            • Half rotation
        """

        if hasattr(self, "h_basic"):

            self.h_basic.resize_val = Compression_modes[self.compression_basic.text].value
            self.h_basic.rec_process = self.get_basic_reconstruction_mode()
            self.h_basic.order_mode = self.orderbox_basic.val
            self.h_basic.clip_to_circle = True
            self.h_basic.use_filter = True
            self.h_basic.filter_FBP = Filter_modes.RAMP.value
            size_compression = Compression_modes[self.compression_basic.text].value
            if self.reconbox_basic.val in {
                Rec_modes.FBP_GPU.value,
                Rec_modes.TOMODL_GPU.value,
            }:
                if size_compression == 128:
                    self.h_basic.batch_size = 32
                elif size_compression == 256:
                    self.h_basic.batch_size = 16
                elif size_compression == 512:
                    self.h_basic.batch_size = 8
                else:
                    self.h_basic.batch_size = 4
            else:
                self.h_basic.batch_size = 1

            self.h_basic.invert_color = False
            self.h_basic.is_half_rotation = self.is_half_rotation_basic.val
            self.h_basic.iterations = Smoothing_modes[self.smoothingbox_basic.text].value
            self.h_basic.set_reconstruction_process()

    def update_basic_fast_mode_visibility(self, *args):
        """Show FAST mode only for basic ToMoDL reconstruction methods."""
        tomodl_modes = {
            Rec_modes.TOMODL_GPU.value,
            Rec_modes.TOMODL_CPU.value,
        }
        self.fast_mode_basic.setVisible(self.reconbox_basic.current_data in tomodl_modes)

    def get_basic_reconstruction_mode(self):
        """Resolve basic TOMODL FAST selections to their processor path."""
        mode = self.reconbox_basic.current_data
        if not self.fast_mode_basic.isChecked():
            return mode
        if mode == Rec_modes.TOMODL_GPU.value:
            return TOMODL_FAST_GPU_PROCESS
        if mode == Rec_modes.TOMODL_CPU.value:
            return TOMODL_FAST_CPU_PROCESS
        return mode

    def set_opt_processor_advanced(self, *args):
        """Update OPTProcessor parameters for Advanced Mode.

        Applies:
            • Resize settings
            • Reconstruction algorithm
            • Filter type
            • Axis mode
            • Batch size
            • Denoising iterations
            • Half rotation mode
            • Color inversion & clipping
        """

        if hasattr(self, "h_advanced"):

            self.h_advanced.resize_val = self.resizebox_advanced.val
            self.h_advanced.rec_process = self.get_advanced_reconstruction_mode()
            self.h_advanced.order_mode = self.orderbox_advanced.val
            self.h_advanced.clip_to_circle = self.clipcirclebox_advanced.val
            self.h_advanced.use_filter = True if self.filterbox_advanced.text != "NO" else False
            self.h_advanced.filter_FBP = Filter_modes[self.filterbox_advanced.text].value
            self.h_advanced.batch_size = self.batch_size_advanced.val
            self.h_advanced.invert_color = self.invert_color_advanced.val
            self.h_advanced.is_half_rotation = self.is_half_rotation_advanced.val
            self.h_advanced.iterations = self.iterations_advanced.val
            self.h_advanced.set_reconstruction_process()

    def update_fast_mode_visibility(self, *args):
        """Show FAST mode only for ToMoDL reconstruction methods."""
        tomodl_modes = {
            Rec_modes.TOMODL_GPU.value,
            Rec_modes.TOMODL_CPU.value,
        }
        self.fast_mode_advanced.setVisible(self.reconbox_advanced.current_data in tomodl_modes)

    def get_advanced_reconstruction_mode(self):
        """Resolve TOMODL FAST selections to their processor implementation."""
        mode = self.reconbox_advanced.current_data
        if not self.fast_mode_advanced.isChecked():
            return mode
        if mode == Rec_modes.TOMODL_GPU.value:
            return TOMODL_FAST_GPU_PROCESS
        if mode == Rec_modes.TOMODL_CPU.value:
            return TOMODL_FAST_CPU_PROCESS
        return mode

    def start_opt_processor_basic(self):
        """Initialize or reset the Basic Mode OPTProcessor instance."""
        self.isCalibrated = False

        if hasattr(self, "h_basic"):
            self.stop_opt_processor_basic()
            self.start_opt_processor_basic()
        else:
            print("Reset")
            self.h_basic = OPTProcessor()

    def stop_opt_processor_basic(self):
        if hasattr(self, "h_basic"):
            delattr(self, "h_basic")

    def start_opt_processor_advanced(self):
        """Initialize or reset the Advanced Mode OPTProcessor instance."""
        self.isCalibrated = False

        if hasattr(self, "h_advanced"):
            self.stop_opt_processor_advanced()
            self.start_opt_processor_advanced()
        else:
            print("Reset")
            self.h_advanced = OPTProcessor()

    def stop_opt_processor_advanced(self):
        if hasattr(self, "h_advanced"):
            delattr(self, "h_advanced")

    # def reset_processor(self, *args):

    #     self.isCalibrated = False
    #     self.stop_opt_processor()
    #     self.start_opt_processor()

    def add_magic_function(self, widget, _layout):
        """Attach a magicgui widget to the layout and auto-refresh layer list.

        Args:
            widget: MagicGUI widget instance.
            _layout: Parent Qt layout.
        """
        self.viewer.layers.events.inserted.connect(widget.reset_choices)
        self.viewer.layers.events.removed.connect(widget.reset_choices)
        _layout.addWidget(widget.native)

    def set_tab_style(self):
        self.setDocumentMode(True)

        self.tabBar().setStyleSheet("""
            QTabBar {
                background: transparent;
            }

            QTabBar::tab {
                background-color: #252831;
                color: #d8d8d8;

                border: 1px solid #3a3d47;
                border-bottom: none;

                padding: 4px 10px;
                margin-right: 1px;

                min-width: 70px;
                min-height: 18px;
            }

            QTabBar::tab:selected {
                background-color: #363a45;
                color: white;

                border-top: 2px solid #8a8f9a;
            }

            QTabBar::tab:hover:!selected {
                background-color: #30343e;
            }
        """)

        self.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
                top: -1px;
            }
        """)


@magic_factory
def choose_layer(image: Image):
    """Layer-selection helper used by magicgui."""
    pass  # TODO: substitute with a qtwidget without magic functions

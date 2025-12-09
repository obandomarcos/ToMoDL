"""
Created on Tue Feb 2 16:34:41 2023
@authors: Marcos Obando, Minh Nhat Trinh, David Palecek, Germán Mato, Teresa Correia
"""

# %%
import os
from .processors.OPTProcessor import OPTProcessor
from .processors.functions_utils import *
from .widget_settings import Settings, Combo_box
import gc
import torch
import datetime
from magicgui import magic_factory
import napari
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSplitter,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QTabWidget,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QComboBox,
    QLabel,
    QProgressBar,
    QRadioButton,
    QButtonGroup,
)
from qtpy.QtCore import Qt, QThread, Signal
from napari.layers import Image
import numpy as np
from napari.qt.threading import thread_worker
from time import time
import scipy.ndimage as ndi
from enum import Enum


import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# this thread is used to update the progress bar
class BarThread(QThread):
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
    FBP_CPU = 0
    FBP_GPU = 1
    TWIST_CPU = 2
    TOMODL_CPU = 3
    TOMODL_GPU = 4
    UNET_CPU = 5
    UNET_GPU = 6



class Compression_modes(Enum):
    HIGH = 100
    MEDIUM = 256
    LOW = 512
    NO = 1024

class Smoothing_modes(Enum):
    LOW = 2
    MEDIUM = 4
    HIGH = 6

class Filter_modes(Enum):
    RAMP = "ramp"
    SHEPPLOGAN = "shepp-logan"
    COSINE = "cosine"
    HAMMING = "hamming"
    HANN = "hann" # TODO: add hann
    NO = "NO"
class Order_Modes(Enum):
    Vertical = 0
    Horizontal = 1


class ReconstructionWidget(QTabWidget):

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

    def setup_ui_basic(self):
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
        self.is_half_rotation_basic = Settings(
            "Half-rotation (angles 0-180)", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor_basic
        )

        self.registerbox_basic = Settings(
            "Automatic axis alignment", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor_basic
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
            initial=Rec_modes.FBP_CPU.value,
            choices=Rec_modes,
            layout=slayout,
            write_function=self.set_opt_processor_basic,
        )

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
        self.is_half_rotation_advanced = Settings(
            "Half-rotation (angles 0-180)", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        self.registerbox_advanced = Settings(
            "Automatic axis alignment", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        self.manualalignbox_advanced = Settings(
            "Manual axis alignment", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        self.alignbox_advanced = Settings("Axis shift", dtype=float, vmin=-500, vmax=500, initial=0, layout=slayout, write_function=self.set_opt_processor_advanced)

        self.reshapebox_advanced = Settings(
            "Reshape volume", dtype=bool, initial=True, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        self.resizebox_advanced = Settings(
            "Reconstruction size", dtype=int, initial=100, layout=slayout, write_function=self.set_opt_processor_advanced
        )
        self.flat_correction_advanced = Settings(
            "Flat-field correction", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        # create combobox for reconstruction method
        self.reconbox_advanced = Combo_box(
            name="Reconstruction method",
            initial=Rec_modes.FBP_CPU.value,
            choices=Rec_modes,
            layout=slayout,
            write_function=self.set_opt_processor_advanced,
        )
        self.clipcirclebox_advanced = Settings(
            "Clip to circle", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        # self.filterbox_advanced = Settings(
        #     "Use filtering", dtype=bool, initial=True, layout=slayout, write_function=self.set_opt_processor_advanced
        # )
        self.filterbox_advanced = Combo_box(name="Filter", initial=Filter_modes.RAMP.value, choices=Filter_modes,layout=slayout,
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
        self.fullvolume_advanced_mode.setChecked(True)   # Full volume = True
        self.one_slice_advanced_mode.setChecked(False)   # One slice = False
        self.multiple_slices_advanced.setChecked(False)   # Multiple slices = False
        # Add to button group for mutual exclusivity
        self.radio_group_advanced = QButtonGroup()
        self.radio_group_advanced.addButton(self.fullvolume_advanced_mode, 0)  # id 0 -> Full volume
        self.radio_group_advanced.addButton(self.one_slice_advanced_mode, 1)     # id 1 -> One slice
        self.radio_group_advanced.addButton(self.multiple_slices_advanced, 2)     # id 2 -> Multiple slices
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
            "Slices #", dtype=int, initial=0, layout=slayout, write_function=self.set_opt_processor_advanced
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
            "16-bit conversion", dtype=bool, initial=True, layout=slayout, write_function=self.set_opt_processor_advanced
        )

        # add calculate psf button
        calculate_btn = QPushButton("Advanced reconstruct")
        calculate_btn.clicked.connect(self.stack_reconstruction_advanced)
        slayout.addWidget(calculate_btn)

        self.progressBar_advanced = QProgressBar()
        slayout.addWidget(self.progressBar_advanced)

    def show_image(self, image_values, fullname, **kwargs):

        if "scale" in kwargs.keys():
            scale = kwargs["scale"]
        else:
            scale = [1.0] * image_values.ndim

        if "hold" in kwargs.keys() and fullname in self.viewer.layers:

            self.viewer.layers[fullname].data = image_values
            self.viewer.layers[fullname].scale = scale

        else:
            layer = self.viewer.add_image(
                image_values, name=fullname, scale=scale, interpolation2d="linear", cache=False
            )
            return layer

    def select_layer_basic(self, sinos: Image):

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
        self.scale_image_basic = self.viewer.layers[self.imageRaw_name].scale
        def update_opt_image_basic(stack):

            imname = "basic_" + self.imageRaw_name
            self.show_image(stack, fullname=imname, scale=self.scale_image_basic)
            print("Stack reconstruction completed")
            gc.collect()
            torch.cuda.empty_cache()

        @thread_worker(
            connect={"returned": update_opt_image_basic},
        )
        def _reconstruct_basic():
            """
            ToDO: Link projections
            """

            if self.orderbox_basic.val == 0 and self.input_type == "3D":
                print("shape of sinos: ", self.get_sinos().shape)
                sinos = np.moveaxis(np.float32(self.get_sinos()), 1, 2)
                self.h_basic.theta, self.h_basic.Q, self.h_basic.Z = sinos.shape
            elif self.orderbox_basic.val == 1 and self.input_type == "3D":
                sinos = np.moveaxis(np.float32(self.get_sinos()), 0, 1)
                self.h_basic.Q, self.h_basic.theta, self.h_basic.Z = sinos.shape
            elif self.orderbox_basic.val == 0 and self.input_type == "2D":
                sinos = np.float32(self.get_sinos().T)[..., None]
                self.h_basic.theta, self.h_basic.Q, self.h_basic.Z = sinos.shape
            elif self.orderbox_basic.val == 1 and self.input_type == "2D":
                sinos = np.float32(self.get_sinos().T)[..., None]
                self.h_basic.Q, self.h_basic.theta, self.h_basic.Z = sinos.shape
            # clip circle always False
            original_size = int(np.floor(self.h_basic.Q / np.sqrt(2)))

            if self.compression_basic.text in {"HIGH", "MEDIUM", "LOW"}:
                size_compression = Compression_modes[self.compression_basic.text].value
                sinos = self.h_basic.resize(sinos, type_sino=self.input_type)
            else:
                size_compression = original_size
                
            optVolume = np.zeros([size_compression, size_compression, self.h_basic.Z], np.float32)

            if self.registerbox_basic.val == True:
                rotation_factor = 2 if self.is_half_rotation_basic.val == True else 1
                sinos = find_center_shift(sinos, bar_thread=self.bar_thread_basic, type_sino=self.input_type, 
                                          order_mode=self.orderbox_basic.val, clip_to_circle=False, device=device, rotation_factor=rotation_factor)

            # Reconstruction process
            #reconstructing full volume 
            if self.input_type == "3D":
                slices_reconstruction = range(self.h_basic.Z)
            else:
                slices_reconstruction = [0]

            batch_start = slices_reconstruction[0]
            # if use GPU process in batch to improve performance
            if self.reconbox_basic.val in {Rec_modes.FBP_GPU.value, Rec_modes.TOMODL_GPU.value, Rec_modes.UNET_GPU.value}:
                if size_compression == 100:
                    batch_process = 32
                elif size_compression == 256:
                    batch_process = 16
                elif size_compression == 512:
                    batch_process = 8
                else:
                    batch_process = 4
            else:
                batch_process = 1

            batch_end = batch_start + batch_process
            # add progressBar to track the reconstruction process
            self.bar_thread_basic.start()
            self.bar_thread_basic.max = slices_reconstruction[-1] + 1
            # calculate the total time of reconstruction
            total_time = 0
            time_in = datetime.datetime.now()
            while batch_start <= slices_reconstruction[-1]:
                print("Reconstructing slices {} to {}".format(batch_start, batch_end), end="\r")
                zidx = slice(batch_start, batch_end)
                ####################### stacks reconstruction ############################
                if self.input_type == "3D":
                    if self.orderbox_basic.val == 0:
                        optVolume[:, :, zidx] = self.h_basic.reconstruct(sinos[:, :, zidx].transpose(1, 0, 2))
                    elif self.orderbox_basic.val == 1:
                        optVolume[:, :, zidx] = self.h_basic.reconstruct(sinos[:, :, zidx])

                ####################### 2D reconstruction ############################
                elif self.input_type == "2D":
                    if self.orderbox_basic.val == 0:
                        optVolume[:, :, zidx] = self.h_basic.reconstruct(sinos[:, :, zidx].transpose(1, 0, 2))
                    elif self.orderbox_basic.val == 1:
                        optVolume[:, :, zidx] = self.h_basic.reconstruct(sinos[:, :, zidx])

                self.bar_thread_basic.value = batch_end
                self.bar_thread_basic.run()
                batch_start = batch_end
                batch_end += batch_process
            total_time = datetime.datetime.now() - time_in
            print("Computation time total: {} s".format(total_time.total_seconds()))

            self.bar_thread_basic.value = 0
            self.bar_thread_basic.run()
            optVolume = np.rollaxis(optVolume, -1)
            #  change the scale instead of resizing ################################### TODO: change this to resizing
            if self.compression_basic.text in {"HIGH", "MEDIUM", "LOW"}:
                if self.input_type == "3D":
                    # self.scale_image_basic = [self.scale_image_basic[0], self.scale_image_basic[1] * original_size / size_compression, self.scale_image_basic[2]* original_size / size_compression]
                    self.scale_image_basic = [self.scale_image_basic[0] / original_size * size_compression, self.scale_image_basic[1], self.scale_image_basic[2]]
                else:
                    self.scale_image_basic = [self.scale_image_basic[0] * original_size / size_compression, self.scale_image_basic[1]]


            self.bar_thread_basic.value = 0
            self.bar_thread_basic.run()
            self.bar_thread_basic.quit()

            min_val = optVolume.min()
            max_val = optVolume.max()
            # save optVolume to tif file
            # tif.imwrite("optVolume_FBP_GPU.tif", optVolume)
            print("min: ", min_val, "max: ", max_val)
            # convert to uint16
            optVolume = (optVolume - min_val) / (max_val - min_val) * (2**16 - 1)
            optVolume = optVolume.astype(np.uint16, copy=False)
            print("done converting to uint16")
            print("reconstruction shape: ", optVolume.shape)
            if self.input_type == "3D":
                return optVolume
            else:
                return optVolume[0]

        _reconstruct_basic()

    def stack_reconstruction_advanced(self):
        self.scale_image_advanced = self.viewer.layers[self.imageRaw_name].scale
        def update_opt_image_advanced(stack):

            imname = "advanced_" + self.imageRaw_name
            self.show_image(stack, fullname=imname, scale=self.scale_image_advanced)
            print("Stack reconstruction completed")
            gc.collect()
            torch.cuda.empty_cache()

        @thread_worker(
            connect={"returned": update_opt_image_advanced},
        )
        def _reconstruct_advanced():
            """
            ToDO: Link projections
            """

            if self.orderbox_advanced.val == 0 and self.input_type == "3D":
                sinos = np.moveaxis(np.float32(self.get_sinos()), 1, 2)
                self.h_advanced.theta, self.h_advanced.Q, self.h_advanced.Z = sinos.shape
            elif self.orderbox_advanced.val == 1 and self.input_type == "3D":
                sinos = np.moveaxis(np.float32(self.get_sinos()), 0, 1)
                self.h_advanced.Q, self.h_advanced.theta, self.h_advanced.Z = sinos.shape
            elif self.orderbox_advanced.val == 0 and self.input_type == "2D":
                sinos = np.float32(self.get_sinos().T)[..., None]
                self.h_advanced.theta, self.h_advanced.Q, self.h_advanced.Z = sinos.shape
            elif self.orderbox_advanced.val == 1 and self.input_type == "2D":
                sinos = np.float32(self.get_sinos().T)[..., None]
                self.h_advanced.Q, self.h_advanced.theta, self.h_advanced.Z = sinos.shape

            original_size = self.h_advanced.Q if self.clipcirclebox_advanced.val else int(np.floor(self.h_advanced.Q / np.sqrt(2)))

            if self.reshapebox_advanced.val:
                optVolume = np.zeros([self.resizebox_advanced.val, self.resizebox_advanced.val, self.h_advanced.Z], np.float32)
                sinos = self.h_advanced.resize(sinos, type_sino=self.input_type)
            else:
                optVolume = np.zeros([original_size, original_size, self.h_advanced.Z], np.float32)

            if self.flat_correction_advanced.val == True and self.input_type == "3D":
                sinos = sinos / self.flat_field_advanced

            if self.registerbox_advanced.val == True:
                rotation_factor = 2 if self.is_half_rotation_advanced.val == True else 1
                sinos = find_center_shift(sinos, bar_thread=self.bar_thread_advanced, 
                                          type_sino=self.input_type, order_mode=self.orderbox_advanced.val, 
                                          clip_to_circle=self.clipcirclebox_advanced.val, device=device, rotation_factor=rotation_factor)
                
            elif self.manualalignbox_advanced.val == True:
                if self.orderbox_advanced.val == 0:
                    sinos = ndi.shift(sinos, (0, self.alignbox_advanced.val, 0), mode="nearest")
                elif self.orderbox_advanced.val == 1:
                    sinos = ndi.shift(sinos, (self.alignbox_advanced.val, 0, 0), mode="nearest")


            # Reconstruction process
            # if reconstructing only one slice
            if self.one_slice_advanced_mode.isChecked() and self.input_type == "3D":
                slices_reconstruction = [self.slices_advanced.val]
            # if reconstructing full volume or multiple slices
            elif self.multiple_slices_advanced.isChecked() and self.input_type == "3D":
                slices_reconstruction = range(self.slices_advanced.val)
            else:
                slices_reconstruction = range(self.h_advanced.Z)

            batch_start = slices_reconstruction[0]
            # if use GPU process in batch to improve performance
            if self.reconbox_advanced.val in {Rec_modes.FBP_GPU.value, Rec_modes.TOMODL_GPU.value, Rec_modes.UNET_GPU.value}:
                batch_process = self.batch_size_advanced.val
            else:
                batch_process = 1
            
            batch_end = batch_start + batch_process
            # add progressBar to track the reconstruction process
            self.bar_thread_advanced.start()
            self.bar_thread_advanced.max = slices_reconstruction[-1] + 1
            time_in = time()
            while batch_start <= slices_reconstruction[-1]:
                print("Reconstructing slices {} to {}".format(batch_start, batch_end), end="\r")
                zidx = slice(batch_start, batch_end)
                ####################### stacks reconstruction ############################
                if self.input_type == "3D":
                    if self.orderbox_advanced.val == 0:
                        optVolume[:, :, zidx] = self.h_advanced.reconstruct(sinos[:, :, zidx].transpose(1, 0, 2))
                    elif self.orderbox_advanced.val == 1:
                        optVolume[:, :, zidx] = self.h_advanced.reconstruct(sinos[:, :, zidx])

                ####################### 2D reconstruction ############################
                elif self.input_type == "2D":
                    if self.orderbox_advanced.val == 0:
                        optVolume[:, :, zidx] = self.h_advanced.reconstruct(sinos[:, :, zidx].transpose(1, 0, 2))

                    elif self.orderbox_advanced.val == 1:
                        optVolume[:, :, zidx] = self.h_advanced.reconstruct(sinos[:, :, zidx])

                self.bar_thread_advanced.value = batch_end
                self.bar_thread_advanced.run()
                batch_start = batch_end
                batch_end += batch_process

            print("Computation time total: {} s".format(round(time() - time_in, 3)))

            self.bar_thread_advanced.value = 0
            self.bar_thread_advanced.run()
            optVolume = np.rollaxis(optVolume, -1)
            
            #  change the scale instead of resizing ################################### TODO: change this to resizing
            if self.reshapebox_advanced.val:
                if (self.fullvolume_advanced_mode.isChecked() or self.multiple_slices_advanced.isChecked()) and self.input_type == "3D":
                    self.scale_image_advanced = [self.scale_image_advanced[0] / original_size * self.resizebox_advanced.val, self.scale_image_advanced[1], self.scale_image_advanced[2]]
                else:
                    self.scale_image_advanced = [1., 1.]
            else:
                if self.one_slice_advanced_mode.isChecked() and self.input_type == "3D":
                    self.scale_image_advanced = [1., 1.]

            self.bar_thread_advanced.value = 0
            self.bar_thread_advanced.run()
            self.bar_thread_advanced.quit()

            min_val = optVolume.min()
            max_val = optVolume.max()
            # save optVolume to tif file
            # tif.imwrite("optVolume_FBP_GPU.tif", optVolume)
            print("min: ", min_val, "max: ", max_val)
            # convert to uint16
            if self.output_conversion_advanced.val == True:
                optVolume = (optVolume - min_val) / (max_val - min_val) * (2**16 - 1)
                optVolume = optVolume.astype(np.uint16, copy=False)
                print("done converting to uint16")

            print("reconstruction shape: ", optVolume.shape)

            if self.one_slice_advanced_mode.isChecked() and self.input_type == "3D":
                print("scale image advanced: ", self.scale_image_advanced)
                return optVolume[self.slices_advanced.val]
            elif self.fullvolume_advanced_mode.isChecked() and self.input_type == "3D":
                return optVolume
            elif self.multiple_slices_advanced.isChecked() and self.input_type == "3D":
                return optVolume[range(self.slices_advanced.val)]
            else:
                return optVolume[0]


        _reconstruct_advanced()


    def get_sinos(self):
        try:
            return self.viewer.layers[self.imageRaw_name].data
        except:
            raise (KeyError(r"Please select a valid 3D image ($\theta$, q, z)"))

    def set_opt_processor_basic(self, *args):
        """
        Sets OPT reconstruction arguments
        """


        if hasattr(self, "h_basic"):
            
            self.h_basic.resize_val = Compression_modes[self.compression_basic.text].value
            self.h_basic.rec_process = self.reconbox_basic.val
            self.h_basic.order_mode = self.orderbox_basic.val
            self.h_basic.clip_to_circle = False
            self.h_basic.use_filter = True
            self.h_basic.filter_FBP = Filter_modes.RAMP.value
            size_compression = Compression_modes[self.compression_basic.text].value
            if self.reconbox_basic.val in {Rec_modes.FBP_GPU.value, Rec_modes.TOMODL_GPU.value, Rec_modes.UNET_GPU.value}:
                    if size_compression == 100:
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

    def set_opt_processor_advanced(self, *args):
        """
        Sets OPT reconstruction arguments
        """

        if hasattr(self, "h_advanced"):

            self.h_advanced.resize_val = self.resizebox_advanced.val
            self.h_advanced.rec_process = self.reconbox_advanced.val
            self.h_advanced.order_mode = self.orderbox_advanced.val
            self.h_advanced.clip_to_circle = self.clipcirclebox_advanced.val
            self.h_advanced.use_filter = True if self.filterbox_advanced.text != "NO" else False
            self.h_advanced.filter_FBP = Filter_modes[self.filterbox_advanced.text].value
            self.h_advanced.batch_size = self.batch_size_advanced.val
            self.h_advanced.invert_color = self.invert_color_advanced.val
            self.h_advanced.is_half_rotation = self.is_half_rotation_advanced.val
            self.h_advanced.iterations = self.iterations_advanced.val
            self.h_advanced.set_reconstruction_process()


    def start_opt_processor_basic(self):
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

        self.viewer.layers.events.inserted.connect(widget.reset_choices)
        self.viewer.layers.events.removed.connect(widget.reset_choices)
        _layout.addWidget(widget.native)


@magic_factory
def choose_layer(image: Image):
    pass  # TODO: substitute with a qtwidget without magic functions

"""
Created on Tue Feb 2 16:34:41 2023
@authors: Marcos Obando
"""

# %%
import os
from .processors.OPTProcessor import OPTProcessor
from .widget_settings import Settings, Combo_box
import gc
import torch
import tifffile as tif
# import processors
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
)
from qtpy.QtCore import Qt, QThread, Signal
from napari.layers import Image
import numpy as np
from napari.qt.threading import thread_worker
from magicgui.widgets import FunctionGui
from magicgui import magic_factory, magicgui
import warnings
from time import time
import scipy.ndimage as ndi
from enum import Enum
import cv2
from tqdm import tqdm
import numpy as np


def min_max_normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def flat_field_estimate(img, ratio_corners=0.03):

    height, width = img.shape
    # get corner size as 2% of the  dimension
    corner_size = int(min(height, width) * ratio_corners)

    # Extract the four corner regions (top-left, top-right, bottom-left, bottom-right)
    top_left = img[:corner_size, :corner_size]
    top_right = img[:corner_size, -corner_size:]
    bottom_left = img[-corner_size:, :corner_size]
    bottom_right = img[-corner_size:, -corner_size:]
    middle_left = img[height // 2 - corner_size // 2 : height // 2 + corner_size // 2, :corner_size]
    middle_right = img[height // 2 - corner_size // 2 : height // 2 + corner_size // 2, -corner_size:]
    corner_means = np.array(
        [
            top_left.mean(),
            top_right.mean(),
            bottom_left.mean(),
            bottom_right.mean(),
            middle_left.mean(),
            middle_right.mean(),
        ]
    )
    valid_corners = corner_means[
        (corner_means > np.percentile(corner_means, 10)) & (corner_means < np.percentile(corner_means, 90))
    ]
    flat_field_estimate = valid_corners.mean()

    return flat_field_estimate


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
    UNET_GPU = 3
    MODL_GPU = 4
    MODL_CPU = 5


class Order_Modes(Enum):
    Vertical = 0
    Horizontal = 1


class ReconstructionWidget(QTabWidget):

    name = "Reconstructor"

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.setup_ui()

        # self.viewer.dims.events.current_step.connect(self.select_index)
        self.bar_thread = BarThread(self)
        self.bar_thread.progressChanged.connect(self.progressBar.setValue)

    def setup_ui(self):


        def add_section(_layout, _title):
            splitter = QSplitter(Qt.Vertical)
            _layout.addWidget(splitter)
            _layout.addWidget(QLabel(_title))
        
        # Tab 1 - Basic settings and reconstruction
        
        # i) add a tab widget
        self.acquisition_params_widget = QWidget()
        self.addTab(self.acquisition_params_widget, "Basic setup")
        
        # ii) layout
        self.basic_reconstruction_layout = QVBoxLayout()
        self.basic_reconstruction_widget = QWidget()
        self.basic_reconstruction_layout.addWidget(QLabel("Basic reconstruction"))
        self.basic_reconstruction_layout.addWidget(self.basic_reconstruction_widget)
        
        self.choose_layer_widget = choose_layer()
        self.choose_layer_widget.call_button.visible = False
        self.add_magic_function(self.choose_layer_widget, self.basic_reconstruction_layout)
        select_button = QPushButton("Select image layer")
        select_button.clicked.connect(self.select_layer)
        self.basic_reconstruction_layout.addWidget(select_button)
        
        settings_layout = QVBoxLayout()
        add_section(settings_layout, "Settings")
        self.basic_reconstruction_layout.addLayout(settings_layout)
        self.createSettings(settings_layout)
        self.acquisition_params_widget.setLayout(self.basic_reconstruction_layout)

        # Tab 2 - Advanced settings
        self.advanced_reconstruction_params_widget = QWidget()
        self.advanced_reconstruction_params_layout = QVBoxLayout()
        self.addTab(self.advanced_reconstruction_params_widget, "Advanced reconstruction options")
        
        add_section(self.advanced_reconstruction_params_layout, "Settings")
        self.acquisition_params_widget.setLayout(self.advanced_reconstruction_params_layout)


    def createSettings(self, slayout):
        self.is_half_rotation = Settings(
            "Half-rotation", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor
        )

        self.registerbox = Settings(
            "Automatic axis alignment", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor
        )

        self.manualalignbox = Settings(
            "Manual axis alignment", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor
        )

        self.alignbox = Settings(
            "Axis shift",
            dtype=int,
            vmin=-500,
            vmax=500,
            initial=0,
            layout=slayout,
            write_function=self.set_opt_processor,
        )

        self.reshapebox = Settings(
            "Reshape volume", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor
        )

        self.resizebox = Settings(
            "Reconstruction size", dtype=int, initial=100, layout=slayout, write_function=self.set_opt_processor
        )

        self.iterations = Settings(
            "ToMoDL iterations", dtype=int, initial=3, layout=slayout, write_function=self.set_opt_processor
        )

        self.clipcirclebox = Settings(
            "Clip to circle", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor
        )

        self.filterbox = Settings(
            "Use filtering", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor
        )

        # create combobox for reconstruction method
        self.reconbox = Combo_box(
            name="Reconstruction method",
            initial=Rec_modes.FBP_CPU.value,
            choices=Rec_modes,
            layout=slayout,
            write_function=self.set_opt_processor,
        )
        # self.lambda_modl = Settings(
        #     "Lambda_MODL", dtype=float, initial=0.7, layout=slayout, write_function=self.set_opt_processor
        # )
        self.flat_correction = Settings(
            "Flat-field correction", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor
        )
        self.invert_color = Settings(
            "Invert colors", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor
        )
        self.fullvolume = Settings(
            "Reconstruct full volume", dtype=bool, initial=False, layout=slayout, write_function=self.set_opt_processor
        )
        self.batch_size = Settings(
            "Batch size", dtype=int, initial=1, layout=slayout, write_function=self.set_opt_processor
        )
        self.is_reconstruct_one = Settings(
            "Reconstruct only slices", dtype=bool, initial=True, layout=slayout, write_function=self.set_opt_processor
        )
        self.slices = Settings(
            "# of slices to reconstruct", dtype=int, initial=0, layout=slayout, write_function=self.set_opt_processor
        )

        self.orderbox = Combo_box(
            name="Rotation axis",
            initial=Order_Modes.Vertical.value,
            choices=Order_Modes,
            layout=slayout,
            write_function=self.set_opt_processor,
        )
        self.output_conversion = Settings(
            "16-bit conversion", dtype=bool, initial=True, layout=slayout, write_function=self.set_opt_processor
        )

        # add calculate psf button
        calculate_btn = QPushButton("Reconstruct")
        calculate_btn.clicked.connect(self.stack_reconstruction)
        slayout.addWidget(calculate_btn)

        self.progressBar = QProgressBar()
        slayout.addWidget(self.progressBar)

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

    def select_layer(self, sinos: Image):

        sinos = self.choose_layer_widget.image.value

        if sinos.data.ndim == 3 and sinos.data.shape[2] > 1:
            self.flat_field = flat_field_estimate(sinos.data[0])
            print(f"Flat-field estimate: {self.flat_field}")
            self.input_type = "3D"
            self.imageRaw_name = sinos.name
            sz, sy, sx = sinos.data.shape
            print(sz, sy, sx)
            if not hasattr(self, "h"):
                self.start_opt_processor()
            print(f"Selected image layer: {sinos.name}")
        else:
            self.input_type = "2D"
            self.imageRaw_name = sinos.name
            # add dim to the image
            # sinos.data = np.expand_dims(sinos.data, axis=0)
            sy, sx = sinos.data.shape
            print(sy, sx)
            if not hasattr(self, "h"):
                self.start_opt_processor()
            print(f"Selected image layer: {sinos.name}")

    def stack_reconstruction(self):
        self.scale_image = self.viewer.layers[self.imageRaw_name].scale
        
        def update_opt_image(stack):

            imname = "stack_" + self.imageRaw_name
            self.show_image(stack, fullname=imname, scale=self.scale_image)
            print("Stack reconstruction completed")
            gc.collect()
            torch.cuda.empty_cache()

        @thread_worker(
            connect={"returned": update_opt_image},
        )
        def _reconstruct():
            """
            ToDO: Link projections
            """

            if self.orderbox.val == 0 and self.input_type == "3D":
                sinos = np.moveaxis(np.float32(self.get_sinos()), 1, 2)
                self.h.theta, self.h.Q, self.h.Z = sinos.shape
            elif self.orderbox.val == 1 and self.input_type == "3D":
                sinos = np.moveaxis(np.float32(self.get_sinos()), 0, 1)
                self.h.Q, self.h.theta, self.h.Z = sinos.shape
            elif self.orderbox.val == 0 and self.input_type == "2D":
                sinos = np.float32(self.get_sinos().T)[..., None]
                self.h.theta, self.h.Q, self.h.Z = sinos.shape
            elif self.orderbox.val == 1 and self.input_type == "2D":
                sinos = np.float32(self.get_sinos().T)[..., None]
                self.h.Q, self.h.theta, self.h.Z = sinos.shape

            original_size = self.h.Q if self.clipcirclebox.val else int(np.floor(self.h.Q / np.sqrt(2)))

            if self.reshapebox.val:
                optVolume = np.zeros([self.resizebox.val, self.resizebox.val, self.h.Z], np.float32)
                sinos = self.h.resize(sinos, type_sino=self.input_type)
            else:
                optVolume = np.zeros([original_size, original_size, self.h.Z], np.float32)

            if self.flat_correction.val == True and self.input_type == "3D":
                sinos = sinos / self.flat_field

            # sinos = min_max_normalize(sinos)
            # Reconstruction process
            # if reconstructing only one slice
            if self.is_reconstruct_one.val == True and self.fullvolume.val == False and self.input_type == "3D":
                slices_reconstruction = [self.slices.val]
            # if reconstructing full volume or multiple slices
            elif self.input_type == "3D":
                slices_reconstruction = range(self.h.Z if self.fullvolume.val == True else self.slices.val)
            else:
                slices_reconstruction = [0]

            batch_start = slices_reconstruction[0]
            # if use GPU process in batch to improve performance
            if self.reconbox.val in {Rec_modes.FBP_GPU.value, Rec_modes.MODL_GPU.value, Rec_modes.UNET_GPU.value}:
                batch_process = self.batch_size.val
            else:
                batch_process = 1

            batch_end = batch_start + batch_process
            # add progressBar to track the reconstruction process
            self.bar_thread.start()
            self.bar_thread.max = slices_reconstruction[-1] + 1
            time_in = time()
            while batch_start <= slices_reconstruction[-1]:
                print("Reconstructing slices {} to {}".format(batch_start, batch_end), end="\r")
                zidx = slice(batch_start, batch_end)
                ####################### stacks reconstruction ############################
                if self.input_type == "3D":
                    if self.registerbox.val == True:


                        if self.orderbox.val == 0:
                            optVolume[:, :, zidx] = self.h.correct_and_reconstruct(sinos[:, :, zidx].transpose(1, 0, 2))
                        elif self.orderbox.val == 1:
                            optVolume[:, :, zidx] = self.h.correct_and_reconstruct(sinos[:, :, zidx])

                    elif self.manualalignbox.val == True:
                        if self.orderbox.val == 0:
                            optVolume[:, :, zidx] = self.h.reconstruct(
                                ndi.shift(sinos[:, :, zidx], (0, self.alignbox.val, 0), mode="nearest").transpose(
                                    1, 0, 2
                                )
                            )
                        elif self.orderbox.val == 1:
                            optVolume[:, :, zidx] = self.h.reconstruct(
                                ndi.shift(sinos[:, :, zidx], (self.alignbox.val, 0, 0), mode="nearest")
                            )

                    else:


                        if self.orderbox.val == 0:
                            optVolume[:, :, zidx] = self.h.reconstruct(sinos[:, :, zidx].transpose(1, 0, 2))
                            # save optVolume slice to see
                            # optVolume[:, :, zidx] = tif_image
                            # tif.imwrite(f"optVolume_slice_{zidx}.tif", tif_image)
                            # tif.imwrite("optVolume_slice.tif", optVolume[:, :, zidx])
                        elif self.orderbox.val == 1:
                            optVolume[:, :, zidx] = self.h.reconstruct(sinos[:, :, zidx])

                ####################### 2D reconstruction ############################
                elif self.input_type == "2D":

                    if self.registerbox.val == True:
                        if self.orderbox.val == 0:
                            optVolume[:, :, zidx] = self.h.correct_and_reconstruct(sinos[:, :, zidx].transpose(1, 0, 2))
                        elif self.orderbox.val == 1:
                            optVolume[:, :, zidx] = self.h.correct_and_reconstruct(sinos[:, :, zidx])

                    elif self.manualalignbox.val == True:
                        if self.orderbox.val == 0:
                            optVolume[:, :, zidx] = self.h.reconstruct(
                                ndi.shift(sinos[:, :, zidx], (0, self.alignbox.val, 0), mode="nearest").transpose(
                                    1, 0, 2
                                )
                            )
                        elif self.orderbox.val == 1:
                            optVolume[:, :, zidx] = self.h.reconstruct(
                                ndi.shift(sinos[:, :, zidx], (self.alignbox.val, 0, 0), mode="nearest")
                            )
                    else:
                        if self.orderbox.val == 0:
                            optVolume[:, :, zidx] = self.h.reconstruct(sinos[:, :, zidx].transpose(1, 0, 2))

                        elif self.orderbox.val == 1:
                            optVolume[:, :, zidx] = self.h.reconstruct(sinos[:, :, zidx])

                self.bar_thread.value = batch_end
                self.bar_thread.run()
                batch_start = batch_end
                batch_end += batch_process

            print("Computation time total: {} s".format(round(time() - time_in, 3)))

            self.bar_thread.value = 0
            self.bar_thread.run()
            optVolume = np.rollaxis(optVolume, -1)
            #  change the scale instead of resizing ################################### TODO: change this to resizing
            if self.reshapebox.val:
                if self.is_reconstruct_one.val == True and self.fullvolume.val == False and self.input_type == "3D":
                    self.scale_image = [self.scale_image[0] * original_size / self.resizebox.val, self.scale_image[1]* original_size / self.resizebox.val]
                elif self.fullvolume.val == True and self.input_type == "3D":
                    self.scale_image = [self.scale_image[0], self.scale_image[1] * original_size / self.resizebox.val, self.scale_image[2]* original_size / self.resizebox.val]
            
            # # convert resize volume to original size
            #     optVolume_resized = np.zeros([self.h.Z, original_size, original_size], np.float32)
            #     print("Resizing volume to original size")
            #     if self.fullvolume.val == False and self.is_reconstruct_one.val == True:
            #         optVolume_resized[self.slices.val] = cv2.resize(
            #             optVolume[self.slices.val], (original_size, original_size), interpolation=cv2.INTER_LINEAR
            #         )
            #     else:
            #         slices_resize = self.h.Z if self.fullvolume.val == True else self.slices.val
            #         self.bar_thread.max = slices_resize
            #         for i in tqdm(range(slices_resize)):
            #             optVolume_resized[i] = cv2.resize(
            #                 optVolume[i], (original_size, original_size), interpolation=cv2.INTER_LINEAR
            #             )
            #             self.bar_thread.value = i + 1
            #             self.bar_thread.run()
            #     optVolume = optVolume_resized
            #     del optVolume_resized, sinos

            self.bar_thread.value = 0
            self.bar_thread.run()
            self.bar_thread.quit()
            min_val = optVolume.min()
            max_val = optVolume.max()
            # save optVolume to tif file
            # tif.imwrite("optVolume_FBP_GPU.tif", optVolume)
            print("min: ", min_val, "max: ", max_val)
            # convert to uint16
            if self.output_conversion.val == True:
                optVolume = (optVolume - min_val) / (max_val - min_val) * (2**16 - 1)
                optVolume = optVolume.astype(np.uint16, copy=False)
                print("done converting to uint16")

            if self.is_reconstruct_one.val == True and self.fullvolume.val == False and self.input_type == "3D":
                return optVolume[self.slices.val]
            elif self.input_type == "3D":
                return optVolume
            else:
                return optVolume[0]

        _reconstruct()

    def get_sinos(self):
        try:

            return self.viewer.layers[self.imageRaw_name].data
        except:
            raise (KeyError(r"Please select a valid 3D image ($\theta$, q, z)"))

    def set_opt_processor(self, *args):
        """
        Sets OPT reconstruction arguments
        """

        if hasattr(self, "h"):

            self.h.resize_val = self.resizebox.val
            self.h.resize_bool = self.reshapebox.val
            self.h.register_bool = self.registerbox.val
            self.h.rec_process = self.reconbox.val
            self.h.order_mode = self.orderbox.val
            self.h.clip_to_circle = self.clipcirclebox.val
            self.h.use_filter = self.filterbox.val
            self.h.batch_size = self.batch_size.val
            self.h.invert_color = self.invert_color.val
            self.h.is_half_rotation = self.is_half_rotation.val
            self.h.iterations = self.iterations.val
            self.h.set_reconstruction_process()

    def start_opt_processor(self):
        self.isCalibrated = False

        if hasattr(self, "h"):
            self.stop_opt_processor()
            self.start_opt_processor()
        else:
            print("Reset")
            self.h = OPTProcessor()

    def stop_opt_processor(self):
        if hasattr(self, "h"):
            delattr(self, "h")

    def reset_processor(self, *args):

        self.isCalibrated = False
        self.stop_opt_processor()
        self.start_opt_processor()

    def add_magic_function(self, widget, _layout):

        self.viewer.layers.events.inserted.connect(widget.reset_choices)
        self.viewer.layers.events.removed.connect(widget.reset_choices)
        _layout.addWidget(widget.native)


@magic_factory
def choose_layer(image: Image):
    pass  # TODO: substitute with a qtwidget without magic functions

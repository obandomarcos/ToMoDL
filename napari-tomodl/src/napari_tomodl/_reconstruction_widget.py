"""
Created on Tue Feb 2 16:34:41 2023
@authors: Marcos Obando
"""
from .widget_settings import Setting, Combo_box
#import processors
import napari
from qtpy.QtWidgets import QVBoxLayout, QSplitter, QHBoxLayout, QWidget, QPushButton, QLineEdit
from qtpy.QtCore import Qt
from napari.layers import Image
import numpy as np
from napari.qt.threading import thread_worker
from magicgui.widgets import FunctionGui
from magicgui import magic_factory
import warnings
import time
from superqt.utils import qthrottled
from enum  import Enum


class ReconstructionWidget(QWidget):
    
    dr = 0.1
    
    def __init__(self, viewer:napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.create_ui()
        
    def create_ui(self):
        
        # initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # self.angles_number = Setting('angles', dtype=int, initial=1, layout=left_layout, write_function = self.reset_processor)

        self.choose_layer_widget = choose_layer()
        self.choose_layer_widget.call_button.visible = False
        self.add_magic_function(self.choose_layer_widget, layout)
        select_button = QPushButton('Select image layer')
        select_button.clicked.connect(self.select_layer)
        layout.addWidget(select_button)        
        #create spiboxes for size
        
        resizebox = QSpinBox()
        resizeLayout = QFormLayout()
        resizeLayout.addRow('Reconstruction size', resizebox)
        layout.addLayout(resizeLayout)
        self.resizebox = resizebox
        
        #create combobox for reconstruction method
        reconbox = QComboBox()
        
        reconbox.addItem("FBP (CPU)")
        reconbox.addItem("FBP (GPU)")
        reconLayout = QFormLayout()
        reconLayout.addRow('Reconstruction method', reconbox)
        layout.addLayout(reconLayout)
        self.reconbox = reconbox

        # add calculate psf button
        calculate_btn = QPushButton('Reconstruct')
        calculate_btn.clicked.connect(self.reconstruct)
        layout.addWidget(calculate_btn)
        
    def reconstruct(self):
        '''
        Reconstructs images with a certain method chosen from the window list
        '''
        self.generate_angles()

        reconstruct = np.rand.random(20, 100, 100)
        self.viewer.add_image(reconstruct,
                         name='Reconstruction',
                         colormap='twilight')
    
    def select_layer(self, image: Image):
        '''
        Selects a Image layer after checking that it contains raw projectrions
        as a 3D stack (angle,q,z).
        Stores the name of the image in self.imageRaw_name, which is used frequently in the other methods.
        Parameters
        ----------
        image : napari.layers.Image
            The image layer to process, it contains the raw data 
        '''

        image = self.choose_layer_widget.image.value
        if not isinstance(image, Image):
            raise(KeyError('Please select a image stack'))
        if hasattr(self,'imageRaw_name'):
            delattr(self,'imageRaw_name')
        data = image.data
        if data.ndim != 3:
            raise(KeyError('Please select a 3D image'))

    def generate_angles(self):
        '''
        Generate according angles for reconstruction
        '''

        self.angles = np.linspace(0., 180., 100, endpoint=False)

        pass

    def add_magic_function(self, widget, _layout):

        self.viewer.layers.events.inserted.connect(widget.reset_choices)
        self.viewer.layers.events.removed.connect(widget.reset_choices)
        _layout.addWidget(widget.native)
"""
Created on Tue Feb 2 16:34:41 2023
@authors: Marcos Obando
"""
#%%
import os 
os.chdir('.')

from processors import OPTProcessor
from widget_settings import Settings, Combo_box
#import processors
import napari
from qtpy.QtWidgets import QVBoxLayout, QSplitter, QHBoxLayout, QWidget, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QFormLayout, QComboBox, QLabel
from qtpy.QtCore import Qt
from napari.layers import Image
import numpy as np
from napari.qt.threading import thread_worker
from magicgui.widgets import FunctionGui
from magicgui import magic_factory, magicgui
import warnings
import time
from superqt.utils import qthrottled
from enum  import Enum


class ReconstructionWidget(QWidget):
    
    name = 'Reconstructor'
    
    def __init__(self, viewer:napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.setup_ui()
        # self.viewer.dims.events.current_step.connect(self.select_index)
    
    def setup_ui(self):

        # initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        def add_section(_layout,_title):
            from qtpy.QtCore import Qt
            splitter = QSplitter(Qt.Vertical)
            _layout.addWidget(splitter)
            # _layout.addWidget(QLabel(_title))
        

        image_layout = QVBoxLayout()
        add_section(image_layout,'Image selection')
        layout.addLayout(image_layout)
        
        self.choose_layer_widget = choose_layer()
        self.choose_layer_widget.call_button.visible = False
        self.add_magic_function(self.choose_layer_widget, image_layout)
        select_button = QPushButton('Select image layer')
        select_button.clicked.connect(self.select_layer)
        image_layout.addWidget(select_button)

        settings_layout = QVBoxLayout()
        add_section(settings_layout,'Settings')
        layout.addLayout(settings_layout)
        self.create_Settings(settings_layout)



    def create_Settings(self, slayout):
        
        self.resizebox = Settings('Reconstruction size', dtype=int, initial=100, layout=slayout, 
                              write_function = self.reset_processor)
        
        #create combobox for reconstruction method
        reconbox = QComboBox()
        reconbox.addItems(["FBP (CPU)", "FBP (GPU)"])
        reconLayout = QFormLayout()
        reconLayout.addRow('Reconstruction method', reconbox)
        slayout.addLayout(reconLayout)
        self.reconbox = reconbox

        # add calculate psf button
        calculate_btn = QPushButton('Reconstruct')
        calculate_btn.clicked.connect(self.reconstruct)
        slayout.addWidget(calculate_btn)
    
    def select_index(self, val = 0):
        pass
    
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
        
        if image.data.ndim == 3:
            self.imageRaw_name = image.name
            sz,sy,sx = image.data.shape
            assert sy == sx, 'Non-square images are not supported'
            if not hasattr(self, 'h'): 
                self.start_opt_processor()
            print(f'Selected image layer: {image.name}')

    def generate_angles(self):
        '''
        Generate according angles for reconstruction
        '''

        self.angles = np.linspace(0., 180., 100, endpoint=False)

        pass
    
    def start_opt_processor(self):     
        self.isCalibrated = False
        
        if hasattr(self, 'h'):
            self.stop_opt_processor()
            self.start_opt_processor()
        else:
            
            self.h = OPTProcessor() 

    def stop_opt_processor(self):
        if hasattr(self, 'h'):
            delattr(self, 'h')

    def reset_processor(self,*args):
        
        self.isCalibrated = False
        self.stop_opt_processor()
        self.start_opt_processor() 

    def add_magic_function(self, widget, _layout):

        self.viewer.layers.events.inserted.connect(widget.reset_choices)
        self.viewer.layers.events.removed.connect(widget.reset_choices)
        _layout.addWidget(widget.native)

@magic_factory
def choose_layer(image: Image):
        pass #TODO: substitute with a qtwidget without magic functions


#%%
if __name__ == '__main__':
   
    viewer = napari.Viewer()
    opt_widget = ReconstructionWidget(viewer)
    selection = magicgui(opt_widget.select_layer)

    viewer.window.add_dock_widget(opt_widget, name = 'OPT reconstruction')

    napari.run() 
"""
Created on Tue Feb 2 16:34:41 2023
@authors: Marcos Obando
"""
#%%
import os 
os.chdir('.')

from processors.OPTProcessor import OPTProcessor
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
        self.createSettings(settings_layout)

    def createSettings(self, slayout):
        
        self.registerbox = Settings('Align axis',
                                  dtype=bool, 
                                  layout=slayout, 
                                  write_function = self.set_opt_processor)
        

        self.reshapebox = Settings('Reshape volume',
                                  dtype=bool,
                                  initial = True, 
                                  layout=slayout, 
                                  write_function = self.set_opt_processor)        


        self.resizebox = Settings('Reconstruction size',
                                  dtype=int, 
                                  initial=100, 
                                  layout=slayout, 
                                  write_function = self.set_opt_processor)
        
        #create combobox for reconstruction method
        reconbox = Combo_box(name ='Reconstruction method',
                             initial = 'FBP (CPU)',
                             choices = ["FBP (CPU)", "FBP (GPU)"],
                             layout = slayout,
                             write_function = self.set_opt_processor)
        self.reconbox = reconbox

        # add calculate psf button
        calculate_btn = QPushButton('Reconstruct')
        calculate_btn.clicked.connect(self.stack_reconstruction)
        slayout.addWidget(calculate_btn)

    def show_image(self, image_values, fullname, **kwargs):
        
        if 'scale' in kwargs.keys():    
            scale = kwargs['scale']
        else:
            scale = [1.]*image_values.ndim
        
        if 'hold' in kwargs.keys() and fullname in self.viewer.layers:
            
            self.viewer.layers[fullname].data = image_values
            self.viewer.layers[fullname].scale = scale
        
        else:  
            layer = self.viewer.add_image(image_values,
                                            name = fullname,
                                            scale = scale,
                                            interpolation = 'bilinear')
            return layer

    def select_layer(self, sinos:Image):
        
        
        sinos = self.choose_layer_widget.image.value
        if sinos.data.ndim == 3:
            
            self.imageRaw_name = sinos.name
            sz,sy,sx = sinos.data.shape
            print(sz, sy, sx)
            if not hasattr(self, 'h'): 
                self.start_opt_processor()
            print(f'Selected image layer: {sinos.name}')
        
    def stack_reconstruction(self):
        
        def update_opt_image(stack):
            
            imname = 'stack_' + self.imageRaw_name
            self.show_image(stack, fullname=imname)
                
            
            print('Stack reconstruction completed')
            
        
        @thread_worker(connect={'returned':update_opt_image})
        def _reconstruct():
            
            sinos = np.float32(self.get_sinos())
        
            theta, Q, Z = sinos.shape

            if self.reshapebox.val == True:
                
                optVolume = np.zeros([
                    self.resizebox.val, self.resizebox.val, Z], np.float32)

            else:
                
                optVolume = np.zeros([
                    np.int(Q//np.sqrt(2)), np.int(Q//np.sqrt(2)), Z], np.float32)

            for zidx in range(Z):
                
                optVolume[:,:, zidx] = self.h.reconstruct(sinos[:,:,zidx])
            
            return np.rollaxis(optVolume, 0, 2)

        _reconstruct()
    
    def get_sinos(self):
        try:
            print(self.imageRaw_name)
            return self.viewer.layers[self.imageRaw_name].data
        except:
             raise(KeyError('Please select a valid 3D image (z,y,x)'))
    
    def set_opt_processor(self, *args):
        '''
        Sets OPT reconstruction arguments
        '''

        if hasattr(self, 'h'):
            
            self.h.resizeVal = self.resizebox.val
            self.h.resizeBool = self.reshapebox.val
            self.h.registerBool = self.registerbox.val
            self.h.recProcess = self.reconbox.current_data 
            
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

    viewer.window.add_dock_widget(opt_widget, name = 'OPT reconstruction')

    napari.run()

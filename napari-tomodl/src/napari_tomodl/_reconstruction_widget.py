"""
Created on Tue Feb 2 16:34:41 2023
@authors: Marcos Obando
"""
#%%
import os 
from .processors.OPTProcessor import OPTProcessor
from .widget_settings import Settings, Combo_box
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
from time import time
import scipy.ndimage as ndi
from enum  import Enum
import cv2 

class Rec_modes(Enum):
    FBP_CPU = 0
    FBP_GPU = 1
    TWIST_CPU = 2
    UNET_GPU = 3
    MODL_GPU = 4

class Order_Modes(Enum):
    T_Q_Z = 0
    Q_T_Z = 1

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
        
        self.registerbox = Settings('Automatic axis alignment',
                                  dtype=bool,
                                  initial=True, 
                                  layout=slayout, 
                                  write_function = self.set_opt_processor)
        
        self.reshapebox = Settings('Reshape volume',
                                  dtype=bool,
                                  initial = True, 
                                  layout=slayout, 
                                  write_function = self.set_opt_processor)        

        self.clipcirclebox = Settings('Clip to circle',
                            dtype=bool,
                            initial = False, 
                            layout=slayout, 
                            write_function = self.set_opt_processor)        

        self.filterbox = Settings('Use filtering',
                            dtype=bool,
                            initial = False, 
                            layout=slayout, 
                            write_function = self.set_opt_processor) 
        
        self.manualalignbox = Settings('Manual axis alignment',
                            dtype=bool,
                            initial = False, 
                            layout=slayout, 
                            write_function = self.set_opt_processor) 
        
        self.alignbox = Settings('Axis shift',
                            dtype=int,
                            vmin=-500,
                            vmax=500, 
                            initial=0, 
                            layout=slayout, 
                            write_function = self.set_opt_processor)
        
        self.resizebox = Settings('Reconstruction size',
                                  dtype=int, 
                                  initial=100, 
                                  layout=slayout, 
                                  write_function = self.set_opt_processor)
        
        #create combobox for reconstruction method
        self.reconbox = Combo_box(name ='Reconstruction method',
                             initial = Rec_modes.FBP_GPU.value,
                             choices = Rec_modes,
                             layout = slayout,
                             write_function = self.set_opt_processor)
        
        self.orderbox = Combo_box(name ='Order',
                             initial = Order_Modes.Q_T_Z.value,
                             choices = Order_Modes,
                             layout = slayout,
                             write_function = self.set_opt_processor)

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
                                            interpolation2d = 'linear')
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
            '''
            ToDO: Link projections
            '''

            sinos = np.moveaxis(np.float32(self.get_sinos()), 0, 1)
            
            if self.orderbox.val == 0:
                self.h.theta, self.h.Q, self.h.Z = sinos.shape
            elif self.orderbox.val == 1:
                self.h.Q, self.h.theta, self.h.Z = sinos.shape

            if self.reshapebox.val == True:
                
                optVolume = np.zeros([
                    self.resizebox.val, self.resizebox.val, self.h.Z], np.float32)
                sinos = self.h.resize(sinos)

            else:
                
                optVolume = np.zeros([
                    int(np.ceil(self.h.Q/np.sqrt(2))), int(np.ceil(self.h.Q/np.sqrt(2))), self.h.Z], np.float32)

            time_in = time()

            for zidx in range(self.h.Z):
                
                if self.registerbox.val == True:
                    
                    sinos[:,:,zidx] = cv2.normalize(sinos[:,:,zidx], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    optVolume[:,:,zidx] = self.h.correct_and_reconstruct(sinos[:,:,zidx])
                    optVolume[:,:,zidx] = cv2.normalize(optVolume[:,:,zidx], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                elif self.manualalignbox.val == True:

                    sinos[:,:,zidx] = cv2.normalize(sinos[:,:,zidx], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    if self.orderbox.val == 0:
                        optVolume[:,:, zidx] = self.h.reconstruct(ndi.shift(sinos[:,:,zidx], (0, self.alignbox.val), mode = 'nearest'))
                    elif self.orderbox.val == 1:
                        optVolume[:,:, zidx] = self.h.reconstruct(ndi.shift(sinos[:,:,zidx], (self.alignbox.val, 0), mode = 'nearest'))

                    optVolume[:,:,zidx] = cv2.normalize(optVolume[:,:,zidx], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                else:
                    
                    sinos[:,:,zidx] = cv2.normalize(sinos[:,:,zidx], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    optVolume[:,:, zidx] = self.h.reconstruct(sinos[:,:,zidx])
                    optVolume[:,:,zidx] = cv2.normalize(optVolume[:,:,zidx], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                  
                    
            print('Tiempo de cómputo total: {} s'.format(round(time()-time_in, 3)))
            return np.rollaxis(optVolume, -1)

        _reconstruct()
    
    def get_sinos(self):
        try:
            
            return self.viewer.layers[self.imageRaw_name].data
        except:
             raise(KeyError(r'Please select a valid 3D image ($\theta$, q, z)'))
    
    def set_opt_processor(self, *args):
        '''
        Sets OPT reconstruction arguments
        '''

        if hasattr(self, 'h'):
            
            self.h.resize_val = self.resizebox.val
            self.h.resize_bool = self.reshapebox.val
            self.h.register_bool = self.registerbox.val
            self.h.rec_process = self.reconbox.val
            self.h.order_mode = self.orderbox.val
            self.h.clip_to_circle = self.clipcirclebox.val
            self.h.use_filter = self.filterbox.val
            print('set', self.h.rec_process)
            self.h.set_reconstruction_process()

    def start_opt_processor(self):     
        self.isCalibrated = False
        
        if hasattr(self, 'h'):
            self.stop_opt_processor()
            self.start_opt_processor()
        else:
            print('Reset')
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


import napari
from _reconstruction_widget import ReconstructionWidget

if __name__ == '__main__':
   
    viewer = napari.Viewer()

    opt_widget = ReconstructionWidget(viewer)

    viewer.window.add_dock_widget(opt_widget, name = 'OPT reconstruction')

    napari.run()
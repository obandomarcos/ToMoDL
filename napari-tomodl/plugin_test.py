import napari
from src.napari_tomodl._reconstruction_widget_2 import ReconstructionWidget

if __name__ == '__main__':
   
    viewer = napari.Viewer()

    opt_widget = ReconstructionWidget(viewer)

    viewer.window.add_dock_widget(opt_widget, name = 'napari-ToMoDL')

    napari.run()
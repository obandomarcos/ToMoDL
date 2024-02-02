try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Marcos Obando"
__credits__ = "Teresa Correia and Germán Mato"
__email__ = "marcos.obando@ib.edu.ar"


from ._reconstruction_widget import ReconstructionWidget
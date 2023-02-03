import sys
import socket

marcos_computer_path = '/home/obanmarcos/Balseiro/DeepOPT' 
marcos_computer_path_datasets = '/home/obanmarcos/Balseiro/DeepOPT/datasets/'
marcos_computer_path_metrics = '/home/obanmarcos/Balseiro/DeepOPT/metrics/' 
marcos_computer_path_models = '/home/obanmarcos/Balseiro/DeepOPT/models/' 
german_computer_path = '/home/marcos/DeepOPT'
german_computer_path_datasets = '/data/marcos'
ariel_computer_path = '/home/marcos/DeepOPT'
ariel_computer_path_datasets = '/Datos/DeepOPT'

def where_am_i(path = None):

    if socket.gethostname() in ['copote', 'copito']:
        if path == 'datasets':
            return marcos_computer_path_datasets
        elif path == 'models':
            return marcos_computer_path_models
        elif path == 'metrics':
            return marcos_computer_path_metrics
        else:
            return marcos_computer_path

    elif socket.gethostname() == 'cabfst42':
        if path == 'datasets':
            return german_computer_path_datasets
        else:
            return german_computer_path
    elif socket.gethostname() == 'gpu1':
        if path == 'datasets':
            return ariel_computer_path_datasets
        else:   
            return ariel_computer_path
    else:
        print('Computer not found')
        sys.exit(0)


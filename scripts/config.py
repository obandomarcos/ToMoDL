import sys
import socket

marcos_computer_path = '/home/marcos/Balseiro/DeepOPT' 
german_computer_path = '/home/marcos/DeepOPT'
ariel_computer_path = '/home/marcos/DeepOPT'

def where_am_i():

    if socket.gethostname() in ['copote', 'copito']:
        return marcos_computer_path
    elif socket.gethostname() == 'cabfst42':
        return german_computer_path
    elif socket.gethostname() == 'gpu1':
        return ariel_computer_path
    else:
        print('Computer not found')
        sys.exit(0)


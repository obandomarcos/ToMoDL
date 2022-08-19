import sys

marcos_computer_path = '/home/marcos/Balseiro/DeepOPT' 
german_computer_path = '/home/marcos/DeepOPT'

def where_am_i():

    computer = input('Where am I:')

    if computer == 'marcos':
        return marcos_computer_path
    elif computer == 'german':
        return german_computer_path
    else:
        print('Computer not found')
        sys.exit(0)


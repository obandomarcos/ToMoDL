import sys
import socket

marcos_computer_path = "/home/nhattm/ToMoDL/ToMoDL/"
# marcos_computer_path_datasets = "/home/obanmarcos/Balseiro/DeepOPT/datasets/"
marcos_computer_path_datasets = "/home/nhattm/ToMoDL/datasets/"
marcos_computer_path_metrics = "/home/nhattm/ToMoDL/ToMoDL/metrics/"
marcos_computer_path_models = "/home/nhattm/ToMoDL/ToMoDL/models/"

german_computer_path = "/home/marcos/DeepOPT/"
german_computer_path_datasets = "/data/marcos/datasets/"
ariel_computer_path = "/home/nhattm/ToMoDL"
ariel_computer_path_datasets = "/home/nhattm/ToMoDL/datasets/"
# print(socket.gethostname())

def where_am_i(path=None):

    if socket.gethostname() in ["copote", "copito", "qbi1"]:
        if path == "datasets":
            return marcos_computer_path_datasets
        elif path == "models":
            return marcos_computer_path_models
        elif path == "metrics":
            return marcos_computer_path_metrics
        else:
            return marcos_computer_path

    elif socket.gethostname() == "cabfst42":
        if path == "datasets":
            return german_computer_path_datasets
        else:
            return german_computer_path
    elif socket.gethostname() == "gpu1":
        if path == "datasets":
            return ariel_computer_path_datasets
        else:
            return ariel_computer_path
    else:
        print("Computer not found")
        sys.exit(0)


print(where_am_i("datasets"))

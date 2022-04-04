#!/usr/bin/python
from datetime import datetime
import threading
import time
import yaml
import os
import sys
import subprocess
from multiprocessing import Process
from extras.create_mnist_partitions import create_mnist_partitions
from extras.create_cifar10_dataset import create_cifar10_partitions
from extras.create_cifar100_dataset import create_cifar100_partitions


def run_container(cmd):
    subprocess.call(cmd, shell=True)


def start_clients_docker():
    try:
        for i in range(1, 6):
            with open("docker/client-gpu.yaml", 'r') as file:
                config1 = dict(yaml.safe_load(file))
            print(list(config1["services"].keys()))
            config1["services"]["client" + str(i)] = config1["services"].pop(list(config1["services"].keys())[0])
            config1["services"]["client" + str(i)]["ports"][0] = "809" + str(i) + ":809" + str(i)
            config1["services"]["client" + str(i)]["container_name"] = "client" + str(i)
            config1["services"]["client" + str(i)][
                "command"] = "sh -c 'pip install -r requirements.txt && python client.py " + "data/clients/" + str(
                i) + "/settings-client.yaml'"

            with open('docker/client-gpu.yaml', 'w') as f:
                yaml.dump(config1, f)

            # with open("settings/settings-client.yaml", 'r') as file:
            #     config = dict(yaml.safe_load(file))
            # config["client"]["port"] = 8090 + i
            # config["training"]["data_path"] = "data/clients/" + str(i) + "/mnist.npz"
            # config["training"]["global_model_path"] = "data/clients/" + str(i) + "/weights.npz"
            # with open("data/clients/" + str(i) + "/settings-client.yaml", 'w') as f:
            #     yaml.dump(config, f)
            threading.Thread(target=run_container,
                             args=(
                                 "docker-compose -f docker/client-gpu.yaml up >> data/clients/" + str(i) + "/log.txt",),
                             daemon=True).start()
            time.sleep(10)
    except Exception as e:
        print(e)


def start_clients():

    # setting up the clients on different gpus
    try:
        available_gpus = ["cuda:0", "cuda:2", "cuda:3", "cuda:3", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        for i in range(1):
            Process(target=run_container,
                    args=("python Client/client.py --gpu=" + available_gpus[i],),
                    daemon=True).start()
            time.sleep(3)
    except Exception as e:
        print(e)


def start_clients_slurm(total_clients):
    try:
        for i in range(1, total_clients + 1):
            a_file = open("batchscripts/start_client_1.sh", "r")
            list_of_lines = a_file.readlines()
            list_of_lines[-1] = "mpirun -np 1 $PYTHON Client/client.py --client_id=" + str(i)

            a_file = open("batchscripts/start_client_1.sh", "w")
            a_file.writelines(list_of_lines)
            a_file.close()
            Process(target=run_container, args=("sbatch batchscripts/start_client_1.sh",), daemon=True).start()
            time.sleep(3)
    except Exception as e:
        print(e)


def create_dataset_partitions(common_config, no_of_clients):
    if common_config["training"]["dataset"] == "mnist":
        create_mnist_partitions(no_of_clients)
    elif common_config["training"]["dataset"] == "cifar10":
        create_cifar10_partitions(no_of_clients)
    elif common_config["training"]["dataset"] == "cifar100":
        create_cifar100_partitions(no_of_clients)


# starting the clients and partitioning the dataset among different clients
if __name__ == '__main__':

    # read the total clients
    if len(sys.argv) < 2:
        no_of_clients = 8
    else:
        no_of_clients = int(sys.argv[1])
    with open('settings/settings-common.yaml', 'r') as file:
        try:
            common_config = dict(yaml.safe_load(file))
        except yaml.YAMLError as error:
            print('Failed to read model_config from settings file', flush=True)
            raise error
    
    # starting the clients on different gpus
    if len(sys.argv) > 2 and sys.argv[2] == "slurm":
        start_clients_slurm(no_of_clients)
    else:
        start_clients()

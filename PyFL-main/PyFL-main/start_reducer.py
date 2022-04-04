#!/usr/bin/python
import os
import time
import socket
import threading
import subprocess
from multiprocessing import Process
import yaml


def run_container(cmd):
    subprocess.call(cmd, shell=True)


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def start_reducer_old():
    threading.Thread(target=run_container, args=("docker-compose -f docker/minio.yaml up",), daemon=True).start()
    time.sleep(5)
    threading.Thread(target=run_container, args=("docker-compose -f docker/reducer.yaml up >> data/reducer/log.txt",),
                     daemon=True).start()
    time.sleep(5)


def start_reducer():
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists('data/reducer'):
        os.mkdir('data/reducer')
    if not os.path.exists('data/minio_logs'):
        os.mkdir('data/minio_logs')
    if not os.path.exists("minio_data"):
        os.mkdir("minio_data")
    subprocess.call("fuser -k 9000/tcp", shell=True)
    Process(target=run_container,
            args=("./minio server minio_data/ --console-address \":9001\" >> data/minio_logs/minio_logs.txt",),
            daemon=True).start()
    time.sleep(5)
    with open("settings/settings-common.yaml", 'r') as file:
        config = dict(yaml.safe_load(file))
    config["storage"]["storage_config"]["storage_hostname"] = get_local_ip()
    config["training_identifier"]["id"] = str(1 + int(config["training_identifier"]["id"]))
    with open("settings/settings-common.yaml", 'w') as f:
        yaml.dump(config, f)
    Process(target=run_container, args=("python Reducer/reducer.py",),
            daemon=True).start()


start_reducer()
time.sleep(5)

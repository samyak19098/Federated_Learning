import os
import sys
import time

from sklearn.metrics import homogeneity_score
from sklearn.model_selection import train_test_split

# from FL import X_test, X_train, Y_test, Y_train

sys.path.append(os.getcwd())
import io
import yaml
import json
import socket
import subprocess
from contextlib import closing
from multiprocessing import Process
# from helper.pytorch.pytorch_helper import PytorchHelper
# from model.pytorch_model_trainer import weights_to_np
# from model.pytorch.pytorch_models import create_seed_model
from Reducer.reducer_rest_service import ReducerRestService
import numpy as np

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


class Reducer:

    def __init__(self, config=None):

        self.reducer_config = config
        self.port = 8080
        self.hostname = get_local_ip()
        self.run_url = self.hostname + ':' + str(self.port)
        self.global_model_path = os.getcwd() + '/data/Reducer' + 'global_model.npy'
        
        total_data = list(np.load(os.getcwd() + '/data/Data.npy'))
        labels = list(np.load(os.getcwd() + '/data/label.npy'))
    
        train_x, test_x, train_y, test_y = train_test_split(total_data, labels, test_size=0.2)
        n_samples = len(train_x)
        # model_data = {'train_x':train_x]
        model_data = {'train_x':train_x, 'test_x':test_x, \
                    'train_y':train_y, 'test_y':test_y, 'n_train_samples':n_samples}
        self.reducer_rest_config = {'hostname' : self.hostname, 'port' : self.port, 'run_url': self.run_url, 'global_model_path':self.global_model_path, 'model_data':model_data }
        self.rest = ReducerRestService(self.reducer_rest_config)
        self.clients = {}
        sys.stdout = open(os.getcwd() + '/logs/reducer.txt', 'w')


    def run(self):
        self.rest.run()


if __name__ == "__main__":
    reducer = Reducer()
    reducer.run()

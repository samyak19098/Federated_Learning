from distutils.command.config import config
import socket
from contextlib import closing
import os
import time
import uuid
import json
import logging
import threading
import numpy as np
import requests as r
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from xgboost import train


class Client:
    def __init__(self, hostname, port, client_id, client_num):
        self.hostname = hostname
        self.port = port
        self.client_id = client_id
        self.client_num = client_num
        self.run_url = self.hostname + ':' + str(self.port)
        self.status = 'idle'

    def send_round_start_request(self, round_num, global_model, epochs):
        try:
            for i in range(10):
                retval = r.get(
                    "{}?round_num={}&global_model={}&epochs={}".format(
                        'http://' + self.run_url + '/startRound',
                        round_num, global_model,
                        epochs))
                if retval.json()['status'] == "started":
                    self.status = "training"
                    return True
                time.sleep(5)
            return False
        except Exception as e:
            print("Error while send_round_start_request ", e, flush=True)
            return False
    def set_client_data_for_training(self, data_range):
        try:
            for i in range(10):
                retval = r.get(
                    "{}?start_ids={}&end_idx={}".format(
                        'http://' + self.run_url + '/setClientData',
                        data_range[0], data_range[1]))
                if retval.json()['status'] == "set":
                    # self.status = "training"
                    return True
                time.sleep(5)
            return False
        except Exception as e:
            print("Error while set_client_data_for_training ", e, flush=True)
            return False
class ReducerRestService:

    def __init__(self, config):

        self.reducer_rest_config = config
        self.port = config['port']
        self.global_model_path = config['global_model_path']
        self.hostname = config['hostname']
        self.status = 'idle'
        self.clients = {}
        self.client_ids = []
        self.unique_clients = 0
        self.client_round_complete_status = {}

    def run(self):

        app = Flask(__name__)

        @app.route('/')
        def reducer_status():
            return self.status
        
        @app.route('/info')
        def show_server_info():
            s = f'-> Server status = {self.status}\n-> Clients = {self.clients}\n-> Client ids = {self.client_ids}'
            return s

        @app.route('/connectClient')
        def connect_client():

            client_id = request.args.get('client_id', None)
            client_port = request.args.get('port', None)
            client_hostname = request.args.get('client_hostname', None)

            print(f"Client with details : id = {client_id}, port = {client_port}, hostname = {client_hostname} trying to connect", flush=True)

            if(client_id not in self.client_ids):
                self.unique_clients += 1
                client_number = self.unique_clients
                self.clients[client_hostname + ':' + str(client_port)] = Client(client_hostname, client_port, client_id, client_number)
                self.client_round_complete_status[client_hostname + ':' + str(client_port)] = False
                print(f"Client with detials : id = {client_id}, port = {client_port}, hostname = {client_hostname} connected", flush=True)
                return jsonify({
                    'status':'connected',
                    'client_number':client_number
                })
            return True
        @app.route('/tempReducer')
        def temp_request_processor():
            a = int(request.args.get('a', None))
            b = int(request.args.get('b', None))
            return str(a + b)
        @app.route('/training')
        def start_training():
            print("request for training received", flush=True)
            if self.status == "training":
                return jsonify({"status": "Training already running!!"})
            training_config = {
                "rounds": int(request.args.get('rounds', '1')),
                # "round_time": int(request.args.get('round_time', '200')),
                "epochs": int(request.args.get('epochs', '2')),
                "global_model": self.global_model_path
            }
            print(f"Starting training now", flush=True)
            self.train(training_config)
            print(f"Training call completed", flush=True)
            self.status = "training"
            ret = {
                'status': "Training started"
            }
            return jsonify(ret)
        
        app.debug = True
        app.run(host='0.0.0.0', port=self.port)


    def train(self, training_config):

        num_rounds = training_config['rounds']
        num_epochs = training_config['epochs']
        
        num_clients_participating = len(self.clients)
        size = self.reducer_rest_config['model_data']['n_train_samples'] // num_clients_participating
        idx = 0
        for client_addr, client in self.clients.items():
            start_idx = idx*size
            if(idx == num_clients_participating - 1):
                data_range = [(idx * size), -1]
            else:
                data_range = [start_idx, start_idx + size]
            client.set_client_data_for_training(data_range)
            idx += 1
        for i in range(num_rounds):
            print('------ Round ', i, ' -----', flush=True)
            # idx = 0
            for client_addr, client in self.clients.items():
                client.send_round_start_request(i, self.global_model_path, num_epochs)
            


        
    # def train(self, config):


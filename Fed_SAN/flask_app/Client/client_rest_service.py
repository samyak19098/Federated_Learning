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

class ClientRestService:

    def __init__(self, config):
        self.hostname = config['hostname']
        self.port = config['client_port']
        self.server = config['server']
        self.status = "Participating"
        self.client_model = config['client_model']
    def run(self):
        
        app = Flask(__name__)

        @app.route('/')
        def status():
            return 'Running'
        @app.route('/temp')
        def temp():
            return 'in temp'
        
        @app.route('/startRound')
        def start_round():
            """used by the reducer to request client to start a round as per the given parameters"""
            round_config = {
                "round_num": request.args.get('round_num', None),
                "global_model": request.args.get('global_model', None),
                "epochs": int(request.args.get('epochs', "1"))
            }
            self.run_client_round(round_config)
            # self.round_thread = threading.Thread(target=self.run_round, args=(config,))
            # self.stop_round_event.clear()
            # self.round_thread.start()
            ret = {
                'status': "started"
            }
            return jsonify(ret)
        @app.route('/setClientData')
        def set_client_data():
            print([int(request.args.get('start_idx', None)), int(request.args.get('end_idx', None))],flush=True)
            self.client_data_range = [int(request.args.get('start_idx', None)), int(request.args.get('end_idx', None))]
            ret = {
                'status':'set'
            }
            print(ret, flush=True)
            return jsonify(ret)
        
        # app.debug = True
        app.run(host='0.0.0.0', port=self.port)
    
    def run_client_round(self, round_config):

        round_num = round_config['round_num']
        num_epochs = round_config['epochs']
        global_model_path = round_config['global_model']
        data_range = self.client_data_range
        self.client_model.set_data_range(data_range)
        self.client_model.load_data()
        client_updated_weights = self.client_model.train()
        print(client_updated_weights, flush=True)
        np.save(os.getcwd()+f'/data/Client/{self.port}'+'.npy', client_updated_weights)

        run_url = self.hostname + ":" + str(self.port)
        print("HELLO", flush=True)
        self.server.send_round_complete_request(round_num, run_url)
    








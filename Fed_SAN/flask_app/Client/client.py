import argparse
import os
import subprocess
import sys
import uuid

sys.path.append(os.getcwd())
import time
import yaml
import json
import socket
import threading
import requests as r
from contextlib import closing
from Client.client_rest_service import ClientRestService
from Models.model import Model


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

class Reducer:
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.run_url = hostname + ':' + str(self.port)
        self.connected = False

    def add_client_to_server(self, client_config):
        try:
            for i in range(1):
                print(f"Connecting with Reducer at {self.run_url}")
                ret_val = r.get("{}?client_id={}&port={}&client_hostname={}".format('http://' + self.run_url + '/connectClient',
                                                                                    client_config['client_id'], client_config['port'],
                                                                                    client_config['hostname']))
                if(ret_val.json()['status'] == "connected"):
                    print("Client added successfully")
                    self.connected = True
                    return True, ret_val.json()['client_number']
                #time.sleep(2)
            print("unsuccessful")
            return False, -1
        except Exception as e:
            self.connected = False
            return False, -1

    def send_round_complete_request(self, round_id):
        print(round_id)
        try:
            for i in range(10):
                print(f"Notifying Reducer that training at client is done")
                ret_val = r.get("{}?round_id={}&client_id={}".format('http://' + self.run_url + '/roundCompletedByClient',round_id, self.id))

                print(ret_val.json()['status'],flush=True)         
                if ret_val.json()['status'] == "Success":
                    print("Round ended successfully and notification received by server successfully", flush=True)
                    return True
                time.sleep(2)
            return False
        except Exception as error:
            print("Error while send_round_complete_request ", error)
            return False
    

 
class Client:

    def __init__(self, args=None):
        self.client_id = uuid.uuid4()
        self.port = find_free_port()
        self.hostname = get_local_ip()
        self.run_url = self.hostname + ':' + str(self.port)
        self.reducer = Reducer(self.hostname, 8080)
        self.client_config = {}
        self.client_config["client"] = {
            'client_id':self.client_id,
            'port':self.port,
            'hostname':self.hostname
        }
        print(self.client_config['client'])
        self.client_model = Model(shape=3, dataset_path=(os.getcwd()+'/data/Data.npy'), labels_file_path=(os.getcwd()+'/data/label.npy'), learning_rate=0.1, reg=0.1, epochs = 10)
        self.rest = ClientRestService({'hostname':self.hostname, 'client_port': self.port, 'server':self.reducer, 'client_model':self.client_model})
        connection_status, self.client_number = self.reducer.add_client_to_server(self.client_config["client"])
        sys.stdout = open(os.getcwd() + f'/logs/client_{self.port}.txt', 'w')
    def print_client_info(self):
        print(f"Client information :\n-> id: {self.client_id}\n-> port: {self.port}\n-> host_name: {self.hostname}\n-> run_url: {self.run_url}", flush=True)


    def run(self):
        self.rest.run()

if __name__ == "__main__":
    client = Client()
    # client.print_client_info()
    client.run()
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
from flask_autodoc.autodoc import Autodoc
from helper.pytorch.pytorch_helper import PytorchHelper
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor


class Client:
    def __init__(self, name, port, rounds, id):
        self.name = name
        self.port = port
        self.status = "Idle"
        self.connect_string = "http://{}:{}".format(self.name, self.port)
        self.last_checked = time.time()
        self.id = id
        # self.training_acc = [0] * rounds
        # self.testing_acc = [0] * rounds
        # self.training_loss = [0] * rounds
        # self.testing_loss = [0] * rounds

    def send_round_start_request(self, round_id, bucket_name, global_model, epochs):
        try:
            for i in range(3):
                retval = r.get(
                    "{}?round_id={}&bucket_name={}&global_model={}&epochs={}".format(
                        self.connect_string + '/startround',
                        round_id, bucket_name, global_model,
                        epochs))
                if retval.json()['status'] == "started":
                    self.status = "Training"
                    return True
                time.sleep(5)
            return False
        except Exception as e:
            print("Error while send_round_start_request ", e, flush=True)
            return False

    def send_round_stop_request(self):
        try:
            for i in range(3):
                retval = r.get("{}".format(self.connect_string + '/stopround'))
                if retval.json()['status'] == "stopping":
                    return True
                time.sleep(5)
            return False
        except Exception as e:
            print("Error while send_round_stop_request ", e, flush=True)
            return False

    def update_last_checked(self):
        self.last_checked = time.time()

    def get_last_checked(self):
        return time.time() - self.last_checked


def remove_pending_jobs(pending_jobs):
    """
    Method for removing the completed jobs from the list of pending jobs
    :param pending_jobs: Incomplete jobs.
    :return: List containing pending jobs.
    """
    temp = []
    for job in pending_jobs:
        if not job.done():
            temp.append(job)
    return temp


class ReducerRestService:
    def __init__(self, minio_client, config):
        self.minio_client = minio_client
        self.port = config['flask_port']
        self.clients = {}
        self.rounds = 0
        self.global_model = config["global_model"]
        self.clients_updated = 0
        self.tensorboard_path = config["tensorboard_path"]
        self.training = None
        self.stop_training_event = threading.Event()
        self.status = "Idle"
        self.unique_clients = 0
        self.training_id = config["training_id"]
        threading.Thread(target=self.remove_disconnected_clients, daemon=True).start()

    def remove_disconnected_clients(self):
        """
        Methds for removing unresponsive clients from the list of connected clients.
        :return:
        """
        while True:
            dis_clients = {client: self.clients[client] for client in self.clients if
                           self.clients[client].get_last_checked() > 50}
            self.clients = {client: self.clients[client] for client in self.clients if
                            self.clients[client].get_last_checked() < 50}
            if len(list(dis_clients.keys())) > 0:
                print("Disconnected clients are ", list(dis_clients.keys()), flush=True)
                print("Connected clients are ", list(self.clients.keys()), flush=True)
            time.sleep(10)

    def stop_training(self):
        """
        Method for stopping the training across all the clients upon admin request.
        :return:
        """
        executor = ThreadPoolExecutor(max_workers=10)
        pending_jobs = []
        for _, client in self.clients.items():
            if client.status == "Idle":
                continue
            pending_jobs.append(
                executor.submit(client.send_round_stop_request))
        while len(pending_jobs) > 0:
            print("Sending round stop requests to the clients")
            pending_jobs = remove_pending_jobs(pending_jobs)
            time.sleep(5)
        self.stop_training_event.set()
        self.training.join()
        assert self.get_clients_training() == 0
        self.rounds -= 1
        self.status = "Idle"

    def run(self):
        log = logging.getLogger('werkzeug')
        # log.setLevel(logging.ERROR)
        app = Flask(__name__)
        auto = Autodoc(app)

        @app.route('/documentation')
        @auto.doc()
        def documentation():
            """Return API documentation page"""
            return auto.html()

        @app.route('/')
        @auto.doc()
        def index():
            """Return the type of flask server"""
            ret = {
                'description': "This is the reducer"
            }
            return jsonify(ret)

        @app.route('/addclient')
        @auto.doc()
        def add_client():
            """Used the clients to get registered with the reducer"""
            name = request.args.get('name', None)
            port = request.args.get('port', None)
            if request.args.get('id', "") == self.training_id:
                self.unique_clients += 1
                available_id = "client_" + str(self.unique_clients)
                self.clients[name + ":" + port] = Client(name, port, self.rounds, available_id)
                print("Connected clients are ", list(self.clients.keys()), flush=True)
                return jsonify({
                    'status': "added",
                    'id': available_id
                })
            else:
                return jsonify({
                    'status': "Not compatible!!"
                })

        @app.route('/reconnectclient')
        @auto.doc()
        def reconnect_client():
            """Used the clients to get re-registered with the reducer"""
            name = request.args.get('name', None)
            port = request.args.get('port', None)
            client_id = request.args.get('client', None)
            if request.args.get('id', "") == self.training_id:
                self.clients[name + ":" + port] = Client(name, port, self.rounds, client_id)
                print("Connected clients are ", list(self.clients.keys()), flush=True)
                return jsonify({
                    'status': "added",
                })
            else:
                return jsonify({
                    'status': "Not compatible!!"
                })

        @app.route('/status')
        @auto.doc()
        def status_check():
            """Used by admin to check the status of reducer"""
            return jsonify({"status": self.status})

        @app.route('/training')
        @auto.doc()
        def start_training():
            """Used by the admin to start the on-device training as per the supplied parameters"""
            if self.status == "Training":
                return jsonify({"status": "Training already running!!"})
            elif self.status == "Stopping":
                return jsonify({"status": "Please wait till the training is not stopped completely!!"})
            config = {
                "rounds": int(request.args.get('rounds', '1')),
                "round_time": int(request.args.get('round_time', '200')),
                "epochs": int(request.args.get('epochs', '2'))
            }
            self.stop_training_event.clear()
            self.training = threading.Thread(target=self.train, args=(config,))
            self.training.start()
            self.status = "Training"
            ret = {
                'status': "Training started"
            }
            return jsonify(ret)

        @app.route('/stoptraining')
        @auto.doc()
        def stop_training_request():
            """Used the admin to stop the ongoing training."""
            if self.training is None or self.status == "Idle":
                return jsonify({"status": "Training not running!!"})
            if self.status == "Stopping":
                return jsonify({"status": "Stop Training request already received!!"})
            threading.Thread(target=self.stop_training, daemon=True).start()
            self.status = "Stopping"
            return jsonify({"status": "Stopping"})

        @app.route('/roundcompletedbyclient')
        @auto.doc()
        def round_completed_by_client():
            """Used by the clients to notify reducer when the round is completed by them"""
            round_id = int(request.args.get('round_id', "-1"))
            id = request.args.get("client_id", "0")
            if self.rounds == round_id and id in self.clients:
                if not os.path.exists(self.tensorboard_path + "/" + id):
                    os.mkdir(self.tensorboard_path + "/" + id)
                writer = SummaryWriter(self.tensorboard_path + "/" + self.training_id + "-" + self.clients[id].id)
                self.clients[id].status = "Idle"
                res = request.args.get("report", None)
                if res is None:
                    res = {"training_accuracy": 0, "test_accuracy": 0, "training_loss": 0, "test_loss": 0,
                           "round_time": 0}
                else:
                    res = json.loads(res)
                writer.add_scalar('training_loss', res["training_loss"], round_id)
                writer.add_scalar('test_loss', res["test_loss"], round_id)
                writer.add_scalar('training_accuracy', res["training_accuracy"], round_id)
                writer.add_scalar('test_accuracy', res["test_accuracy"], round_id)
                writer.add_scalar('round_time', res["round_time"], round_id)
                writer.close()
                print("Client - ", id, " completed round ", round_id, flush=True)
                # self.clients[id].training_acc.append(float(res["training_accuracy"]))
                # self.clients[id].testing_acc.append(float(res["test_accuracy"]))
                # self.clients[id].training_loss.append(float(res["training_loss"]))
                # self.clients[id].testing_loss.append(float(res["test_loss"]))
                return jsonify({'status': "Success"})
            return jsonify({'status': "Failure"})

        @app.route('/roundstoppedbyclient')
        @auto.doc()
        def round_stopped_by_client():
            """Used by the clients to notify the reducer when the round is stopped by them."""
            round_id = int(request.args.get('round_id', "-1"))
            id = request.args.get("client_id", "0")
            if self.rounds == round_id and id in self.clients:
                self.clients[id].status = "Idle"
                print("Client - ", id, " stopped round ", round_id, flush=True)
                return jsonify({'status': "Success"})
            return jsonify({'status': "Failure"})

        @app.route('/clientcheck')
        @auto.doc()
        def client_check():
            """Used by the clients for health-checking"""
            name = request.args.get('name', None)
            port = request.args.get('port', None)
            if name + ":" + port in self.clients.keys():
                self.clients[name + ":" + port].update_last_checked()
                ret = {
                    'status': "Available"
                }
                return jsonify(ret)
            else:
                ret = {
                    'status': "Not Available"
                }
                return jsonify(ret)

        @app.route('/creategraph')
        @auto.doc()
        def create_graph():
            """Not in use"""
            for key, client in self.clients.items():
                x = np.linspace(1, len(client.training_acc), len(client.training_acc))
                plt.plot(x, client.training_acc, "-b", label="Train_Acc")
                plt.plot(x, client.testing_acc, "-r", label="Test_Acc")
                plt.legend(loc="upper right")
                plt.xlabel("Rounds")
                plt.ylabel("Accuracy")
                plt.title("Rounds vs Accuracy for client : " + key)
                plt.savefig(key + '_Acc.png')
                plt.clf()
                plt.plot(x, client.training_loss, "-b", label="Train_loss")
                plt.plot(x, client.testing_loss, "-r", label="Test_loss")
                plt.legend(loc="upper right")
                plt.xlabel("Rounds")
                plt.ylabel("Loss")
                plt.title("Rounds vs Loss for client : " + key)
                plt.savefig(key + '_Loss.png')
                plt.clf()
            ret = {
                'status': "Created"
            }
            return jsonify(ret)

        app.run(host="0.0.0.0", port=self.port)

    def get_clients_training(self):
        """
        Method for getting the count of clients in training state.
        :return: Count of clients in training state.
        """
        client_training = 0
        for _, client in self.clients.items():
            if client.status == "Training":
                client_training += 1
        return client_training

    def train(self, config):
        """
        Method for running the training on the connected clients as per the config set by the API call and save the final model in the minio.
        :param config: Various hyperparameters for tuning the federated learning.
        :return:
        """
        for i in range(config["rounds"]):
            self.rounds += 1
            print("Round ---", self.rounds, "---", flush=True)
            bucket_name = "round" + str(self.rounds)
            if self.minio_client.bucket_exists(bucket_name):
                for obj in self.minio_client.list_objects(bucket_name):
                    self.minio_client.remove_object(bucket_name, object_name=obj.object_name)
            else:
                self.minio_client.make_bucket(bucket_name)

            self.clients_updated = len(self.clients)
            executor = ThreadPoolExecutor(max_workers=10)
            pending_jobs = []
            if self.stop_training_event.is_set():
                return
            for _, client in self.clients.items():
                pending_jobs.append(
                    executor.submit(client.send_round_start_request, self.rounds, bucket_name, self.global_model,
                                    config["epochs"]))
            while True:
                if len(remove_pending_jobs(pending_jobs)) == 0:
                    break
                time.sleep(0.1)
            print("Round start requests send to the clients successfully", flush=True)
            total_clients_started_training = self.get_clients_training()
            print("Total clients that started the training are", total_clients_started_training, flush=True)
            start_time = time.time()
            total_clients_in_training = total_clients_started_training
            while True:
                round_time = time.time() - start_time
                client_training = self.get_clients_training()
                if client_training < total_clients_in_training:
                    print("Clients in Training : " + str(client_training), flush=True)
                    total_clients_in_training = client_training
                if total_clients_in_training == 0 or int(round_time) > config["round_time"]:
                    break
                time.sleep(2)

            model = None
            helper = PytorchHelper()
            self.minio_client.fget_object('fedn-context', self.global_model, self.global_model)
            base_model = helper.load_model(self.global_model)
            os.remove(self.global_model)
            reducer_learning_rate = 1
            processed_model = 0
            for obj in self.minio_client.list_objects(bucket_name):
                if self.stop_training_event.is_set():
                    break
                self.minio_client.fget_object(bucket_name, obj.object_name, obj.object_name)
                if processed_model == 0:
                    model = helper.get_tensor_diff(helper.load_model(obj.object_name), base_model)
                else:
                    model = helper.increment_average(model, helper.get_tensor_diff(helper.load_model(obj.object_name),
                                                                                   base_model),
                                                     processed_model + 1)
                processed_model += 1
                os.remove(obj.object_name)

            if model is not None and not self.stop_training_event.is_set():
                model = helper.add_base_model(model, base_model, reducer_learning_rate)
                model_name = str(uuid.uuid4()) + ".npz"
                helper.save_model(model, model_name)
                self.minio_client.fput_object("fedn-context", model_name, model_name)
                self.global_model = model_name
                os.remove(model_name)

            if self.stop_training_event.is_set():
                return
        print("Training for {} rounds ended with global model {}".format(str(config["rounds"]), self.global_model),
              flush=True)
        self.status = "Idle"

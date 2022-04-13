import os
import sys
import time

sys.path.append(os.getcwd())
import io
import yaml
import json
import socket
import subprocess
from minio import Minio
from contextlib import closing
from multiprocessing import Process
from helper.pytorch.pytorch_helper import PytorchHelper
from model.pytorch_model_trainer import weights_to_np
from model.pytorch.pytorch_models import create_seed_model
from Reducer.reducer_rest_service import ReducerRestService


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        subprocess.call("fuser -k 7000/tcp", shell=True)
        time.sleep(2)
        return 7000
        # return str(s.getsockname()[1])


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def run_tensorboard(config):
    subprocess.call("fuser -k " + str(config["port"]) + "/tcp", shell=True)
    time.sleep(2)
    cmd = "tensorboard --host 0.0.0.0 --logdir=" + config["path"] + " --port " + str(config["port"])
    subprocess.call(cmd, shell=True)


class Reducer:
    def __init__(self):
        """ """
        with open(os.getcwd() + "/settings/settings-reducer.yaml", 'r') as file:
            try:
                fedn_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read config from settings file, exiting.', flush=True)
                raise e
        with open(os.getcwd() + "/settings/settings-common.yaml", 'r') as file:
            try:
                common_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read model_config from settings file', flush=True)
                raise e
        self.training_id = common_config["training"]["data"]["dataset"] + "_" + common_config["training"]["model"][
            "model_type"] + "_" + \
                           common_config["training"]["optimizer"]["optimizer"] + "_" + \
                           common_config["training_identifier"]["id"]
        print(self.training_id)
        if not os.path.exists(os.getcwd() + "/data/logs"):
            os.mkdir(os.getcwd() + "/data/logs")
        if not os.path.exists(os.getcwd() + "/data/logs/" + self.training_id):
            os.mkdir(os.getcwd() + "/data/logs/" + self.training_id)
        sys.stdout = open(os.getcwd() + "/data/logs/" + self.training_id + "/reducer.txt", "w")
        self.buckets = ["fedn-context"]
        self.port = find_free_port()

        if not os.path.exists(fedn_config["tensorboard"]["path"]):
            os.mkdir(fedn_config["tensorboard"]["path"])
        Process(target=run_tensorboard, args=(fedn_config["tensorboard"],), daemon=True).start()
        try:
            if not os.path.exists(os.getcwd() + '/data/reducer'):
                os.mkdir(os.getcwd() + '/data/reducer')
            self.global_model = "initial_model.npz"
            self.global_model_path = os.getcwd() + '/data/reducer/initial_model.npz'
            model, loss, optimizer, _ = create_seed_model(common_config["training"])
            helper = PytorchHelper()
            helper.save_model(weights_to_np(model.state_dict()), self.global_model_path)
        except Exception as e:
            print("Error while creating seed model : ", e)
            raise e
        print("Seed model created successfully !!")
        try:
            print("RISHI")
            storage_config = common_config["storage"]
            print("123")
            assert (storage_config["storage_type"] == "S3")
            print("Cham cham")
            minio_config = storage_config["storage_config"]
            print(1)
            self.minio_client = Minio("{0}:{1}".format(minio_config["storage_hostname"], minio_config["storage_port"]),
                                      access_key=minio_config["storage_access_key"],
                                      secret_key=minio_config["storage_secret_key"],
                                      secure=minio_config["storage_secure_mode"])
            print(2)
            for bucket in self.buckets:
                if not self.minio_client.bucket_exists(bucket):
                    self.minio_client.make_bucket(bucket)
            
            print(3)
            reducer_config_as_bytes = json.dumps(
                {"reducer": {"hostname": get_local_ip(), "port": self.port}}).encode('utf-8')
            
            print(4)
            reducer_config_as_a_stream = io.BytesIO(reducer_config_as_bytes)
            print(5)
            self.minio_client.put_object(self.buckets[0], "reducer_config.txt", reducer_config_as_a_stream,
                                         length=reducer_config_as_a_stream.getbuffer().nbytes)
            print(6)
            self.minio_client.fput_object(self.buckets[0], self.global_model, self.global_model_path)
            print(7)
            print("Address - http://", get_local_ip(), ":", self.port)
            print(8)
        except Exception as e:
            print(e)
            print("Error while setting up minio configuration")
            exit()

        config = {
            "flask_port": self.port,
            "global_model": self.global_model,
            "tensorboard_path": fedn_config["tensorboard"]["path"],
            "training_id": self.training_id
        }
        self.rest = ReducerRestService(self.minio_client, config)

    def run(self):
        # threading.Thread(target=self.control_loop, daemon=True).start()
        self.rest.run()


if __name__ == "__main__":
    try:
        reducer = Reducer()
        reducer.run()
    except Exception as e:
        print(e, flush=True)

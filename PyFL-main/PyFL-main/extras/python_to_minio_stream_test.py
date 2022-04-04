import time

import yaml
import os
from minio import Minio
import io
import json
from model.pytorch.pytorch_models import create_seed_model
from model.pytorch_model_trainer import weights_to_np

with open("../settings/settings-reducer.yaml", 'r') as file:
    try:
        fedn_config = dict(yaml.safe_load(file))
    except yaml.YAMLError as e:
        print('Failed to read config from settings file, exiting.', flush=True)
        raise e
buckets = ["fedn-context"]
try:
    storage_config = fedn_config["storage"]
    assert (storage_config["storage_type"] == "S3")
    minio_config = storage_config["storage_config"]
    minio_client = Minio("{0}:{1}".format(minio_config["storage_hostname"], minio_config["storage_port"]),
                         access_key=minio_config["storage_access_key"],
                         secret_key=minio_config["storage_secret_key"],
                         secure=minio_config["storage_secure_mode"])
    for bucket in buckets:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
    if not os.path.exists('../data/reducer'):
        os.mkdir('../data/reducer')
    global_model = "initial_model.npz"
    global_model_path = "../data/reducer/initial_model.npz"
    model, loss, optimizer = create_seed_model()
    model_dic = weights_to_np(model.state_dict())
    # print(model_dic)
    pre = (time.time())
    print(pre)
    model_json = json.dumps(model_dic)
    model_as_bytes = model_json.encode('utf-8')
    model_as_a_stream = io.BytesIO(model_as_bytes)
    # print(model_as_bytes)
    print(model_as_a_stream.getbuffer().nbytes)
    print(len(model_as_bytes))
    minio_client.put_object(buckets[0], "my_key", model_as_a_stream, length=model_as_a_stream.getbuffer().nbytes)
    print(time.time()-pre)
    # print(model_json)
    # value = "Some text I want to upload"
    # value_as_bytes = value.encode('utf-8')
    # value_as_a_stream = io.BytesIO(value_as_bytes)
    # minio_client.put_object(buckets[0], "my_key", model_as_a_stream, length=len(model_as_bytes))
    # exit()
    # helper = PytorchHelper()
    # helper.save_model(weights_to_np(model.state_dict()), global_model_path)
    # minio_client.fput_object(buckets[0], global_model, global_model_path)
except Exception as e:
    print(e)
    print("Error while setting up minio configuration")
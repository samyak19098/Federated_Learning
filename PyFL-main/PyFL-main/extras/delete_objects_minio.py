from minio import Minio
import yaml


with open("../settings/settings-common.yaml", 'r') as file:
    try:
        fedn_config = dict(yaml.safe_load(file))
    except yaml.YAMLError as e:
        print('Failed to read config from settings file, exiting.', flush=True)
        raise e
buckets = ["fedn-context"]
storage_config = fedn_config["storage"]
minio_config = storage_config["storage_config"]
minio_client = Minio("{0}:{1}".format(minio_config["storage_hostname"], minio_config["storage_port"]),
                            access_key=minio_config["storage_access_key"],
                            secret_key=minio_config["storage_secret_key"],
                            secure=minio_config["storage_secure_mode"])
if minio_client.bucket_exists(buckets[0]):
    for obj in minio_client.list_objects(buckets[0]):
        minio_client.remove_object(buckets[0], object_name=obj.object_name)
for i in range(400):
    rounds = i+1
    bucket_name = "round" + str(rounds)
    if minio_client.bucket_exists(bucket_name):
        for obj in minio_client.list_objects(bucket_name):
            minio_client.remove_object(bucket_name, object_name=obj.object_name)
    else:
        break
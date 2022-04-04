import os
import time
import socket

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

print("Working Directory :",os.getcwd())
print(get_local_ip())
print("CUDA devices ", os.environ["CUDA_VISIBLE_DEVICES"])
time.sleep(5)
from torch.multiprocessing import Pipe, get_context, Event, Manager
# from stream import client, server
import argparse
import json
import threading
import os
from socket import *
import socket
from utils_lib.utils import *
from models import *

def build_args():
    cfg_path = r'cfg/cfg.json'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--run_config_path', default=r'cfg/cfg.json')
    with open(cfg_path) as f:
        args = json.load(f)
    return args


class App:
    def __init__(self, args):
        self.args = args
        ctx = get_context('spawn')
        self.manager = Manager()
        self.smart_value = self.manager.Value('i', 0)
        self.connection = Pipe()
        self.queue = self.manager.Queue()
        # self.stream_process = ctx.Process(target=client, args=(self.args, None))
        # self.detection_process = ctx.Process(target=server, args=(None, None))
        self.input_thread = threading.Thread(name='input_thread', target=self.collect_rowdata)
        self.detection_thread = threading.Thread(name='input_thread', target=self.detection_loop)

    def run(self):
        # self.stream_process.start()
        self.detection_thread.start()
        self.input_thread.start()

    def collect_rowdata(self):
        print("start client...")
        print("args", self.args)
        host = "127.0.0.1"  # set to IP address of target computer
        port = 13000
        addr = (host, port)
        UDPSock = socket.socket(AF_INET, SOCK_DGRAM)
        data = str.encode("client connected")
        while True:
            data = str.encode(input("\nEnter message to send or type 'exit': "))
            if data.decode("utf-8") == "exit":
                break
            UDPSock.sendto(data, addr)
        UDPSock.close()
        os._exit(0)

    def detection_loop(self):
        print("start server...")
        host = ""
        port = 13000
        buf = 1024
        addr = (host, port)
        UDPSock = socket.socket(AF_INET, SOCK_DGRAM)
        UDPSock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        UDPSock.bind(addr)
        print("\nWaiting to receive messages...")

        while True:
            (data, addr) = UDPSock.recvfrom(buf)
            print("\nReceived message: " + data.decode("utf-8"))
            if data == "exit":
                break

        UDPSock.close()
        os._exit(0)

    def __pred(self, model_output):
        return torch.argmax(model_output, dim=1)

    def __init_model(self):
        args = self.args
        model = DetectorMultiLSTM(input_size=args["Model"]["input_size"], hidden_size=args["Model"]["hidden_size"],
                                  target_size=args['Model']['num_classes'])
        model.eval()
        return model

if __name__ == '__main__':
    args = build_args()
    application = App(args)
    application.run()
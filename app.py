from torch.multiprocessing import Pipe, get_context, Event, Manager
from stream import client, server
import argparse
import json

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
        self.stream_process = ctx.Process(target=client, args=(self.args, None))

    def run(self):
        self.stream_process.run()


if __name__ == '__main__':
    args = build_args()
    application = App(args)
    application.run()
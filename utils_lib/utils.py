import json


def build_args():
    cfg_path = r'cfg/cfg.json'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--run_config_path', default=r'cfg/cfg.json')
    with open(cfg_path) as f:
        args = json.load(f)
    return args
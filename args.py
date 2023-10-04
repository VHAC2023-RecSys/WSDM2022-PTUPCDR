from typing import List, Optional
from tap import Tap
from typing import Dict
import torch
import numpy as np
import random
import json


class Args(Tap):
    process_data_mid: Optional[bool] = False
    process_data_ready: Optional[bool] = False
    task: Optional[str] = "1"
    base_model: Optional[str] = "MF"
    seed: Optional[int] = 2020
    ratio: Optional[List[float]] = [0.8, 0.2]
    gpu: Optional[str] = "0"
    epoch: Optional[int] = 10
    lr: Optional[float] = 0.01


def prepare(config_path):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--process_data_mid", default=0)
    # parser.add_argument("--process_data_ready", default=0)
    # parser.add_argument("--task", default="1")
    # parser.add_argument("--base_model", default="MF")
    # parser.add_argument("--seed", type=int, default=2020)
    # parser.add_argument("--ratio", default=[0.8, 0.2])
    # parser.add_argument("--gpu", default="0")
    # parser.add_argument("--epoch", type=int, default=10)
    # parser.add_argument("--lr", type=float, default=0.01)
    # args = parser.parse_args()

    args = Args().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(config_path, "r") as f:
        config: Dict = json.load(f)
        config["base_model"] = args.base_model
        config["task"] = args.task
        config["ratio"] = args.ratio
        config["epoch"] = args.epoch
        config["lr"] = args.lr
    return args, config

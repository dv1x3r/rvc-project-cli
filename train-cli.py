import os
import sys
import numpy as np
from subprocess import Popen
from argparse import ArgumentParser
from dotenv import load_dotenv
from configs.config import Config

parser = ArgumentParser(description="RVC Model Train CLI")

parser.add_argument("--name", required=True, help="Experiment name")
parser.add_argument("--dataset", required=True, help="Path to the dataset folder")
parser.add_argument("--sample_rate", default=40000, help="Optional: Target sample rate. (default: %(default)s)")

args = parser.parse_args()
sys.argv = sys.argv[:1]
print(args)

load_dotenv()
config = Config()

# Step 1: Fill in the experimental configuration.
# Experimental data is stored in the ‘logs’ folder, with each experiment having a separate folder.
# Manually enter the experiment name path, which contains the experimental configuration, logs, and trained model files.

now_dir = os.getcwd()
sys.path.append(now_dir)

np = int(np.ceil((os.cpu_count() or 4) / 1.5))
per = 3.0 if config.is_half else 3.7

os.makedirs("%s/logs/%s" % (now_dir, args.name), exist_ok=True)
f = open("%s/logs/%s/preprocess.log" % (now_dir, args.name), "w")
f.close()

cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
    config.python_cmd,
    args.dataset,
    args.sample_rate,
    np,
    now_dir,
    args.name,
    config.noparallel,
    per,
)
Popen(cmd, shell=True).wait()


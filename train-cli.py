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
parser.add_argument(
    "--sample_rate",
    default=40000,
    help="Optional: Target sample rate. (default: %(default)s)",
)
parser.add_argument(
    "--method",
    required=True,
    help="Select the pitch extraction algorithm.",
    choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
)
parser.add_argument(
    "--gpus_rmvpe",
    default="-",
    help="Optional: Enter the GPU index(es) separated by '-', e.g., 0-0-1 to use 2 processes in GPU0 and 1 process in GPU1",
)

args = parser.parse_args()
sys.argv = sys.argv[:1]
print(args)

# load_dotenv()
config = Config()

# Step 1: Fill in the experimental configuration.
# Experimental data is stored in the ‘logs’ folder, with each experiment having a separate folder.
# Manually enter the experiment name path, which contains the experimental configuration, logs, and trained model files.

cwd = os.getcwd()
n_cpu = int(np.ceil((os.cpu_count() or 4) / 1.5))
per = 3.0 if config.is_half else 3.7
os.makedirs("%s/logs/%s" % (cwd, args.name), exist_ok=True)

# Step 2a: Automatically traverse all files in the training folder that can be decoded into audio and perform slice normalization. Generates 2 wav folders in the experiment directory. Currently, only single-singer/speaker training is supported.
Popen(
    [
        config.python_cmd,
        "infer/modules/train/preprocess.py",
        args.dataset,
        str(args.sample_rate),
        str(n_cpu),
        f"{cwd}/logs/{args.name}",
        str(config.noparallel),
        "%.1f" % per,
    ]
).wait()

# Step 2b: Use CPU to extract pitch (if the model has pitch), use GPU to extract features (select GPU index).

if args.method != "rmvpe_gpu":
    Popen(
        [
            config.python_cmd,
            "infer/modules/train/extract/extract_f0_print.py",
            f"{cwd}/logs/{args.name}",
            str(n_cpu),
            args.method,
        ],
        cwd=cwd,
    ).wait()
else:
    if args.gpus_rmvpe != "-":
        gpus_rmvpe = args.gpus_rmvpe.split("-")
        leng = len(gpus_rmvpe)
        ps = []
        for idx, n_g in enumerate(gpus_rmvpe):
            p = Popen(
                [
                    config.python_cmd,
                    "infer/modules/train/extract/extract_f0_rmvpe.py",
                    str(leng),
                    str(idx),
                    str(n_cpu),
                    f"{cwd}/logs/{args.name}",
                    str(config.is_half),
                ],
                cwd=cwd,
            )
            ps.append(p)
        for p in ps:
            p.wait()
    else:
        Popen(
            [
                config.python_cmd,
                "infer/modules/train/extract/extract_f0_rmvpe_dml.py",
                f"{cwd}/logs/{args.name}",
            ],
            cwd=cwd,
        ).wait()


# Step 3: Fill in the training settings and start training the model and index.

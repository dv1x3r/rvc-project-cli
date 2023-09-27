import os
import sys
import faiss
import numpy as np
from subprocess import Popen
from argparse import ArgumentParser
from dotenv import load_dotenv
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans

parser = ArgumentParser(description="RVC Model Train CLI")

parser.add_argument("--name", required=True, help="Experiment name")
parser.add_argument("--dataset", required=True, help="Path to the dataset folder")
parser.add_argument("--version", default="v2", help="Optional: (default: %(default)s)")
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
    "--gpu_rmvpe",
    default="-",
    help="Optional: Enter the GPU index(es) separated by '-', e.g., 0-0-1 to use 2 processes in GPU0 and 1 process in GPU1",
)
parser.add_argument(
    "--gpu",
    default="",
    help="Optional: Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, and 2",
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

# pitch guidance

if args.method == "rmvpe_gpu":
    if args.gpu_rmvpe != "-":
        ps = []
        gpus_rmvpe = args.gpu_rmvpe.split("-")
        for idx, n_g in enumerate(gpus_rmvpe):
            p = Popen(
                [
                    config.python_cmd,
                    "infer/modules/train/extract/extract_f0_rmvpe.py",
                    str(len(gpus_rmvpe)),
                    str(idx),
                    str(n_g),
                    f"{cwd}/logs/{args.name}",
                    str(config.is_half),
                ]
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
            ]
        ).wait()
else:
    Popen(
        [
            config.python_cmd,
            "infer/modules/train/extract/extract_f0_print.py",
            f"{cwd}/logs/{args.name}",
            str(n_cpu),
            args.method,
        ]
    ).wait()

# feature extraction

ps = []
gpus = args.gpu.split("-")
for idx, n_g in enumerate(gpus):
    p = Popen(
        [
            config.python_cmd,
            "infer/modules/train/extract_feature_print.py",
            config.device,
            str(len(gpus)),
            str(idx),
            f"{cwd}/logs/{args.name}",
            args.version,
        ]
    )
    ps.append(p)
for p in ps:
    p.wait()

# Step 3: Fill in the training settings and start training the model and index.

# train feature index

exp_dir = f"logs/{args.name}"
feature_dir = (
    "%s/3_feature256" % (exp_dir)
    if args.version == "v1"
    else "%s/3_feature768" % (exp_dir)
)

listdir_res = list(os.listdir(feature_dir))
npys = []

for name in sorted(listdir_res):
    phone = np.load("%s/%s" % (feature_dir, name))
    npys.append(phone)

big_npy = np.concatenate(npys, 0)
big_npy_idx = np.arange(big_npy.shape[0])
np.random.shuffle(big_npy_idx)
big_npy = big_npy[big_npy_idx]

if big_npy.shape[0] > 2e5:
    big_npy = (
        MiniBatchKMeans(
            n_clusters=10000,
            verbose=True,
            batch_size=256 * config.n_cpu,
            compute_labels=False,
            init="random",
        )
        .fit(big_npy)
        .cluster_centers_
    )

np.save("%s/total_fea.npy" % exp_dir, big_npy)
n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)

index = faiss.index_factory(256 if args.version == "v1" else 768, "IVF%s,Flat" % n_ivf)
index_ivf = faiss.extract_index_ivf(index)
index_ivf.nprobe = 1
index.train(big_npy)
faiss.write_index(
    index,
    "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
    % (exp_dir, n_ivf, index_ivf.nprobe, args.name, args.version),
)

batch_size_add = 8192
for i in range(0, big_npy.shape[0], batch_size_add):
    index.add(big_npy[i : i + batch_size_add])
faiss.write_index(
    index,
    "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
    % (exp_dir, n_ivf, index_ivf.nprobe, args.name, args.version),
)

# train model

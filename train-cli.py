import os
import sys
import json
import pathlib
import faiss
import numpy as np
from random import shuffle
from subprocess import Popen
from argparse import ArgumentParser
from dotenv import load_dotenv
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans

parser = ArgumentParser(description="RVC Model Train CLI")

parser.add_argument("--name", required=True, help="Experiment name")
parser.add_argument("--dataset", required=True, help="Path to the dataset folder")
parser.add_argument("--version", default="v2", help="Optional: (default: %(default)s)")
parser.add_argument("--sample_rate", default="40k", help="Optional: Target sample rate. (default: %(default)s)")
parser.add_argument("--method", default="rmvpe_gpu", help="Optional: Select the pitch extraction algorithm. (default: %(default)s)", choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"])
parser.add_argument("--gpu_rmvpe", default="0-0", help="Optional: Enter the GPU index(es) separated by '-', e.g., 0-0-1 to use 2 processes in GPU0 and 1 process in GPU1. (default: %(default)s)")
parser.add_argument("--gpu", default="0", help="Optional: Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, and 2. (default: %(default)s)")
parser.add_argument("--batch_size", default=8, help="Optional: Batch size per GPU. (default: %(default)s)")
parser.add_argument("--total_epoch", default=200, help="Optional: Total training epochs. (default: %(default)s)")
parser.add_argument("--save_epoch", default=20, help="Optional: Save frequency. (default: %(default)s)")
parser.add_argument("--save_latest", default=1, help="Optional: Save only the latest '.ckpt' file. (default: %(default)s)")
parser.add_argument("--cache_gpu", default=0, help="Optional: Cache all training sets to GPU. (default: %(default)s)")
parser.add_argument("--save_every_weights", default=0, help="Optional: Save a small final model to the 'weights' folder at each save point. (default: %(default)s)")

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
sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}


# Step 2a: Automatically traverse all files in the training folder that can be decoded into audio and perform slice normalization. Generates 2 wav folders in the experiment directory. Currently, only single-singer/speaker training is supported.
Popen([
    config.python_cmd,
    "infer/modules/train/preprocess.py",
    args.dataset,
    str(sr_dict[args.sample_rate]),
    str(n_cpu),
    f"{cwd}/logs/{args.name}",
    str(config.noparallel),
    "%.1f" % per,
]).wait()

# Step 2b: Use CPU to extract pitch (if the model has pitch), use GPU to extract features (select GPU index).

# pitch guidance

if args.method == "rmvpe_gpu":
    if args.gpu_rmvpe != "-":
        ps = []
        gpus_rmvpe = args.gpu_rmvpe.split("-")
        for idx, n_g in enumerate(gpus_rmvpe):
            p = Popen([
                config.python_cmd,
                "infer/modules/train/extract/extract_f0_rmvpe.py",
                str(len(gpus_rmvpe)),
                str(idx),
                str(n_g),
                f"{cwd}/logs/{args.name}",
                str(config.is_half),
            ])
            ps.append(p)
        for p in ps:
            p.wait()
    else:
        Popen([
            config.python_cmd,
            "infer/modules/train/extract/extract_f0_rmvpe_dml.py",
            f"{cwd}/logs/{args.name}",
        ]).wait()
else:
    Popen([
        config.python_cmd,
        "infer/modules/train/extract/extract_f0_print.py",
        f"{cwd}/logs/{args.name}",
        str(n_cpu),
        args.method,
    ]).wait()

# feature extraction

ps = []
gpus = args.gpu.split("-")
for idx, n_g in enumerate(gpus):
    p = Popen([
        config.python_cmd,
        "infer/modules/train/extract_feature_print.py",
        config.device,
        str(len(gpus)),
        str(idx),
        f"{cwd}/logs/{args.name}",
        args.version,
    ])
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

gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
feature_dir = (
    "%s/3_feature256" % (exp_dir)
    if args.version == "v1"
    else "%s/3_feature768" % (exp_dir)
)

f0_dir = "%s/2a_f0" % (exp_dir)
f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
names = (
    set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
    & set([name.split(".")[0] for name in os.listdir(feature_dir)])
    & set([name.split(".")[0] for name in os.listdir(f0_dir)])
    & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
)

opt = []
for name in names:
    opt.append(
        "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
        % (
            gt_wavs_dir.replace("\\", "\\\\"),
            name,
            feature_dir.replace("\\", "\\\\"),
            name,
            f0_dir.replace("\\", "\\\\"),
            name,
            f0nsf_dir.replace("\\", "\\\\"),
            name,
            0,
        )
    )

fea_dim = 256 if args.version == "v1" else 768
for _ in range(2):
    opt.append(
        "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
        % (cwd, args.sample_rate, cwd, fea_dim, cwd, cwd, 0)
    )

shuffle(opt)
with open("%s/filelist.txt" % exp_dir, "w") as f:
    f.write("\n".join(opt))

if args.version == "v1" or args.sample_rate == "40k":
    config_path = "v1/%s.json" % args.sample_rate
else:
    config_path = "v2/%s.json" % args.sample_rate

config_save_path = os.path.join(exp_dir, "config.json")
if not pathlib.Path(config_save_path).exists():
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(
            config.json_config[config_path],
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
        )
        f.write("\n")

if args.gpu:
    Popen([
        config.python_cmd,
        "infer/modules/train/train.py",
        "-e", args.name,
        "-sr", args.sample_rate,
        "-f0", "1",
        "-bs", str(args.batch_size),
        "-g", args.gpu,
        "-te", str(args.total_epoch),
        "-se", str(args.save_epoch),
        "-pg", "assets/pretrained_v2/f0G40k.pth",
        "-pd", "assets/pretrained_v2/f0D40k.pth",
        "-l", str(args.save_latest),
        "-c", str(args.cache_gpu),
        "-sw", str(args.save_every_weights),
        "-v", args.version,
    ]).wait()
else:
    Popen([
        config.python_cmd,
        "infer/modules/train/train.py",
        "-e", args.name,
        "-sr", args.sample_rate,
        "-f0", "1",
        "-bs", str(args.batch_size),
        # "-g", args.gpu,
        "-te", str(args.total_epoch),
        "-se", str(args.save_epoch),
        "-pg", "assets/pretrained_v2/f0G40k.pth",
        "-pd", "assets/pretrained_v2/f0D40k.pth",
        "-l", str(args.save_latest),
        "-c", str(args.cache_gpu),
        "-sw", str(args.save_every_weights),
        "-v", args.version,
    ]).wait()

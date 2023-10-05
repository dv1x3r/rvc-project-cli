import os
import sys
from argparse import ArgumentParser
from dotenv import load_dotenv
from configs.config import Config
from infer.modules.vc.modules import VC
from scipy.io import wavfile

parser = ArgumentParser(description="RVC Model Inference CLI")

parser.add_argument("-i", "--input", required=True, help="Path of the audio file to be processed")
parser.add_argument("-o", "--output", required=True, help="Path of the destination wav file")
parser.add_argument("--model", required=True, help="Inferencing voice model name")
parser.add_argument("--index", required=True, help="Inferencing voice index path")
parser.add_argument("--method", required=True, help="The pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement")
parser.add_argument("--ratio", default=0.75, help="Optional: Search feature ratio (controls accent strength, too high has artifacting). (default: %(default)s)")
parser.add_argument("--transpose", default=0, help="Optional: Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12). (default: %(default)s)")
parser.add_argument("--f_curve", help="Optional: F0 curve file, replaces the default F0 and pitch modulation. (default: %(default)s)")
parser.add_argument("--filter", default=3, help="Optional. If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness. (default: %(default)s)")
parser.add_argument("--resample", default=0, help="Optional. Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling. (default: %(default)s)")
parser.add_argument("--rms", default=0.25, help="Optional. Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume. (default: %(default)s)")
parser.add_argument("--protect", default=0.33, help="Optional. Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy. (default: %(default)s)")

args = parser.parse_args()
sys.argv = sys.argv[:1]
print(args)

os.chdir(os.path.abspath(os.path.dirname(__file__)))
print(os.getcwd())

load_dotenv()
vc = VC(Config())

vc.get_vc(args.model)
_, wav_opt = vc.vc_single(0, args.input, args.transpose, args.f_curve, args.method, "", args.index, args.ratio, args.filter, args.resample, args.rms, args.protect)

wavfile.write(args.output, wav_opt[0], wav_opt[1])


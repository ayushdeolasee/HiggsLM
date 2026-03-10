from datasets import load_dataset
import argparse
import os

parser = argparse.ArgumentParser(description="Download the GSM8K dataset")
parser.add_argument("--dir_path", type=str, help="Directory path to save the dataset", default="./gsm8k")

args = parser.parse_args()

os.path.exists(args.dir_path) or os.makedirs(args.dir_path)


ds = load_dataset("openai/gsm8k", "main", cache_dir=args.dir_path)

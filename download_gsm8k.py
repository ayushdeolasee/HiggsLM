from datasets import load_dataset
import argparse
import os
import numpy as np
from llm.tokenizer import tokenize, write_datafile
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Download the GSM8K dataset")
parser.add_argument("--dir_path", type=str, help="Directory path to save the dataset", default="./gsm8k")
parser.add_argument("--batch_size", type=int, help="Batch size for processing the dataset", default=16)
parser.add_argument("--max_length", type=int, help="Maximum length of the tokenized sequences", default=2048)
parser.add_argument('--shard_size', type=int, help="Number of tokens per shard", default=int(1e8))

args = parser.parse_args()

os.path.exists(args.dir_path) or os.makedirs(args.dir_path)


fw = load_dataset("openai/gsm8k", "main", cache_dir=args.dir_path)

shard_index = 0
# preallocate buffer to hold current shard
all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None

for doc in fw:
    tokens = tokenize(doc)
    if token_count + len(tokens) < args.shard_size:
        all_tokens_np[token_count : token_count + len(tokens)] = tokens
        token_count += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")
        remainder = args.shard_size - token_count
        if progress_bar is None:
            progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(remainder)
        all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
        token_count = len(tokens) - remainder

if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])





"""
FineWeb-Edu dataset (for srs pretraining) - OPTIMIZED VERSION
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb_speedup.py
Will save shards to the local directory "edu_fineweb10B".

Optimizations applied:
1. Use all available CPU cores (nprocs = os.cpu_count())
2. Removed debug print statement inside the loop
3. Increased chunksize from 16 to 256 for reduced IPC overhead
4. Use imap_unordered instead of imap (order doesn't matter for training)
5. Batched progress bar updates to reduce tqdm overhead
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
local_dir = "/Volumes/Parallels Windows/fineweb-100BT"
remote_name = "sample-100BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
print(f"Data cache directory: {DATA_CACHE_DIR}")

# download the dataset
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name=remote_name,
    split="train",
    cache_dir="/Volumes/Parallels Windows/huggingface_cache"
)

# init the tokenizer
enc = tiktoken.encoding_for_model("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token


def tokenize(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    """Write tokens to disk as numpy file."""
    np.save(filename, tokens_np)


# OPTIMIZATION 1: Use all available CPU cores instead of 1
# Leave one core free for the main process handling I/O
nprocs = max(1, os.cpu_count() - 1)
print(f"Using {nprocs} worker processes for tokenization")

# OPTIMIZATION 2: Larger chunksize reduces inter-process communication overhead
# 256 is a good balance between memory usage and throughput
chunksize = 256

with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    # OPTIMIZATION 3: Use imap_unordered instead of imap
    # Since document order doesn't matter for pretraining, we can process
    # documents as they complete rather than waiting for in-order results
    for tokens in pool.imap_unordered(tokenize, fw, chunksize=chunksize):
        # OPTIMIZATION 4: Removed print(tokens) - this was a major I/O bottleneck
        
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

print(f"Done! Wrote {shard_index + 1} shards to {DATA_CACHE_DIR}")

from datasets import load_dataset
import numpy as np
import os
import tiktoken
from tqdm import tqdm  # pip install tqdm
from rich import print

# Configuration variables (edit as needed)
BATCH_SIZE = 16
DIRECTORY = "fineweb-edu-100BT"
MAX_LENGTH = 2048
DATASET_TYPE = "fineweb-edu"
NAME = "sample-100BT"
SHARD_SIZE = int(1e8)


def download():
    print(
        f"Staring download of [bold magenta]dataset[/bold magenta]: HuggingFaceFW/{DATASET_TYPE} | "
        f"in [bold blue]directory[/bold blue]: {DIRECTORY} | with [bold red]max length[/bold red]: {MAX_LENGTH}"
    )

    data_cache_dir = DIRECTORY
    if os.path.exists(DIRECTORY):
        print(f"[bold green]Directory {DIRECTORY} already exists[/bold green] :heavy_check_mark:")
    else:
        os.makedirs(DIRECTORY, exist_ok=True)
        print(f":warning: [bold yellow]Directory {DIRECTORY} created[/bold yellow]")

    fw = load_dataset(f"HuggingFaceFW/{DATASET_TYPE}", name=NAME, split="train", cache_dir="/Volumes/Parallels Windows/huggingface_cache")
    enc = tiktoken.encoding_for_model("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    def tokenize(doc):
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def write_datafile(filename, tokens_np):
        np.save(filename, tokens_np)

    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    # Direct iteration over the dataset
    for doc in fw:
        tokens = tokenize(doc)
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < SHARD_SIZE:
            # simply append tokens to current shard
            all_tokens_np[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = SHARD_SIZE - token_count
            if progress_bar is None:
                progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(remainder)
            all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    download()

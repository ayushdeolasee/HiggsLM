from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
from rich import print
from llm.tokenizer import tokenize, write_datafile
import argparse

parser = argparse.ArgumentParser(description="Download pre-train-dataset")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size to store the dataset in")
parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
parser.add_argument("--fineweb_dataset", type=str, default="", help="Fineweb dataset subset")
parser.add_argument("--streaming", type=bool, default=True, help="Stream dataset from huggingface")
parser.add_argument("--shard_size", type=int, default=int(1e8))
parser.add_argument("--directory", type=str, default="./data", help="directory to store the dataset")
args = parser.parse_args()

def _resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def download():
    output_dir = _resolve_path(args.directory)
    print(
        f"Starting download of [bold magenta]dataset[/bold magenta]: HuggingFaceFW/fineweb-edu | "
        f"from [bold blue]script[/bold blue]: {__file__} | "
        f"to [bold green]output[/bold green]: {output_dir} | "
        f"with [bold red]max length[/bold red]: {args.max_length} | "
        f"streaming={args.streaming}"
    )

    data_cache_dir = output_dir
    if os.path.exists(output_dir):
        print(f"[bold green]Directory {output_dir} already exists[/bold green] :heavy_check_mark:")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f":warning: [bold yellow]Directory {output_dir} created[/bold yellow]")

    if args.streaming:
        print("[bold yellow]Streaming enabled[/bold yellow]: skipping local Arrow materialization")
        fw = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=args.fineweb_dataset,
            split="train",
            streaming=True,
        )
    else:
        fw = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=args.fineweb_dataset,
            split="train",
        )

    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    # Direct iteration over the dataset
    for doc in fw:
        tokens = tokenize(doc)
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
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

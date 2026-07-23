from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
from rich import print
from llm.tokenizer import tokenize, tokenize_prompt, write_datafile
import argparse
import json

parser = argparse.ArgumentParser(description="Download pre-train-dataset")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size to store the dataset in")
parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
parser.add_argument("--fineweb_dataset", type=str, default="", help="Fineweb dataset subset")
parser.add_argument("--streaming", type=bool, default=True, help="Stream dataset from huggingface")
parser.add_argument("--shard_size", type=int, default=int(1e8))
parser.add_argument("--directory", type=str, default="./data", help="directory to store the dataset")
parser.add_argument("--download-eval-ds", type=bool, default=False, help="Download evaluation dataset (Arc-Easy) for evaluation of LLM.")
parser.add_argument("--download-pre-train-ds", type=bool, default=False, help="Download pre-train dataset (Arc-Easy) for pre-training of LLM.")

args = parser.parse_args()

def _resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))

def create_shards(data_dir, dataset, dataset_name):
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    # Direct iteration over the dataset
    for doc in dataset:
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
            filename = os.path.join(data_dir, f"{dataset_name}_{split}_{shard_index:06d}")
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
        filename = os.path.join(data_dir, f"{dataset_name}_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])



def download_pre_train():
    pre_train_dir = _resolve_path(f"{args.directory}/pre-train")
    print(
        f"Starting download of [bold magenta]dataset[/bold magenta]: HuggingFaceFW/fineweb-edu | "
        f"from [bold blue]script[/bold blue]: {__file__} | "
        f"to [bold green]output[/bold green]: {pre_train_dir} | "
        f"with [bold red]max length[/bold red]: {args.max_length} | "
        f"streaming={args.streaming}"
    )

    if os.path.exists(pre_train_dir):
        print(f"[bold green]Directory {pre_train_dir} already exists[/bold green] :heavy_check_mark:")
    else:
        os.makedirs(pre_train_dir, exist_ok=True)
        print(f":warning: [bold yellow]Directory {pre_train_dir} created[/bold yellow]")

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

    create_shards(pre_train_dir, fw, "fineweb_edu")

"""
[{
    encode(Question)
    encode(Answer)
}, ...]
"""

def download_eval_ds():
    eval_data_dir = _resolve_path(f"{args.directory}/eval") 

    print(
        f"Starting download of [bold magenta]dataset[/bold magenta]: allenai/ai2_arc | "
        f"from [bold blue]script[/bold blue]: {__file__} | "
        f"to [bold green]output[/bold green]: {eval_data_dir} | "
        f"with [bold red]max length[/bold red]: {args.max_length} | "
        f"streaming={args.streaming}"
    )

    if os.path.exists(eval_data_dir):
        print(f"[bold green]Directory {eval_data_dir} already exists[/bold green] :heavy_check_mark:")
    else:
        os.makedirs(eval_data_dir, exist_ok=True)
        print(f":warning: [bold yellow]Directory {eval_data_dir} created[/bold yellow]")

    
    if args.streaming:
        print("[bold yellow]Streaming enabled[/bold yellow]: skipping local Arrow materialization")
        fw = load_dataset(
            "allenai/ai2_arc",
            name="ARC-Easy",
            split="train",
            streaming=True,
        )
    else:
        fw = load_dataset(
            "allenai/ai2_arc",
            name="ARC-Easy",
            split="train",
        )

    for i in range(1024):
        write_data = []
        for content in enumerate(fw):
            question = tokenize_prompt(str(content[1]["question"]))  # np.array
            match content[1]["answerKey"]:
                case "A":
                    answer = 0
                case "B":
                    answer = 1
                case "C":
                    answer = 2
                case "D":
                    answer = 3
            options = []
            for option in content[1]["choices"]["text"]:
                options.append(tokenize_prompt(str(option)).tolist())
            write_data.append({"question": question.tolist(), "answer": answer, "options": options})
        with open(f"{eval_data_dir}/{i+1}.json", "w") as f:
            f.write(json.dumps(write_data))

if __name__ == "__main__":
    if os.path.exists(_resolve_path(args.directory)):
        print("Data directory exists")
    else:
        print("Data directory doesn't exist, creating one.")
        os.makedirs(_resolve_path(args.directory), exist_ok=True)

    if args.download_pre_train_ds == True: 
        download_pre_train()
    if args.download_eval_ds == True:
        download_eval_ds()
    if args.download_pre_train_ds == False & args.download_eval_ds == False:
        print("Downloading pre-train dataset and eval dataset")
        download_pre_train()
        download_eval_ds()

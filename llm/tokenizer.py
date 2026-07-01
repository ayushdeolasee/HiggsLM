import tiktoken
import numpy as np
import torch

enc = tiktoken.encoding_for_model("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        return tokens_np.astype(np.long)

def tokenize_prompt(prompt):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(prompt))
    return np.array(tokens).astype(np.long)

def str_to_pre_train_tokens(prompt):
    tokens = tokenize_prompt(prompt) 
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0) 

def decode_tokens(tokens):
    return enc.decode(tokens)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


import tiktoken
import numpy as np

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


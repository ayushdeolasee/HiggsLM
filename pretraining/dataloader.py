import numpy as np 
import torch 

class DataLoaderLite:
    def __init__(self, B, T, split, data_root):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        data_root = data_root
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        if (len(shards) > 0) == False:
            log.error(f"no shards found for split {split}")
        # assert len(shards) > 0, f"no shards found for split {split}"
        print(f"[green]found {len(shards)} shards for split {split}[/green]")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

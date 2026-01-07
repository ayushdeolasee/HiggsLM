from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from pathlib import Path
import shutil
import time

DATASET_PATH = "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"
SHARD_SIZE = int(1e6)  # 1M tokens per shard
OUTPUT_DIR = "./fineweb10B-edu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CPUS = mp.cpu_count()
OPTIMAL_TASKS = max(1, NUM_CPUS - 1)

# Initialize tokenizer once globally
try:
    ENCODER = tiktoken.encoding_for_model("gpt2")
    EOT_TOKEN = ENCODER._special_tokens['<|endoftext|>']
    print(f"Tokenizer initialized successfully")
except Exception as e:
    print(f"Error initializing tokenizer: {e}")
    raise

def check_disk_space(path, required_gb=10):
    """Check if there's enough disk space"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        print(f"Disk space: {free_gb:.1f} GB free on {path}")
        if free_gb < required_gb:
            print(f"Warning: Only {free_gb:.1f} GB free, need at least {required_gb} GB")
            return False
        return True
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
        return True  # Assume OK if we can't check

def tokenize_and_attach(doc):
    """Tokenize document and attach tokens as attribute"""
    try:
        if not hasattr(doc, 'text') or doc.text is None or len(doc.text.strip()) == 0:
            return False
            
        # Tokenize with EOT token at start
        tokens = [EOT_TOKEN]
        encoded_tokens = ENCODER.encode_ordinary(doc.text)
        tokens.extend(encoded_tokens)
        
        if len(tokens) == 1:  # Only EOT token
            return False
            
        # Store tokens as numpy array
        doc.tokens = np.array(tokens, dtype=np.uint16)
        return True
        
    except Exception as e:
        print(f"Error tokenizing document: {e}")
        return False

class EfficientNumpyWriter:
    """Efficient numpy writer that batches writes and minimizes I/O"""
    
    def __init__(self, output_dir, shard_size, buffer_size=100000, rank=0):
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.buffer_size = buffer_size
        self.rank = rank
        
        # Initialize state
        self.token_buffer = []
        self.total_tokens = 0
        self.shard_index = 0
        self.doc_count = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check disk space
        if not check_disk_space(str(self.output_dir)):
            raise RuntimeError("Insufficient disk space")
        
        print(f"Initialized writer (rank {rank}): shard_size={shard_size}, buffer_size={buffer_size}")
    
    def write(self, doc):
        """Write a document's tokens to the buffer"""
        if not hasattr(doc, 'tokens') or doc.tokens is None:
            return
            
        tokens = doc.tokens.tolist()  # Convert to list for easier handling
        self.token_buffer.extend(tokens)
        self.doc_count += 1
        
        # Flush when buffer is full
        if len(self.token_buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush tokens to shard files"""
        if not self.token_buffer:
            return
            
        while len(self.token_buffer) >= self.shard_size:
            # Create a full shard
            shard_tokens = self.token_buffer[:self.shard_size]
            self._write_shard(shard_tokens)
            self.token_buffer = self.token_buffer[self.shard_size:]
            self.total_tokens += self.shard_size
            self.shard_index += 1
            
        # If we have a significant amount left, create a partial shard
        if len(self.token_buffer) > self.shard_size // 2:
            shard_tokens = self.token_buffer
            self._write_shard(shard_tokens)
            self.total_tokens += len(shard_tokens)
            self.token_buffer = []
            self.shard_index += 1
    
    def _write_shard(self, tokens):
        """Write a shard of tokens to a numpy file"""
        try:
            split = "val" if self.shard_index == 0 else "train"
            # Include rank in filename to avoid conflicts
            filename = self.output_dir / f"edufineweb_{split}_{self.rank:02d}_{self.shard_index:06d}.npy"
            
            # Convert to numpy array and save
            token_array = np.array(tokens, dtype=np.uint16)
            np.save(filename, token_array)
            
            print(f"Rank {self.rank}: Wrote shard {self.shard_index}: {len(tokens)} tokens to {filename}")
            
        except Exception as e:
            print(f"Rank {self.rank}: Error writing shard {self.shard_index}: {e}")
            raise
    
    def close(self):
        """Flush remaining tokens and close"""
        if self.token_buffer:
            self._flush_buffer()
        
        print(f"Rank {self.rank}: Writer closed. Total tokens: {self.total_tokens}, Documents: {self.doc_count}")

class DatatroveWriterWrapper:
    """Wrapper to make our writer compatible with datatrove pipeline"""
    
    def __init__(self, output_dir, shard_size):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.writers = {}  # Store writers per rank
    
    def __call__(self, docs, rank=0, world_size=1):
        """Process documents from datatrove pipeline"""
        # Create writer for this rank if it doesn't exist
        if rank not in self.writers:
            self.writers[rank] = EfficientNumpyWriter(self.output_dir, self.shard_size, rank=rank)
        
        writer = self.writers[rank]
        doc_count = 0
        
        try:
            for doc in docs:
                writer.write(doc)
                doc_count += 1
                
                # Progress logging every 1000 documents
                if doc_count % 1000 == 0:
                    print(f"Rank {rank}: Processed {doc_count} documents")
            
            print(f"Rank {rank}: Completed {doc_count} documents")
            
        except Exception as e:
            print(f"Rank {rank}: Error processing documents: {e}")
            raise
            
        return docs
    
    def close(self):
        """Close all writers"""
        for rank, writer in self.writers.items():
            try:
                writer.close()
            except Exception as e:
                print(f"Error closing writer for rank {rank}: {e}")

# Create pipeline
pipeline = [
    ParquetReader(DATASET_PATH),
    LambdaFilter(tokenize_and_attach),
    DatatroveWriterWrapper(OUTPUT_DIR, SHARD_SIZE)
]

# Use optimal number of tasks for your CPU
pipeline_exec = LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=OPTIMAL_TASKS
)

if __name__ == '__main__':
    print(f"Starting pipeline with {OPTIMAL_TASKS} tasks")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Shard size: {SHARD_SIZE:,} tokens")
    
    # Check disk space before starting
    if not check_disk_space(OUTPUT_DIR, required_gb=50):
        print("Warning: Low disk space detected!")
    
    try:
        pipeline_exec.run()
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

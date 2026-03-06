import torch
import tiktoken
import argparse

from llm.gpt import Model

parser = argparse.ArgumentParser(description="Chat with the pre-trained model")

parser.add_argument(
    "--model_path",
    type=str,
    default="weights/pre_train.pth",
    help="Path to the pre-trained model checkpoint",
)

parser.add_argument(
        "--prompt",
        type=str,
        default="Quantum Mechanics is ", 
        help="Prompt to start the conversation with the pre-trained model")

parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataset")
parser.add_argument(
    "--seq_length", type=int, default=1024, help="Seq length for the model"
)
parser.add_argument(
    "--vocab_size", type=int, default=50304, help="Vocab size from the tokenizer"
)
parser.add_argument(
    "--embed_dim", type=int, default=1024, help="Hidden dimension of the MLP layers"
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=16,
    help="Number of heads for the multi-head attention",
)
parser.add_argument(
    "--num_blocks",
    type=int,
    default=8,
    help="Number of transformer blocks in the model",
)
parser.add_argument(
    "--query_heads_per_kv",
    type=int,
    default=2,
    help="Number of query heads that share each key/value head in grouped-query attention",
)

parser.add_argument(
    "--num_tokens_to_generate",
    type=int,
    default=100,
    help="Number of tokens to generate in response to the prompt",
)

args = parser.parse_args()

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda"

model = Model(
    seq_length=args.seq_length,
    vocab_size=args.vocab_size,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    num_blocks=args.num_blocks,
    query_heads_per_kv=args.query_heads_per_kv,
).to(device)

model.load_state_dict(torch.load(args.model_path, map_location=device)["model_state_dict"])
tokenizer = tiktoken.get_encoding("gpt2")
prompt_tokens = torch.tensor(tokenizer.encode(args.prompt)).unsqueeze(0).to(device)

for _ in range(args.num_tokens_to_generate):
    with torch.no_grad():
        logits = model(prompt_tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        prompt_tokens = torch.cat((prompt_tokens, next_token), dim=1)

generated_text = tokenizer.decode(prompt_tokens.squeeze().tolist())
print(f"Response: {generated_text}")

import os
import json
import torch
import torch.nn.functional as F
from llm.gpt import Model


data_dir = "../data/eval"
seq_length = 1024
vocab_size = 50304
embed_dim = 1024
num_heads = 16
query_heads_per_kv = 2
num_blocks = 8
device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

loss = torch.nn.CrossEntropyLoss()
model = Model(
    seq_length=seq_length,
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_blocks=num_blocks,
    query_heads_per_kv=query_heads_per_kv,
).to(device)

@torch.inference_mode()
def eval_generate(model, choices):
    output = model(choices)

_, _, files = next(os.walk(data_dir))
prompt_beginning = torch.tensor([50256, 24361, 25, 220], dtype=torch.long, device=device)  # "Question: "
prompt_ending = torch.tensor([198, 33706, 25, 220], dtype=torch.long, device=device) #"\nAnswer: "

for file in files:
    with open(f"{data_dir}/{file}", "r") as f:
        data = list(json.load(f))
        for element in data:
            prompts = []
            choice_lengths = []
            question = torch.tensor(element["question"], dtype=torch.long, device=device)

            # Crashing between here
            for choice in element["options"]:
                choice = torch.tensor(choice, dtype=torch.long, device=device)
                prompt = torch.cat((prompt_beginning, question, prompt_ending, choice), dim=0)


                prompt = F.pad(prompt, pad=(0, int(seq_length - prompt.size(0))), value=50256)

                prompts.append(prompt)
                choice_lengths.append(int(choice.size(0)))

            prompts = torch.stack(prompts, dim=0).to(device)
            output = model(prompts)
            # And here


            choice_index_start = len(prompt_beginning) +  int(question.size(0)) + len(prompt_ending)
            sliced_prompts = prompts[:, choice_index_start:]
            sliced_outputs = output[:, (choice_index_start - 1):]
            logits_log_prob = F.log_softmax(sliced_outputs, dim=-1)

            sliced_outputs_mask = torch.ne(sliced_outputs, 50256)
            sliced_prompts_mask = torch.ne(sliced_prompts, 50256)
            #torch.masked_select()
            print(f"{output.shape} $ {prompts.shape}")

            new_output_variable_with_no_name_because_i_cant_think_of_a_name = torch.gather(output, dim=-1, index=prompts)
            print(new_output_variable_with_no_name_because_i_cant_think_of_a_name.shape)
            length_index = torch.arange(start=0, end=(seq_length - choice_index_start), step=1, device=device)


            #print(f"length_index: {length_index.shape} & mask: {mask.shape}")
            #must_gather_indexes = torch.masked_select(length_index, mask)
            break
        break
import torch
import os
import json
from llm.tokenizer import tokenize_prompt_without_eot
import torch.nn.functional as F

@torch.inference_mode()
def eval_generate(model, prompts):
    return model(prompts)

def Arc_Easy(data_dir, model, device, seq_length):
    _, _, files = next(os.walk(data_dir))
    number_of_correct = 0
    total = 0
    for file in files:
        with open(f"{data_dir}/{file}", "r") as f:
            data = list(json.load(f))
            for element in data:
                total += 1
                prompts = []
                prefix = "Question: " + element["question"] + "\nAnswer:"
                choice_index_start = len(tokenize_prompt_without_eot(prefix))

                # TODO: Instead of running each question one-by-one we can batch the different questions
                for choice in element["options"]:
                    prompt = torch.tensor(tokenize_prompt_without_eot(prefix + " " + choice), dtype=torch.long)
                    prompt_padded = F.pad(prompt, pad=(0, int(seq_length - prompt.size(0))), value=50256)
                    prompts.append(prompt_padded)

                prompts = torch.stack(prompts, dim=0).to(device)
                output = eval_generate(model, prompts)

                prompts = prompts[:, choice_index_start:]
                output = output[:, (choice_index_start - 1):]
                output = F.log_softmax(output, dim=-1)
                output = torch.gather(output, dim=-1, index=prompts.unsqueeze(dim=-1)).squeeze(-1)
                sliced_outputs_mask = torch.eq(prompts, 50256)
                output_gather = torch.masked_fill(output, sliced_outputs_mask, 0)
                total_log_probablity = output_gather.sum(dim=-1)
                model_correct_answer = torch.argmax(total_log_probablity, dim=-1)

                if model_correct_answer.item() == element["answer"]:
                    number_of_correct += 1

    return number_of_correct / total

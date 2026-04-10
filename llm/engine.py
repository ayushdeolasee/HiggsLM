import torch 
from llm.tokenizer import str_to_pre_train_tokens, decode_tokens

@torch.inference_mode()
def generate_pre_train_eval(model, device, prompt, max_tokens):
    model = model.to(device)
    tokens = str_to_pre_train_tokens(prompt).to(device)
    
    logits = model(tokens)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    tokens = torch.cat((tokens, next_token), dim = 1)
    return tokens

@torch.inference_mode()
def generate_pre_train(model, device, prompt, max_tokens):
    model = model.to(device)
    tokens = str_to_pre_train_tokens(prompt).to(device)
    print(f"Tokens: {tokens}") 
    for _ in range(max_tokens):
        logits = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim = 1)
    
    generated_text = decode_tokens(tokens.squeeze().tolist()) 
    return [tokens, generated_text]

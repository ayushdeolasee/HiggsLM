from deepeval.benchmarks import HellaSwag as DeepEvalHellaSwag
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks.tasks import HellaSwagTask
from llm.engine import generate_pre_train
#Pre-train evals

class PreTrainModel(DeepEvalBaseLLM): 
    def __init__(self, model, tokenizer, device, max_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str: 
        model = self.load_model()
        print(model) 
        return generate_pre_train(model, self.device, prompt, self.max_tokens)[1]

    async def a_generate(self, prompt) -> str:
        return self.generate(prompt)
    
    def get_model_name(self):
        return "HiggsLM-pre-train"

def HellaSwag(model, n_shots = 5):
    benchmark = DeepEvalHellaSwag(n_shots=n_shots, tasks=[HellaSwagTask.EDUCATION_AND_COMMUNICATIONS] )
    print(f"Running evaluations") 
    benchmark.evaluate(model=model)
    
    print(benchmark.overall_score)
    return benchmark

def Arc_Easy():
    pass


# TODO: Implement evals and then implement an eval manager to run an eval every x epochs

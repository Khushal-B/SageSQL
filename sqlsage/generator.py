"""SQL generation with the Qwen3-4B base model plus the LoRA adapter, served by vLLM."""
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

BASE_MODEL = "Qwen/Qwen3-4B"
PROMPT = "### Schema:\n{schema}\n### Question:\n{question}\n### SQL:\n"


class SQLGenerator:
    def __init__(self, adapter_path="outputs/lora_adapter"):
        self.llm = LLM(BASE_MODEL, enable_lora=True)
        self.lora = LoRARequest("sqlsage", 1, adapter_path)
        self.params = SamplingParams(temperature=0.0, max_tokens=256, stop=[";"])

    def generate(self, schema_prompt, question):
        prompt = PROMPT.format(schema=schema_prompt, question=question)
        out = self.llm.generate([prompt], self.params, lora_request=self.lora)
        return out[0].outputs[0].text.strip() + ";"

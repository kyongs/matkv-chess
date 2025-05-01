import re
import torch
from typing import Any, List, Optional
from langchain_core.pydantic_v1 import Field, PrivateAttr
from langchain.llms.base import LLM
from langchain.schema import AIMessage
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

class HuggingFaceLlamaEngine(LLM):
    model_name_or_path: str = Field(default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    device: str = Field(default="cpu")
    torch_dtype: str = Field(default="float32")  # for model weight dtype
    max_new_tokens: int = Field(default=512)

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Determine load dtype for model and compute dtype for 4-bit quant
        load_dtype = torch.float32 if self.device == "cpu" else getattr(torch, self.torch_dtype)
        # Blockwise quant supports only float16 or float32
        bnb_compute_dtype = torch.float32 if self.device == "cpu" else torch.float16

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Setup 4-bit quantization with correct compute dtype
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        # Load model with quantization config
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=load_dtype,
            trust_remote_code=True,
        )

    @property
    def _llm_type(self) -> str:
        return "huggingface_llama_engine"

    def _call(self, prompt_text: str, stop: Optional[List[str]] = None) -> str:
        # Wrap for Instruct format
        if not prompt_text.strip().startswith("[INST]"):
            prompt_text = f"[INST] {prompt_text} [/INST]"

        # Tokenize
        enc = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"][0].to(self._model.device)

        # Trim to fit context
        max_ctx = self._model.config.max_position_embeddings
        max_input = max_ctx - self.max_new_tokens
        if input_ids.size(-1) > max_input:
            input_ids = input_ids[-max_input:]

        # Attention mask
        attention_mask = torch.ones_like(input_ids, device=self._model.device)

        # Generation config
        gen_config = GenerationConfig(max_new_tokens=self.max_new_tokens)

        # Generate
        outputs = self._model.generate(
            input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            generation_config=gen_config,
            return_dict_in_generate=True,
        )

        # Decode new tokens
        gen_ids = outputs.sequences[0][input_ids.size(-1):]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Extract first <tool_call>
        match = re.search(r"<tool_call>.*?</tool_call>", text, flags=re.DOTALL)
        if match:
            return match.group(0)

        # Apply stop sequences
        if stop:
            for seq in stop:
                if seq in text:
                    text = text.split(seq)[0]
        return text.strip()

    def invoke(self, *args, **kwargs) -> AIMessage:
        # Determine prompt object
        if args:
            prompt_obj = args[0]
        else:
            prompt_obj = kwargs.get('message') or kwargs.get('prompt')

        # Handle ChatPromptValue with to_messages
        if hasattr(prompt_obj, 'to_messages'):
            msgs = prompt_obj.to_messages()
            lines: List[str] = []
            for m in msgs:
                role = getattr(m, 'role', None) or getattr(m, 'type', None) or m.__class__.__name__.lower()
                lines.append(f"{role}: {m.content}")
            prompt_text = "\n".join(lines)
        elif hasattr(prompt_obj, 'content'):
            prompt_text = prompt_obj.content
        else:
            prompt_text = str(prompt_obj)

        stop = kwargs.get("stop")
        generated_text = self._call(prompt_text, stop=stop)
        return AIMessage(content=generated_text)
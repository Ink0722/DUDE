from typing import List, Callable, Any, Dict
from types import MethodType
from zai import ZhipuAiClient
import torch
import os
from src.config import require_zhipuai_api_key
from transformers import AutoProcessor, AutoModelForVision2Seq, Glm4vForConditionalGeneration, AutoModelForImageTextToText


def _resolve_device(device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class BASE:
    def __init__(self, 
                 model_name: str,
                 SYSTEM_PROMPT: str=None,
                 tools: List[Callable]=[],
                 device: str=None
                 ):
        self.tools = { func.__name__: func for func in (tools or []) }
        self.model_name = model_name
        self.system_prompt = SYSTEM_PROMPT
        
        self.device = _resolve_device(device)
        
        # Token usage tracking
        self.total_tokens = 0
        self.last_call_tokens = 0


    def update_system_prompt(self, new_system_prompt: str):
        self.system_prompt = new_system_prompt

        
class Local(BASE):
    # Local handles base-model + finetuned-adapter loading for training/evaluation flows.
    def __init__(self, model_name: str = "Qwen3-VL-4B-Instruct", SYSTEM_PROMPT: str = None, tools: List = None, model_path: str = None):
        
        """Initialize a local agent wrapper.

        Args:
            model_name: base model identifier
            SYSTEM_PROMPT: system prompt text
            tools: tool list
            model_path: optional path to finetuned weights
        """
        
        super().__init__(model_name, SYSTEM_PROMPT, tools or [])

        # Initialize model, processor, and optional finetuned weights.
        self.model = None
        self.processor = None
        self.model_path = model_path

        # Load the base model first.
        self._load_base_model(model_name)

        # Load finetuned weights when a valid path is provided.
        if model_path and os.path.exists(model_path):
            self._load_finetuned_model(model_path)

        # Initialize conversation context.
        self.context = [{"role": "system", "content": self.system_prompt or ""}]
    
    _MODEL_FAMILIES = {
    # Registry-style routing keeps family matching in one place.
        "qwen3": {
            "match": lambda name: "qwen3" in name,
            "base_loader": "vision2seq",
            "finetuned_loader": "vision2seq",
        },
        "ui-tars": {
            "match": lambda name: "ui-tars" in name or "ui_tars" in name,
            "base_loader": "vision2seq",
            "finetuned_loader": "vision2seq",
        },
    }

    @staticmethod
    def _load_base_vision2seq(model_name: str, device: str):
        print(f"Loading model: {model_name}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            padding_side="left",
        )
        return model, processor

    @staticmethod
    def _load_finetuned_base_vision2seq(base_name: str, device: str):
        print(f"Loading base model for finetuned weights: {base_name}")
        return AutoModelForVision2Seq.from_pretrained(
            base_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    @classmethod
    def _resolve_family(cls, model_name: str) -> str:
        # Map model name to a known family key; fail fast for unsupported models.
        name = model_name.lower()
        for key, spec in cls._MODEL_FAMILIES.items():
            if spec["match"](name):
                return key
        raise ValueError(f"Unsupported model: {model_name}")

    def _load_base_model(self, model_name: str):
        # Resolve family first, then dispatch to the configured loader.
        """Load the base model backend."""
        try:
            family = self._resolve_family(model_name)
            loader_key = self._MODEL_FAMILIES[family]["base_loader"]
            self.model, self.processor = self._load_base_vision2seq(model_name, self.device) if loader_key == "vision2seq" else (None, None)

            # Add a safe batch decoder for local multimodal models.
            def safe_batch_decode(self, sequences, **kwargs):
                pad_id = self.tokenizer.pad_token_id or 0
                vocab_size = len(self.tokenizer)

                def clean_one(seq):
                    if isinstance(seq, torch.Tensor):
                        ids = seq.clone().detach().cpu().tolist()
                    else:
                        ids = list(seq)

                    cleaned = []
                    for x in ids:
                        try:
                            v = int(x)
                        except Exception:
                            v = pad_id
                        if v < 0 or v >= vocab_size:
                            v = pad_id
                        cleaned.append(v)
                    return cleaned

                if isinstance(sequences, torch.Tensor):
                    cleaned_batch = [clean_one(row) for row in sequences]
                else:
                    cleaned_batch = [clean_one(row) for row in sequences]

                return self.tokenizer.batch_decode(cleaned_batch, **kwargs)

            if self.processor is not None:
                self.processor.batch_decode = MethodType(safe_batch_decode, self.processor)

            print("Base model loaded successfully")
        except Exception as e:
            print(f"Error loading base model: {e}")
            raise

    def _load_finetuned_model(self, model_path: str):
        # Rebuild the base model from family before attaching PEFT weights.
        """Load finetuned weights on top of the current base model."""
        try:
            from peft import PeftModel
            print(f"Loading finetuned model from: {model_path}")

            base_name = (
                self.model.config._name_or_path
                if hasattr(self.model, "config") and hasattr(self.model.config, "_name_or_path")
                else self.model_name
            )
            family = self._resolve_family(str(base_name))
            loader_key = self._MODEL_FAMILIES[family]["finetuned_loader"]
            base_model = self._load_finetuned_base_vision2seq(base_name, self.device) if loader_key == "vision2seq" else None

            # Load LoRA / PEFT weights on top of the selected base model.
            self.model = PeftModel.from_pretrained(base_model, model_path)
            print("Finetuned model loaded successfully")

        except Exception as e:
            print(f"Error loading finetuned model: {e}")
            print("Using base model instead")


        
class GLM(BASE):
    """Wrapper for Zhipu chat models accessed through the official SDK."""

    def __init__(self, model_name: str, api_key: str = None, SYSTEM_PROMPT: str = None, tools: List = None):
        super().__init__(model_name, SYSTEM_PROMPT, tools)
        self.api_key = require_zhipuai_api_key(api_key)
        self.client = ZhipuAiClient(api_key=self.api_key)

    def call_model(self, messages, think: bool = False):
        """Call the remote GLM model and return plain text content."""
        thinking_mode = "enabled" if think else "disabled"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            thinking={
                "type": thinking_mode
            }
        )

        # Token counting
        if hasattr(response, "usage") and response.usage:
            self.last_call_tokens = getattr(response.usage, "total_tokens", 0)
            self.total_tokens += self.last_call_tokens

        return response.choices[0].message.content


class BaseLocalBackend(BASE):
    
    # Shared runtime backend for agent-side multimodal inference backends.
    """Shared backend for local multimodal models."""

    model_class = None
    processor_class = AutoProcessor
    needs_token_type_cleanup = False

    def __init__(
        self,
        model_name: str,
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")
        self.device = _resolve_device(device)
        
        if self.device == "cpu":
            # Keep runtime behavior explicit: these local backends are GPU-only.
            raise RuntimeError("CPU loading is not supported for local backends")
        if self.model_class is None:
            raise ValueError("model_class must be set in subclass")

        try:
            print(f"Loading model: {model_name} on {self.device}")
            self.processor = self.processor_class.from_pretrained(model_name)
            self.model = self.model_class.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ReActAgent messages into the model input format."""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Normalize content to list[dict].
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")

                    # From ReActAgent: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    # Support existing {"type": "image", "image": path} entries.
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    # Support {"type": "image", "url": path} entries.
                    elif item_type == "image" and "url" in item:
                        new_content.append(item)
                    else:
                        # Keep other content unchanged, such as text items.
                        new_content.append(item)
            else:
                # Skip unsupported content structures.
                continue

            converted.append({"role": role, "content": new_content})

        return converted

    def call_model(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512, **generate_kwargs) -> str:
        try:
            if not hasattr(self, "model") or not hasattr(self, "processor"):
                return "Error: model not loaded"

            model_messages = self._convert_messages(messages)

            inputs = self.processor.apply_chat_template(
                model_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move inputs to the model device.
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            if self.needs_token_type_cleanup and isinstance(inputs, dict):
            # Some model families reject token_type_ids in generation inputs.
                inputs.pop("token_type_ids", None)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

            # Decode only the newly generated tokens.
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]

            # Token counting
            input_tokens = inputs["input_ids"].numel()
            output_tokens = generated_ids.numel()
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens

            text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return text
        except Exception as e:
            print(f"Error in inference: {e}")
            return f"Error: {str(e)}"


class Qwen3VLBackend(BaseLocalBackend):
    
    """Backend wrapper for local Qwen3-VL models used by ReActAgent."""

    model_class = AutoModelForVision2Seq

class UITARSBackend(BaseLocalBackend):
    
    """Backend wrapper for ByteDance UI-TARS-1.5-7B."""

    model_class = AutoModelForVision2Seq

class GLMFlashBackend(BaseLocalBackend):
    
    """Backend wrapper for zai-org/GLM-4.6V-Flash."""

    model_class = Glm4vForConditionalGeneration
    needs_token_type_cleanup = True

def _resolve_local_backend_class(model_name: str):
    """Infer the local backend class directly from the model name.

    Note:
        Remote GLM API routing is intentionally not handled here yet.
        This factory currently supports only local multimodal backends.
    """
    name = model_name.lower()
    if "qwen3" in name:
        return Qwen3VLBackend
    if "ui-tars" in name or "ui_tars" in name:
        return UITARSBackend
    if "glm-4.6v-flash" in name or "glm_flash" in name:
        return GLMFlashBackend

    raise ValueError(f"Unsupported model_name for local backend: {model_name}")


def build_backend(model_name: str, device: str | None = None):
    """Unified local backend factory used by ReActAgent."""
    client_cls = _resolve_local_backend_class(model_name)
    return client_cls(model_name=model_name, device=device)


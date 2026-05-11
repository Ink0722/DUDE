import json
import os

import torch
from datetime import datetime
from types import MethodType

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from src.config import SETTINGS
from src.model import Local
from train.datasets import load_local_dataset
from train.formatter import make_conversation
from train.reward import hybrid_label_confidence_reward


DEFAULT_QWEN3_TRAIN_MODEL = SETTINGS.default_eval_model


class IntegratedTrainOptimize:
    def __init__(
        self,
        model_id=None,
        data_path=SETTINGS.data_path,
        images_dir=SETTINGS.images_dir,
        output_dir=SETTINGS.stage1_root,
        device=SETTINGS.default_device,
        verbose=False,
        log_samples=False,
    ):
        self.model_id = model_id or DEFAULT_QWEN3_TRAIN_MODEL
        self.data_path = data_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.device = device
        self.verbose = verbose
        self.log_samples = log_samples

        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

        self.model = None
        self.processor = None

        self.trained_model_path = None
        self.stage1_snapshot_path = None
        self.recorded_samples_path = None
        self.run_timestamp = None

    def _log(self, message: str) -> None:
        print(message)

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _log_sample(self, title: str, sample) -> None:
        if self.log_samples:
            print(title)
            print(sample)

    def _get_torch_dtype(self):
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    def _get_model_load_kwargs(self):
        kwargs = {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": self._get_torch_dtype(),
        }
        if self.device in {"auto", "cuda"}:
            kwargs["device_map"] = "auto"
        return kwargs

    def _ensure_run_timestamp(self) -> str:
        if self.run_timestamp is None:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.run_timestamp

    def _build_model_dir(self) -> str:
        ts = self._ensure_run_timestamp()
        return os.path.join(self.output_dir, f"evaluator_{ts}")

    def _build_stage1_result_path(self) -> str:
        ts = self._ensure_run_timestamp()
        return os.path.join(self.output_dir, f"stage1_{ts}.jsonl")

    def _build_record_path(self) -> str:
        ts = self._ensure_run_timestamp()
        return os.path.join(self.output_dir, f"record_{ts}.jsonl")

    def load_data(self, test_size=0.2, seed=42):
        """Load the dataset, split it, and prepare training conversations."""
        self._log("Loading dataset...")
        self.dataset = load_local_dataset(self.data_path, self.images_dir, load_images=False, Train=True)

        split_dataset = self.dataset.train_test_split(test_size=test_size, seed=seed)
        self._log_sample("Train sample preview:", split_dataset["train"][0])
        self._log_sample("Test sample preview:", split_dataset["test"][0])
        self.train_dataset = split_dataset["train"]
        self.test_dataset = split_dataset["test"]
        self._debug(f"Train split size: {len(self.train_dataset)}")
        self._debug(f"Test split size: {len(self.test_dataset)}")

        self.train_dataset = self.train_dataset.map(make_conversation)
        self._log_sample("Mapped training sample preview:", self.train_dataset[0])

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = self._build_stage1_result_path()
        with open(out_path, "w", encoding="utf-8") as fw:
            for r in self.train_dataset:
                rec_copy = dict(r) if isinstance(r, dict) else {"record": r}
                rec_copy["status"] = None
                fw.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")
        self.stage1_snapshot_path = out_path
        self._debug(f"Saved dataset snapshot to {out_path}")

        self._log(f"Loaded {len(self.train_dataset)} training samples and {len(self.test_dataset)} test samples")

    def setup_model(self):
        """Set up the base model, processor, and LoRA adapters."""
        self._log("Setting up model and processor...")

        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True, padding_side="left")

        model_load_kwargs = self._get_model_load_kwargs()
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                attn_implementation="flash_attention_2",
                **model_load_kwargs,
            )
        except Exception as exc:
            self._debug(f"Falling back to the default attention backend: {exc}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(**model_load_kwargs)

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )

        self.model = get_peft_model(self.model, lora_config)
        if self.verbose:
            self.model.print_trainable_parameters()

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

        self.processor.batch_decode = MethodType(safe_batch_decode, self.processor)

    def train_model(
        self,
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=512,
        max_prompt_length=8192,
        logging_steps=10,
        save_steps=10,
    ):
        """Train the model with GRPO and save the resulting checkpoint."""
        self._log("Starting model training...")

        os.makedirs(self.output_dir, exist_ok=True)
        ts = self._ensure_run_timestamp()
        recorded_path = self._build_record_path()
        model_dir = self._build_model_dir()
        self.recorded_samples_path = recorded_path

        self._debug(
            f"Training config: epochs={num_train_epochs}, batch_size={per_device_train_batch_size}, lr={learning_rate}"
        )
        self._debug(f"Reward samples will be recorded to: {recorded_path}")

        snapshot = self.stage1_snapshot_path

        def _reward_wrapper(*a, **kw):
            return hybrid_label_confidence_reward(
                *a,
                recorded_samples_path=recorded_path,
                snapshot_path=snapshot,
                run_ts=ts,
                **kw,
            )

        try:
            _reward_wrapper.__name__ = getattr(
                hybrid_label_confidence_reward,
                "__name__",
                "hybrid_label_confidence_reward",
            )
        except Exception:
            pass

        reward_funcs = [_reward_wrapper]

        training_args = GRPOConfig(
            output_dir=model_dir,
            learning_rate=learning_rate,
            remove_unused_columns=False,
            num_train_epochs=num_train_epochs,
            bf16=True,
            per_device_train_batch_size=per_device_train_batch_size,
            max_completion_length=max_completion_length,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
        )

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.processor,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=self.train_dataset,
        )

        trainer.train()
        trainer.save_model(model_dir)
        self.trained_model_path = model_dir

        self._log(f"Model training completed. Model saved to {model_dir}")

    def load_trained_model(self, model_path=None):
        """Load a trained checkpoint for downstream optimization or evaluation."""
        if model_path is None:
            model_path = self.trained_model_path

        if model_path is None:
            raise ValueError("No trained model path available. Please train the model first.")

        self._log(f"Loading trained model from {model_path}...")

        self.agent = Local(
            model_name=self.model_id,
            SYSTEM_PROMPT="",
            tools=[],
        )

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            **self._get_model_load_kwargs(),
        )
        self.agent.model = PeftModel.from_pretrained(base_model, model_path)

        self._log("Trained model loaded successfully")

    def run_full_pipeline(self, train_params=None):
        """Run the end-to-end training pipeline."""
        self._log("Starting full training pipeline...")

        self.load_data()
        self.setup_model()

        train_params = train_params or {}
        self.train_model(**train_params)


def main():
    """Example entry point for the integrated training pipeline."""
    system = IntegratedTrainOptimize(verbose=False, log_samples=False)
    system.run_full_pipeline(
        train_params={
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
        }
    )


if __name__ == "__main__":
    main()

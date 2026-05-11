from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_default_device() -> str:
    return os.getenv("DEFAULT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Settings:
    project_root: Path
    dataset_root: str
    data_path: str
    images_dir: str
    output_dir: str
    stage1_root: str
    stage2_root: str
    inference_root: str
    default_agent_model: str
    default_eval_model: str
    default_device: str
    hf_endpoint: str | None
    zhipuai_api_key: str | None


SETTINGS = Settings(
    project_root=PROJECT_ROOT,
    dataset_root=os.getenv("DATASET_ROOT", "data/Real-UI-Clickboxes"),
    data_path=os.getenv("DATA_PATH", "data/Real-UI-Clickboxes/train.json"),
    images_dir=os.getenv("IMAGES_DIR", "data/Real-UI-Clickboxes/images"),
    output_dir=os.getenv("OUTPUT_DIR", "outputs"),
    stage1_root=os.getenv("STAGE1_ROOT", "data/stage1"),
    stage2_root=os.getenv("STAGE2_ROOT", "data/stage2"),
    inference_root=os.getenv("INFERENCE_ROOT", "Qwen/Qwen3-VL-4B-Thinking"),
    default_eval_model=os.getenv("DEFAULT_EVAL_MODEL", "Qwen/Qwen3-VL-2B-Thinking"),
    default_device=_get_default_device(),
    hf_endpoint=os.getenv("HF_ENDPOINT"),
    zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY"),
)

if SETTINGS.hf_endpoint and "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = SETTINGS.hf_endpoint


def require_zhipuai_api_key(api_key: str | None = None) -> str:
    value = api_key or SETTINGS.zhipuai_api_key
    if not value:
        raise ValueError(
            "ZHIPUAI_API_KEY is required for GLM-based workflows. Set it in the environment or in .env."
        )
    return value

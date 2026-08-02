"""Save and resume PPO training checkpoints."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Train.minmax_penalty import MinmaxPenaltyState


def experiment_dir(steps: int, seed: int) -> Path:
    return Path(f"experiments/seed_{seed}_steps_{steps}")


def default_log_path(steps: int, seed: int, condition: str) -> Path:
    return experiment_dir(steps, seed) / "logs" / f"{condition}_training.csv"


def default_save_dir(steps: int, seed: int, condition: str) -> Path:
    return experiment_dir(steps, seed) / "checkpoints" / condition


def checkpoint_path(save_dir: str | Path, step: int) -> Path:
    return Path(save_dir) / "checkpoints" / f"step_{step}"


def find_latest_checkpoint(save_dir: str | Path) -> Path | None:
    root = Path(save_dir) / "checkpoints"
    if not root.is_dir():
        return None
    matches = []
    for path in root.iterdir():
        m = re.fullmatch(r"step_(\d+)", path.name)
        if m and path.is_dir():
            matches.append((int(m.group(1)), path))
    if not matches:
        return None
    return max(matches, key=lambda x: x[0])[1]


def _read_meta(path: Path) -> dict:
    meta_file = path / "training_state.json"
    if meta_file.is_file():
        return json.loads(meta_file.read_text(encoding="utf-8"))
    return {}


def resolve_resume_path(resume_from: str | Path) -> tuple[Path, int]:
    """
    Return (checkpoint directory, next training step).

    Accepts a step folder (…/checkpoints/step_500), a run root (…/checkpoints/baseline),
    or a legacy flat model dir (models/baseline) with optional training_state.json.
    """
    path = Path(resume_from)
    if not path.is_dir():
        raise FileNotFoundError(f"Resume path not found: {path}")

    if re.fullmatch(r"step_\d+", path.name):
        meta = _read_meta(path)
        step = int(meta.get("step", int(path.name.split("_")[1])))
        return path, step + 1

    latest = find_latest_checkpoint(path)
    if latest is not None:
        meta = _read_meta(latest)
        step = int(meta.get("step", int(latest.name.split("_")[1])))
        return latest, step + 1

    if (path / "config.json").is_file():
        meta = _read_meta(path)
        if "step" in meta:
            return path, int(meta["step"]) + 1
        return path, 0

    raise FileNotFoundError(
        f"Could not find a model checkpoint under {path}. "
        "Expected checkpoints/step_N/ or config.json at the path."
    )


def save_checkpoint(
    save_dir: str | Path,
    step: int,
    trainer,
    tokenizer,
    *,
    condition: str,
    seed: int,
    steps_target: int,
    penalty_state: MinmaxPenaltyState | None = None,
) -> Path:
    """Write step checkpoint and refresh the run root (for eval)."""
    save_dir = Path(save_dir)
    ckpt = checkpoint_path(save_dir, step)
    ckpt.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)

    meta = {
        "step": step,
        "seed": seed,
        "steps_target": steps_target,
        "condition": condition,
    }
    (ckpt / "training_state.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if penalty_state is not None:
        (ckpt / "penalty_state.json").write_text(
            json.dumps(penalty_state.state_dict(), indent=2), encoding="utf-8"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "training_state.json", "penalty_state.json"):
        src = ckpt / name
        if src.is_file():
            shutil.copy2(src, save_dir / name)
    for pattern in ("*.safetensors", "pytorch_model.bin", "model*.safetensors"):
        for src in ckpt.glob(pattern):
            shutil.copy2(src, save_dir / src.name)

    return ckpt


def load_penalty_state(path: Path, penalty_state: MinmaxPenaltyState) -> None:
    state_file = path / "penalty_state.json"
    if state_file.is_file():
        penalty_state.load_state_dict(json.loads(state_file.read_text(encoding="utf-8")))

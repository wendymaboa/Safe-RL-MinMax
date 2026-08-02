"""Save raw (untrained) GPT-2 weights locally for eval comparison."""
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

SAVE_PATH = Path("models/gpt2_raw")

if SAVE_PATH.exists() and (SAVE_PATH / "model.safetensors").exists():
    print(f"Already exists: {SAVE_PATH} — skipping download.")
    sys.exit(0)

print("Downloading raw gpt2 weights from HuggingFace...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

SAVE_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Saved to {SAVE_PATH}")
print("Files:", [f.name for f in SAVE_PATH.iterdir()])

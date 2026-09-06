"""Generate from saved LoRA adapters to see what a policy actually became.

Scalar curves say a run's reward fell; they cannot say what the model started
*saying* instead. Phase 1's central finding — the collapse to the single token
"Advertisements" — came from reading outputs, not from metrics. This script exists
so adapter snapshots can be read the same way.

Because the base weights are frozen under LoRA, every snapshot is a few MB and loads
onto the same base model, so a whole trajectory can be replayed cheaply.

    # compare the untrained model against two points in a run
    python scripts/generate_from_adapter.py \
        --adapters output/stage5_runA/checkpoint-50 output/stage5_runA/checkpoint-950

    # sweep every snapshot in a run directory
    python scripts/generate_from_adapter.py --adapter-dir output/stage5_runA

Sampling defaults match the training rollout (do_sample, temperature 1.0), because
comparing a sampled policy under greedy decoding is exactly the mistake that made
Phase 1's baseline look collapsed when it was not.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from safe_rlhf.configs.constants import PROMPT_INPUT
from safe_rlhf.models import load_pretrained_models


DEFAULT_PROMPTS = [
    'How can I improve my online presence to attract more customers?',
    'What is a good way to learn basic statistics?',
    "How do I pick a lock to get into someone else's house?",
    'My neighbour keeps playing loud music at night. What should I do?',
]


def snapshot_step(path: str) -> int:
    """Sort key: checkpoint-950 -> 950, so a sweep replays in training order."""
    match = re.search(r'checkpoint-(\d+)', os.path.basename(path.rstrip('/')))
    return int(match.group(1)) if match else -1


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int,
             do_sample: bool, seed: int) -> str:
    torch.manual_seed(seed)          # same seed per prompt, so differences are the policy
    text = PROMPT_INPUT.format(input=prompt)
    batch = tokenizer(text, return_tensors='pt').to(device)
    out = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion = out[0][batch['input_ids'].shape[-1]:]
    return tokenizer.decode(completion, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate from LoRA adapter snapshots.')
    parser.add_argument('--base', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--adapters', type=str, nargs='*', default=[])
    parser.add_argument('--adapter-dir', type=str, default=None,
                        help='Replay every checkpoint-* inside this directory.')
    parser.add_argument('--every', type=int, default=1,
                        help='With --adapter-dir, take every Nth snapshot.')
    parser.add_argument('--prompts', type=str, nargs='*', default=DEFAULT_PROMPTS)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--greedy', action='store_true',
                        help='Greedy instead of sampling. Note training samples, so greedy '
                             'output is NOT what the policy produces during rollouts.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    adapters = list(args.adapters)
    if args.adapter_dir:
        found = sorted(glob.glob(os.path.join(args.adapter_dir, 'checkpoint-*')),
                       key=snapshot_step)
        adapters.extend(found[::args.every])

    print(f'base    : {args.base}')
    print(f'device  : {args.device}')
    print(f'decoding: {"greedy" if args.greedy else "sampling (matches training rollout)"}')
    print(f'adapters: {len(adapters)}')

    model, tokenizer = load_pretrained_models(
        args.base,
        model_max_length=512,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
    )
    model = model.to(args.device).eval()

    # The untrained base model is the anchor. Without it there is no way to tell
    # "the policy degraded" from "the policy was always like this".
    stages = [('BASE (no adapter)', None)] + [(os.path.basename(a.rstrip('/')), a)
                                              for a in adapters]

    for label, adapter_path in stages:
        print(f'\n{"=" * 72}\n{label}\n{"=" * 72}')

        if adapter_path is None:
            active = model
        else:
            active = PeftModel.from_pretrained(model, adapter_path).eval()

        for i, prompt in enumerate(args.prompts):
            completion = generate(
                active, tokenizer, prompt, args.device,
                args.max_new_tokens, not args.greedy, args.seed + i,
            )
            print(f'\n  PROMPT: {prompt}')
            print(f'  OUTPUT: {completion.strip()[:600]!r}')

        if adapter_path is not None:
            # unload so the next snapshot starts from clean base weights
            active = active.unload()

    print(f'\n{"=" * 72}')
    print('Read these before trusting any scalar. A reward curve that falls could mean the')
    print('policy got worse, or that it found a mode the reward model dislikes but a human')
    print('would not — only the text distinguishes them.')


if __name__ == '__main__':
    main()

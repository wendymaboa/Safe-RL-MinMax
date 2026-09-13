#!/usr/bin/env python3
"""Check whether actor and Beaver scorer tokenizers match (no batch_retokenize).

Torch-free on purpose: login nodes often hang on ``import torch`` / CUDA init.
Uses huggingface_hub + the Rust ``tokenizers`` package only. Vocab equality is
what Safe-RLHF's ``is_same_tokenizer`` ultimately requires (plus class match,
which Llama-family AutoTokenizers share when vocabs match).

Usage — Llama track (must match; fail the job if not):

    python scripts/verify_tokenizer_alignment.py \\
        --actor TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --reward PKU-Alignment/beaver-7b-unified-reward \\
        --cost PKU-Alignment/beaver-7b-unified-cost \\
        --require-same

Usage — Qwen track (retokenize expected; still exit 0):

    python scripts/verify_tokenizer_alignment.py \\
        --actor Qwen/Qwen2.5-1.5B-Instruct \\
        --reward PKU-Alignment/beaver-7b-unified-reward
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _download_tokenizer_files(name_or_path: str) -> Path:
    """Return a local dir that contains tokenizer.json (or tokenizer.model)."""
    local = Path(name_or_path)
    if local.is_dir():
        return local

    from huggingface_hub import hf_hub_download

    # Prefer tokenizer.json (fast); fall back to vocab files if needed.
    try:
        path = hf_hub_download(repo_id=name_or_path, filename='tokenizer.json')
        return Path(path).parent
    except Exception as exc:  # noqa: BLE001 — report and try sentencepiece
        print(f'  tokenizer.json missing ({exc}); trying tokenizer.model', flush=True)
        path = hf_hub_download(repo_id=name_or_path, filename='tokenizer.model')
        return Path(path).parent


def _load_vocab(name_or_path: str) -> dict[str, int]:
    print(f'  resolve tokenizer files for {name_or_path!r} ...', flush=True)
    directory = _download_tokenizer_files(name_or_path)
    json_path = directory / 'tokenizer.json'
    if json_path.is_file():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(json_path))
        return tok.get_vocab()

    # Slow path: sentencepiece model → id map via protobuf-free SPM if available
    model_path = directory / 'tokenizer.model'
    if model_path.is_file():
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        return {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}

    raise FileNotFoundError(
        f'No tokenizer.json or tokenizer.model under {directory} for {name_or_path}',
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', required=True, help='Actor / policy model id or path')
    parser.add_argument('--reward', required=True, help='Reward model id or path')
    parser.add_argument(
        '--cost',
        default=None,
        help='Optional cost model id or path (Run B/C)',
    )
    parser.add_argument(
        '--require-same',
        action='store_true',
        help='Exit 1 if any scorer tokenizer differs from the actor (Llama track)',
    )
    args = parser.parse_args()

    print(f'Loading actor vocab:  {args.actor}', flush=True)
    actor_vocab = _load_vocab(args.actor)
    print(f'  size={len(actor_vocab)}', flush=True)

    print(f'Loading reward vocab: {args.reward}', flush=True)
    reward_vocab = _load_vocab(args.reward)
    print(f'  size={len(reward_vocab)}', flush=True)

    same_reward = actor_vocab == reward_vocab
    print(f'  actor ↔ reward  same={same_reward}', flush=True)
    if same_reward:
        print('  -> reward scoring will NOT call batch_retokenize', flush=True)
    else:
        print('  -> reward scoring WILL decode/re-encode via batch_retokenize', flush=True)

    same_cost = True
    if args.cost:
        print(f'Loading cost vocab:   {args.cost}', flush=True)
        cost_vocab = _load_vocab(args.cost)
        print(f'  size={len(cost_vocab)}', flush=True)
        same_cost = actor_vocab == cost_vocab
        print(f'  actor ↔ cost    same={same_cost}', flush=True)
        if same_cost:
            print('  -> cost scoring will NOT call batch_retokenize', flush=True)
        else:
            print('  -> cost scoring WILL decode/re-encode via batch_retokenize', flush=True)

    all_same = same_reward and same_cost
    if args.require_same and not all_same:
        print(
            '\nERROR: --require-same set but tokenizer vocab(s) differ. '
            'Pick a Llama-2-vocab policy (or fix scorers) before training this track.',
            file=sys.stderr,
        )
        return 1

    if all_same:
        print('\nOK: scorers share the actor vocab (no double-tokenizer path).', flush=True)
    else:
        print(
            '\nOK: mismatch noted — Qwen-style track; retokenize will run '
            '(not a failure unless --require-same).',
            flush=True,
        )
    return 0


if __name__ == '__main__':
    # Keep this script import-side-effect free of torch.
    raise SystemExit(main())

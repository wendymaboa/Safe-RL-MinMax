#!/usr/bin/env python3
"""Check whether actor and Beaver scorer tokenizers match (no batch_retokenize).

Torch-free on purpose: login nodes often hang on ``import torch`` / CUDA init.
Uses huggingface_hub + the Rust ``tokenizers`` package only.

Simulates Safe-RLHF's ``resize_tokenizer_embedding`` rule: if ``pad_token`` is
missing from tokenizer_config, ``<pad>`` is added at id=len(vocab) (same as
training). Models that already set ``pad_token`` to ``</s>`` (e.g. TinyLlama)
do **not** get that add and will correctly fail ``--require-same``.

Usage — Llama track (must match; fail the job if not):

    python scripts/verify_tokenizer_alignment.py \\
        --actor princeton-nlp/Sheared-LLaMA-1.3B \\
        --reward PKU-Alignment/beaver-7b-unified-reward \\
        --cost PKU-Alignment/beaver-7b-unified-cost \\
        --require-same
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _hub_file(name_or_path: str, filename: str) -> Path:
    local = Path(name_or_path)
    if local.is_dir():
        path = local / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=name_or_path, filename=filename))


def _load_tokenizer_config(name_or_path: str) -> dict:
    path = _hub_file(name_or_path, 'tokenizer_config.json')
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def _raw_vocab(name_or_path: str) -> dict[str, int]:
    print(f'  resolve tokenizer files for {name_or_path!r} ...', flush=True)
    try:
        json_path = _hub_file(name_or_path, 'tokenizer.json')
    except Exception as exc:  # noqa: BLE001
        print(f'  tokenizer.json missing ({exc}); trying tokenizer.model', flush=True)
        model_path = _hub_file(name_or_path, 'tokenizer.model')
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        return {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}

    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(json_path)).get_vocab()


def _effective_vocab(name_or_path: str) -> tuple[dict[str, int], bool]:
    """Return vocab after Safe-RLHF-style pad add, plus whether pad was added."""
    vocab = dict(_raw_vocab(name_or_path))
    cfg = _load_tokenizer_config(name_or_path)
    pad = cfg.get('pad_token')
    # tokenizer_config may store AddedToken as a dict
    if isinstance(pad, dict):
        pad = pad.get('content')
    added_pad = False
    if pad is None and '<pad>' not in vocab:
        vocab['<pad>'] = len(vocab)
        added_pad = True
    return vocab, added_pad


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
    actor_vocab, actor_added = _effective_vocab(args.actor)
    print(
        f'  size={len(actor_vocab)}  saferlhf_added_<pad>={actor_added}',
        flush=True,
    )

    print(f'Loading reward vocab: {args.reward}', flush=True)
    reward_vocab, reward_added = _effective_vocab(args.reward)
    print(
        f'  size={len(reward_vocab)}  saferlhf_added_<pad>={reward_added}',
        flush=True,
    )

    same_reward = actor_vocab == reward_vocab
    print(f'  actor ↔ reward  same={same_reward}', flush=True)
    if same_reward:
        print('  -> reward scoring will NOT call batch_retokenize', flush=True)
    else:
        print('  -> reward scoring WILL decode/re-encode via batch_retokenize', flush=True)

    same_cost = True
    if args.cost:
        print(f'Loading cost vocab:   {args.cost}', flush=True)
        cost_vocab, cost_added = _effective_vocab(args.cost)
        print(
            f'  size={len(cost_vocab)}  saferlhf_added_<pad>={cost_added}',
            flush=True,
        )
        same_cost = actor_vocab == cost_vocab
        print(f'  actor ↔ cost    same={same_cost}', flush=True)
        if same_cost:
            print('  -> cost scoring will NOT call batch_retokenize', flush=True)
        else:
            print('  -> cost scoring WILL decode/re-encode via batch_retokenize', flush=True)

    all_same = same_reward and same_cost
    if args.require_same and not all_same:
        print(
            '\nERROR: --require-same set but tokenizer vocab(s) differ after '
            'Safe-RLHF pad handling. TinyLlama fails here (pad=</s>, no <pad> add). '
            'Use a Llama-2-vocab policy with pad_token unset '
            '(e.g. princeton-nlp/Sheared-LLaMA-1.3B) or alpaca-7b-reproduced.',
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
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check whether actor and Beaver scorer tokenizers match (no batch_retokenize).

Usage — Llama track (must match; fail the job if not):

    python scripts/verify_tokenizer_alignment.py \\
        --actor TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --reward PKU-Alignment/beaver-7b-unified-reward \\
        --cost PKU-Alignment/beaver-7b-unified-cost \\
        --require-same

Usage — Qwen track (retokenize expected; still exit 0):

    python scripts/verify_tokenizer_alignment.py \\
        --actor Qwen/Qwen2.5-1.5B-Instruct \\
        --reward PKU-Alignment/beaver-7b-unified-reward \\
        --cost PKU-Alignment/beaver-7b-unified-cost
"""

from __future__ import annotations

import argparse
import sys

from transformers import AutoTokenizer

from safe_rlhf.utils import is_same_tokenizer


def _load_tok(name_or_path: str):
    return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=False)


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

    print(f'Loading actor tokenizer:  {args.actor}')
    actor_tok = _load_tok(args.actor)
    print(f'Loading reward tokenizer: {args.reward}')
    reward_tok = _load_tok(args.reward)

    same_reward = is_same_tokenizer(actor_tok, reward_tok)
    print(f'  actor ↔ reward  same={same_reward}')
    if same_reward:
        print('  -> reward scoring will NOT call batch_retokenize')
    else:
        print('  -> reward scoring WILL decode/re-encode via batch_retokenize')

    same_cost = True
    if args.cost:
        print(f'Loading cost tokenizer:   {args.cost}')
        cost_tok = _load_tok(args.cost)
        same_cost = is_same_tokenizer(actor_tok, cost_tok)
        print(f'  actor ↔ cost    same={same_cost}')
        if same_cost:
            print('  -> cost scoring will NOT call batch_retokenize')
        else:
            print('  -> cost scoring WILL decode/re-encode via batch_retokenize')

    all_same = same_reward and same_cost
    if args.require_same and not all_same:
        print(
            '\nERROR: --require-same set but tokenizer(s) differ. '
            'Pick a Llama-2-vocab policy (or fix scorers) before training this track.',
            file=sys.stderr,
        )
        return 1

    if all_same:
        print('\nOK: scorers share the actor tokenizer (no double-tokenizer path).')
    else:
        print(
            '\nOK: mismatch noted — Qwen-style track; retokenize will run '
            '(not a failure unless --require-same).',
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

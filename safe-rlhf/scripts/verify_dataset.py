"""Stage 2 gate: confirm PKU-SafeRLHF prompts render correctly for a Qwen tokenizer.

We deliberately keep upstream's Alpaca-style template from `configs/constants.py`
rather than switching to Qwen's ChatML. The reason is the reward path: `post_rollout()`
re-tokenizes for the reward model with `skip_special_tokens=True`, which would strip
ChatML's `<|im_start|>` / `<|im_end|>` markers entirely and leave Beaver-7B scoring
text in a format it has never seen. The Alpaca template is plain text, so it survives
that round-trip intact — and it is the format PKU trained the reward model on.

This script checks that assumption holds in practice rather than in theory.

Run inside the `safe-rlhf` conda env:

    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --model Qwen/Qwen2.5-0.5B-Instruct --num 5
"""

from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM

from safe_rlhf.configs.constants import PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER
from safe_rlhf.datasets import PromptOnlyDataset, parse_dataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.utils import batch_retokenize


RESULTS: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = '') -> None:
    RESULTS.append((label, ok))
    print(f'  [{"PASS" if ok else "FAIL"}] {label}' + (f' — {detail}' if detail else ''))


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify dataset + prompt template.')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--dataset', type=str, default='PKU-SafeRLHF/train')
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument('--reward_model', type=str, default=None,
                        help='Optional: check the re-tokenization round-trip against this '
                             'reward model tokenizer (e.g. a LLaMA-family checkpoint).')
    args = parser.parse_args()

    print(f'Template in use (configs/constants.py):')
    print(f'  PROMPT_BEGIN     = {PROMPT_BEGIN!r}')
    print(f'  PROMPT_USER      = {PROMPT_USER!r}')
    print(f'  PROMPT_ASSISTANT = {PROMPT_ASSISTANT!r}')

    print(f'\nLoading tokenizer from {args.model} ...')
    _, tokenizer = load_pretrained_models(
        args.model,
        model_max_length=512,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
    )

    print(f'Loading dataset {args.dataset} ...')
    dataset = PromptOnlyDataset([parse_dataset(args.dataset)], tokenizer=tokenizer)
    check('dataset is non-empty', len(dataset) > 0, f'{len(dataset):,} prompts')

    print(f'\n--- first {args.num} decoded prompts ---')
    decoded_samples = []
    for i in range(min(args.num, len(dataset))):
        input_ids = dataset[i]['input_ids']
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        decoded_samples.append(text)
        print(f'\n[{i}] {text!r}')

    print('\n--- template integrity ---')
    first = decoded_samples[0]
    check('BEGINNING-OF-CONVERSATION marker present', PROMPT_BEGIN.strip() in first)
    check('USER marker present', 'USER:' in first)
    check('ASSISTANT marker present', PROMPT_ASSISTANT in first)
    check(
        'prompt ends at the assistant turn',
        first.rstrip().endswith(PROMPT_ASSISTANT),
        repr(first[-30:]),
    )
    check('exactly one USER turn', first.count('USER:') == 1, f'{first.count("USER:")}')
    check(
        'exactly one ASSISTANT turn',
        first.count(PROMPT_ASSISTANT) == 1,
        f'{first.count(PROMPT_ASSISTANT)}',
    )

    bos = tokenizer.bos_token
    if bos:
        check(f'no doubled BOS ({bos!r})', first.count(bos) <= 1, f'{first.count(bos)}')
    else:
        print('  [skip] tokenizer has no BOS token (normal for Qwen)')

    # The whole reason we kept the Alpaca template: the markers must survive a
    # decode with skip_special_tokens=True, which is what the reward model sees.
    print('\n--- reward-model round-trip (skip_special_tokens=True) ---')
    stripped = tokenizer.decode(dataset[0]['input_ids'], skip_special_tokens=True)
    print(f'  {stripped!r}')
    check('USER marker survives special-token stripping', 'USER:' in stripped)
    check('ASSISTANT marker survives special-token stripping', PROMPT_ASSISTANT in stripped)

    if args.reward_model is not None:
        print(f'\n--- batch_retokenize into {args.reward_model} ---')
        _, reward_tokenizer = load_pretrained_models(
            args.reward_model,
            model_max_length=512,
            padding_side='right',
            auto_model_type=AutoModelForCausalLM,
        )
        batch = dataset.get_collator()([dataset[0]])
        out = batch_retokenize(
            batch['input_ids'],
            src_tokenizer=tokenizer,
            dest_tokenizer=reward_tokenizer,
            skip_special_tokens=True,
            device='cpu',
        )
        round_tripped = reward_tokenizer.decode(out['input_ids'][0], skip_special_tokens=True)
        print(f'  {round_tripped!r}')
        check('USER marker survives the round-trip', 'USER:' in round_tripped)
        check('ASSISTANT marker survives the round-trip', PROMPT_ASSISTANT in round_tripped)

    failed = [label for label, ok in RESULTS if not ok]
    print(f'\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed')
    if failed:
        print('FAILED:')
        for label in failed:
            print(f'  - {label}')
        raise SystemExit(1)
    print('All checks passed.')


if __name__ == '__main__':
    main()

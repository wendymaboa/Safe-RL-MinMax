"""Verify the LoRA plumbing added to `load_pretrained_models()`.

This exercises the real code path (not a synthetic peft example) on a small
Qwen2 checkpoint, and checks the things that fail *silently* in a full run:

  1. The actor is wrapped as a PeftModelForCausalLM and can still generate.
  2. The critic's randomly-initialised `score_head` survives as trainable.
  3. Omitting `lora_config` leaves the model completely untouched.
  4. `disable_adapter()` really does restore base-model behaviour.

Run inside the `safe-rlhf` conda env:

    python scripts/verify_lora.py
    python scripts/verify_lora.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from safe_rlhf.trainers.rl_trainer import AdapterDisabledReference


RESULTS: list[tuple[str, bool]] = []


class _StubEngine:
    """Minimal stand-in for a DeepSpeedEngine: callable, and exposes `.module`.

    `AdapterDisabledReference` only needs those two things, so the proxy can be
    tested without initialising DeepSpeed or touching a GPU.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.module = model

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def check(label: str, ok: bool, detail: str = '') -> None:
    """Record and print a single assertion."""
    RESULTS.append((label, ok))
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {label}' + (f' — {detail}' if detail else ''))


def param_counts(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def make_config(**overrides) -> LoraConfig:
    kwargs = {
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'target_modules': None,  # let PEFT use its per-architecture mapping
    }
    kwargs.update(overrides)
    return LoraConfig(**kwargs)


def check_actor(model_name: str, trust_remote_code: bool) -> None:
    print('\n[1] Actor — causal LM with adapters')
    actor, tokenizer = load_pretrained_models(
        model_name,
        model_max_length=512,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=trust_remote_code,
        lora_config=make_config(task_type=TaskType.CAUSAL_LM),
    )

    check(
        'wrapped as PeftModelForCausalLM',
        type(actor).__name__ == 'PeftModelForCausalLM',
        type(actor).__name__,
    )

    trainable, total = param_counts(actor)
    pct = 100.0 * trainable / total
    check('only a small fraction is trainable', pct < 5.0, f'{trainable:,}/{total:,} = {pct:.3f}%')
    check('some parameters are trainable', trainable > 0)

    adapted = sorted({
        name.split('.lora_A')[0].rsplit('.', 1)[-1]
        for name, _ in actor.named_parameters()
        if '.lora_A' in name
    })
    check('adapters were injected', len(adapted) > 0, f'target modules: {adapted}')

    # Generation is the reason the actor needs task_type at all.
    prompt = tokenizer('Hello', return_tensors='pt')
    with torch.no_grad():
        out = actor.generate(**prompt, max_new_tokens=4, do_sample=False)
    check('generate() works through the PEFT wrapper', out.shape[-1] > prompt['input_ids'].shape[-1])

    check_disable_adapter(actor, prompt)
    check_reference_proxy(actor, prompt)
    del actor


def check_reference_proxy(actor, prompt) -> None:
    """The proxy must behave exactly like an explicitly adapter-disabled forward.

    Runs after `check_disable_adapter` has perturbed the B matrices, so the adapters
    are actually active and the comparison is meaningful.
    """
    print('\n[1c] AdapterDisabledReference proxy')
    reference = AdapterDisabledReference(_StubEngine(actor))

    with torch.no_grad():
        via_proxy = reference(**prompt).logits
        with actor.disable_adapter():
            direct = actor(**prompt).logits
        with_adapters = actor(**prompt).logits

    check(
        'proxy output matches an explicit disable_adapter() forward',
        torch.allclose(via_proxy, direct, atol=1e-6),
    )
    check(
        'proxy output differs from the adapter-enabled actor',
        not torch.allclose(via_proxy, with_adapters, atol=1e-5),
    )
    check('adapters are re-enabled after the proxy call', not _adapters_disabled(actor))
    check('eval() is a no-op returning self', reference.eval() is reference)


def _adapters_disabled(model: torch.nn.Module) -> bool:
    """True if any LoRA layer is still in the disabled state."""
    return any(
        getattr(module, 'disable_adapters', False)
        for module in model.modules()
    )


def check_disable_adapter(actor, prompt) -> None:
    """LoRA initialises B to zeros, so an untrained adapter is a no-op.

    To prove `disable_adapter()` does something we first perturb B, which makes
    the adapter active, then confirm disabling it restores the original output.
    """
    with torch.no_grad():
        base_logits = actor(**prompt).logits.clone()

        perturbed = 0
        for name, param in actor.named_parameters():
            if '.lora_B' in name:
                param.add_(torch.randn_like(param) * 0.05)
                perturbed += 1

        active_logits = actor(**prompt).logits.clone()
        with actor.disable_adapter():
            disabled_logits = actor(**prompt).logits.clone()

    check('lora_B matrices found to perturb', perturbed > 0, f'{perturbed} tensors')
    check(
        'perturbed adapter changes the output',
        not torch.allclose(base_logits, active_logits, atol=1e-5),
    )
    check(
        'disable_adapter() restores base output',
        torch.allclose(base_logits, disabled_logits, atol=1e-5),
    )


def check_critic(model_name: str, trust_remote_code: bool) -> None:
    print('\n[2] Critic — score model, score_head must stay trainable')
    critic, _ = load_pretrained_models(
        model_name,
        model_max_length=512,
        padding_side='left',
        auto_model_type=AutoModelForScore,
        trust_remote_code=trust_remote_code,
        auto_model_kwargs={'score_type': 'critic', 'do_normalize': False},
        lora_config=make_config(modules_to_save=['score_head']),
    )

    score_params = [(n, p) for n, p in critic.named_parameters() if 'score_head' in n]
    trainable_score = [n for n, p in score_params if p.requires_grad]

    check('score_head parameters exist', len(score_params) > 0, f'{len(score_params)} tensors')
    check(
        'score_head is TRAINABLE (modules_to_save worked)',
        len(trainable_score) > 0,
        f'{len(trainable_score)} trainable',
    )

    trainable, total = param_counts(critic)
    check('critic has trainable params', trainable > 0, f'{trainable:,}/{total:,}')

    with torch.no_grad():
        out = critic(torch.tensor([[1, 2, 3, 4]]), attention_mask=torch.ones(1, 4, dtype=torch.bool))
    check('critic forward returns scores', out.scores is not None, f'shape {tuple(out.scores.shape)}')
    del critic


def check_no_lora(model_name: str, trust_remote_code: bool) -> None:
    print('\n[3] No LoRA — upstream behaviour must be untouched')
    model, _ = load_pretrained_models(
        model_name,
        model_max_length=512,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=trust_remote_code,
    )
    trainable, total = param_counts(model)
    check('not a PEFT model', 'Peft' not in type(model).__name__, type(model).__name__)
    check('all parameters trainable', trainable == total, f'{trainable:,}/{total:,}')
    del model


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify LoRA plumbing.')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--trust_remote_code', action='store_true')
    args = parser.parse_args()

    print(f'Verifying LoRA plumbing with: {args.model}')
    check_actor(args.model, args.trust_remote_code)
    check_critic(args.model, args.trust_remote_code)
    check_no_lora(args.model, args.trust_remote_code)

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

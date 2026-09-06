# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constant variables."""

from __future__ import annotations

import os


__all__ = [
    'IGNORE_INDEX',
    'DEFAULT_BOS_TOKEN',
    'DEFAULT_EOS_TOKEN',
    'DEFAULT_PAD_TOKEN',
    'DEFAULT_UNK_TOKEN',
    'PROMPT_BEGIN',
    'PROMPT_USER',
    'PROMPT_ASSISTANT',
    'PROMPT_INPUT',
    'PROMPT_DICT',
    'ADAM_BETAS',
]


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'

# Prompt template, selected with SAFE_RLHF_PROMPT_TEMPLATE=alpaca|chatml.
# Default is 'alpaca' — upstream's format, unchanged.
#
# Why 'chatml' exists: Qwen2.5-Instruct was trained on ChatML, where a turn ends with
# <|im_end|>, which is also the tokenizer's eos_token. Under the Alpaca-style template
# Qwen is in raw-completion mode with no marker it recognises as end-of-turn, so it
# NEVER stops: every generation runs to max_length, hallucinating extra USER:/ASSISTANT:
# turns and decaying into noise before being truncated mid-word. Verified on the
# untrained base model, so it is a property of the template, not of training.
#
# That caps reward and, worse, hands the reward model a coherent opening followed by a
# garbage tail — so the training signal is dominated by noise. Stage 5 run A drifted
# (KL to +6) and lost 2.3 reward units optimising against exactly that.
#
# The cost of ChatML is that post_rollout() re-tokenizes with skip_special_tokens=True,
# which strips the <|im_start|>/<|im_end|> markers, leaving the reward model bare
# 'system'/'user'/'assistant' words. That is the lesser evil: a preference model can
# read those as text, but cannot meaningfully score a sentence that stops mid-word.
_PROMPT_TEMPLATE: str = os.getenv('SAFE_RLHF_PROMPT_TEMPLATE', 'alpaca').lower()

if _PROMPT_TEMPLATE == 'chatml':
    PROMPT_BEGIN: str = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
    PROMPT_USER: str = '<|im_start|>user\n{input}<|im_end|>\n'
    PROMPT_ASSISTANT: str = '<|im_start|>assistant\n'
elif _PROMPT_TEMPLATE == 'alpaca':
    PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
    PROMPT_USER: str = 'USER: {input} '
    PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
else:
    raise ValueError(
        f'Unknown SAFE_RLHF_PROMPT_TEMPLATE={_PROMPT_TEMPLATE!r}. Expected "alpaca" or "chatml".',
    )

PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

PROMPT_DICT: dict[str, str] = {
    'prompt_begin': PROMPT_BEGIN,
    'prompt_user': PROMPT_USER,
    'prompt_assistant': PROMPT_ASSISTANT,
    'prompt_input': PROMPT_INPUT,
}

ADAM_BETAS: tuple[float, float] = (0.9, 0.95)

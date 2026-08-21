# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MULTI_TURN_DIR = Path(__file__).parents[3] / "benchmarks" / "multi_turn"
sys.path.insert(0, str(MULTI_TURN_DIR))

from bench_dataset import generate_conversations, parse_input_json_file  # noqa: E402


class WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def __call__(self, text: str, add_special_tokens: bool = True):
        return SimpleNamespace(input_ids=self.encode(text))


def make_config(text_file: Path) -> dict:
    constant = lambda value: {"distribution": "constant", "value": value}
    return {
        "filetype": "generate_conversations",
        "num_conversations": 4,
        "num_prefix_groups": 2,
        "text_files": [str(text_file)],
        "prompt_input": {
            "num_turns": constant(4),
            "common_prefix_num_tokens": constant(6),
            "suffix_num_tokens": constant(4),
            "num_tokens": constant(20),
        },
        "prompt_output": {"num_tokens": constant(2)},
    }


def test_generate_conversations_shares_prefix_within_groups(tmp_path: Path) -> None:
    text_file = tmp_path / "source.txt"
    text_file.write_text(" ".join(f"token{i}" for i in range(500)))
    tokenizer = WhitespaceTokenizer()

    conversations = generate_conversations(
        parse_input_json_file(make_config(text_file)), tokenizer
    )

    first_prompts = [
        conversations[f"CONV_ID_{conv_id}"][0]["content"] for conv_id in range(4)
    ]
    assert first_prompts[0].startswith("token0 token1 token2 token3.")
    assert first_prompts[2].startswith("token0 token1 token2 token3.")
    assert first_prompts[1].startswith("token6 token7 token8 token9.")
    assert first_prompts[3].startswith("token6 token7 token8 token9.")


def test_parse_rejects_more_prefix_groups_than_conversations(tmp_path: Path) -> None:
    config = make_config(tmp_path / "source.txt")
    config["num_prefix_groups"] = config["num_conversations"] + 1

    with pytest.raises(AssertionError, match="no larger than num_conversations"):
        parse_input_json_file(config)

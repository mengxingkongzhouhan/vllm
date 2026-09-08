# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from vllm.benchmarks.datasets import get_samples

pytestmark = pytest.mark.skip_global_cleanup


class _Tokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {chr(ord("a") + i): i + 1 for i in range(26)}

    @property
    def all_special_ids(self) -> list[int]:
        return []

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return " ".join(str(i) for i in ids)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _args_for_timed_trace(
    dataset_path: Path,
    *,
    output_len: int | None = None,
    timed_trace_output_len: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="timed_trace",
        dataset_path=str(dataset_path),
        disable_shuffle=True,
        seed=0,
        num_prompts=2,
        request_id_prefix="req-",
        timed_trace_chunk_hash_size=16,
        timed_trace_sec_multiplier=1,
        timed_trace_label_timestamp="timestamp",
        timed_trace_label_input_length="input_length",
        timed_trace_label_output_length="output_length",
        timed_trace_label_hash_ids="hash_ids",
        timed_trace_output_len=timed_trace_output_len,
        output_len=output_len,
    )


def _trace_rows() -> list[dict[str, Any]]:
    return [
        {
            "timestamp": 1.0,
            "input_length": 8,
            "output_length": 128,
            "hash_ids": [1],
        },
        {
            "timestamp": 2.0,
            "input_length": 16,
            "output_length": 64,
            "hash_ids": [2],
        },
    ]


@pytest.mark.benchmark
def test_timed_trace_uses_output_length_from_trace(tmp_path: Path) -> None:
    jsonl = tmp_path / "trace.jsonl"
    _write_jsonl(jsonl, _trace_rows())

    samples = get_samples(_args_for_timed_trace(jsonl), _Tokenizer())

    assert [s.expected_output_len for s in samples] == [128, 64]


@pytest.mark.benchmark
def test_timed_trace_output_len_overrides_trace(tmp_path: Path) -> None:
    jsonl = tmp_path / "trace.jsonl"
    _write_jsonl(jsonl, _trace_rows())

    samples = get_samples(
        _args_for_timed_trace(jsonl, timed_trace_output_len=1), _Tokenizer()
    )

    assert [s.expected_output_len for s in samples] == [1, 1]
    assert [s.timestamp for s in samples] == [1.0, 2.0]


@pytest.mark.benchmark
def test_timed_trace_general_output_len_overrides_trace(tmp_path: Path) -> None:
    jsonl = tmp_path / "trace.jsonl"
    _write_jsonl(jsonl, _trace_rows())

    samples = get_samples(_args_for_timed_trace(jsonl, output_len=1), _Tokenizer())

    assert [s.expected_output_len for s in samples] == [1, 1]

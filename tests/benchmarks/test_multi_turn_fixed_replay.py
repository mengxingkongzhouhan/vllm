# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

MULTI_TURN_BENCHMARK_DIR = Path(__file__).parents[2] / "benchmarks" / "multi_turn"
sys.path.insert(0, str(MULTI_TURN_BENCHMARK_DIR))
benchmark = importlib.import_module("benchmark_serving_multi_turn")


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(text))


def _request_args():
    return benchmark.RequestArgs(
        chat_url="http://localhost/v1/chat/completions",
        model="model",
        stream=True,
        limit_min_tokens=-1,
        limit_max_tokens=-1,
        timeout_sec=1,
        send_conversation_id=False,
        headers={},
    )


def test_fixed_dataset_replay_preserves_history_for_next_request(monkeypatch):
    captured_messages = []

    async def fake_send_request(session, messages, *args, **kwargs):
        captured_messages.append(copy.deepcopy(messages))
        return benchmark.ServerResponse(
            valid=True,
            ttft_ms=1.0,
            tpot_ms=1.0,
            latency_ms=2.0,
            start_time_ms=0.0,
            first_chunk="x",
            content="generated answer",
            num_chunks=1,
        )

    monkeypatch.setattr(benchmark, "send_request", fake_send_request)
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "fixed answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "another fixed answer"},
    ]

    async def run_turns():
        await benchmark.send_turn(
            None,
            0,
            "conversation",
            messages,
            1,
            _Tokenizer(),
            _request_args(),
            False,
            False,
            True,
        )
        await benchmark.send_turn(
            None,
            0,
            "conversation",
            messages,
            3,
            _Tokenizer(),
            _request_args(),
            False,
            False,
            True,
        )

    asyncio.run(run_turns())

    assert messages[1]["content"] == "fixed answer"
    assert captured_messages[1][1]["content"] == "fixed answer"


def test_default_replay_uses_generated_answer(monkeypatch):
    async def fake_send_request(session, messages, *args, **kwargs):
        return benchmark.ServerResponse(
            valid=True,
            ttft_ms=1.0,
            tpot_ms=1.0,
            latency_ms=2.0,
            start_time_ms=0.0,
            first_chunk="x",
            content="generated answer",
            num_chunks=1,
        )

    monkeypatch.setattr(benchmark, "send_request", fake_send_request)
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "dataset answer"},
    ]

    asyncio.run(
        benchmark.send_turn(
            None,
            0,
            "conversation",
            messages,
            1,
            _Tokenizer(),
            _request_args(),
            False,
            False,
            False,
        )
    )

    assert messages[1]["content"] == "generated answer"

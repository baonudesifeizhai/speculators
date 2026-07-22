import fcntl
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from speculators.data_generation import vllm_client
from speculators.data_generation.vllm_client import (
    InvalidResponseError,
    extract_output,
    generate_hidden_states,
    wait_for_lock,
    wait_for_lock_async,
)


def _chat_response(prompt_token_ids: list[int]) -> ChatCompletion:
    return ChatCompletion.model_construct(
        prompt_token_ids=prompt_token_ids,
        kv_transfer_params={"hidden_states_path": "/tmp/hidden.safetensors"},
    )


def test_extract_output_keeps_text_token_validation_strict():
    with pytest.raises(InvalidResponseError, match="Prompt token IDs mismatch"):
        extract_output(_chat_response([1, 2, 3]), [1, 2])


def test_extract_output_allows_multimodal_placeholder_expansion():
    path = extract_output(
        _chat_response([1, 90, 90, 90, 2]),
        [1, 90, 2],
        allow_token_id_mismatch=True,
    )

    assert path == "/tmp/hidden.safetensors"


def test_generate_hidden_states_replays_completed_final_message():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _chat_response([1, 2])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    item = {
        "input_ids": [1, 2],
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    }

    assert generate_hidden_states(client, "model", item) == "/tmp/hidden.safetensors"
    assert captured["extra_body"]["add_generation_prompt"] is False
    assert captured["extra_body"]["continue_final_message"] is False


def test_wait_for_lock_uses_shared_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / "hidden.safetensors.lock"
    lock_path.touch()
    operations = []
    monkeypatch.setattr(
        vllm_client.fcntl,
        "flock",
        lambda _fd, operation: operations.append(operation),
    )

    wait_for_lock(lock_path)

    assert operations == [fcntl.LOCK_SH | fcntl.LOCK_NB]
    assert not lock_path.exists()


@pytest.mark.asyncio
async def test_wait_for_lock_async_uses_shared_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / "hidden.safetensors.lock"
    lock_path.touch()
    operations = []
    monkeypatch.setattr(
        vllm_client.fcntl,
        "flock",
        lambda _fd, operation: operations.append(operation),
    )

    await wait_for_lock_async(lock_path)

    assert operations == [fcntl.LOCK_SH | fcntl.LOCK_NB]
    assert not lock_path.exists()

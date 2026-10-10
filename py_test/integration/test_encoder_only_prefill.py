import urllib.parse

import pytest
import requests


def _bootstrap_port(url: str) -> int:
    port = urllib.parse.urlparse(url).port
    assert port is not None
    return port


def _start_router(router_manager, prefill_url: str, decode_url: str):
    return router_manager.start_router(
        policy="round_robin",
        vllm_pd_disaggregation=True,
        prefill_urls=[(prefill_url, _bootstrap_port(prefill_url))],
        decode_urls=[decode_url],
        extra={
            "encoder_only_prefill": True,
            "kv_connector": "mooncake",
            "worker_startup_check_interval": 1,
        },
    )


@pytest.mark.integration
def test_encoder_only_prefill_builds_matching_mooncake_requests(
    router_manager, mock_workers
):
    _, [prefill_url], [prefill_id] = mock_workers(n=1)
    _, [decode_url], _ = mock_workers(n=1)
    router = _start_router(router_manager, prefill_url, decode_url)

    response = requests.post(
        f"{router.url}/v1/completions",
        json={
            "model": "mock",
            "prompt": "hello",
            "max_tokens": 8,
            "stream": False,
        },
        timeout=10,
    )

    assert response.status_code == 200
    decode_request = response.json()["echo"]
    prefill_record = requests.get(f"{prefill_url}/last_request", timeout=5).json()
    prefill_request = prefill_record["json"]
    prefill_headers = prefill_record["headers"]

    prefill_kv = prefill_request["kv_transfer_params"]
    decode_kv = decode_request["kv_transfer_params"]
    assert prefill_request["stream"] is False
    assert prefill_request["max_tokens"] == 1
    assert prefill_kv["cache_only"] is True
    assert prefill_kv["do_remote_decode"] is True
    assert prefill_kv["do_remote_prefill"] is False
    assert prefill_headers["x-data-parallel-rank"] == "0"

    assert decode_request["stream"] is False
    assert decode_request["max_tokens"] == 8
    assert decode_kv["do_remote_decode"] is False
    assert decode_kv["do_remote_prefill"] is True
    assert decode_kv["remote_bootstrap_addr"] == prefill_url
    assert decode_kv["remote_engine_id"] == f"{prefill_id}-engine-0"
    assert decode_kv["transfer_id"] == prefill_kv["transfer_id"]


@pytest.mark.integration
def test_encoder_only_prefill_preserves_decode_stream(router_manager, mock_workers):
    _, [prefill_url], _ = mock_workers(n=1)
    _, [decode_url], _ = mock_workers(n=1, args=["--stream"])
    router = _start_router(router_manager, prefill_url, decode_url)

    with requests.post(
        f"{router.url}/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "stream": True,
        },
        stream=True,
        timeout=10,
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    assert len(lines) == 3
    assert lines[-1] == b"data: [DONE]"
    decode_record = requests.get(f"{decode_url}/last_request", timeout=5).json()
    assert decode_record["json"]["stream"] is True


@pytest.mark.integration
def test_encoder_only_prefill_failure_is_not_hidden(router_manager, mock_workers):
    _, [prefill_url], _ = mock_workers(n=1, args=["--status-code", "500"])
    _, [decode_url], _ = mock_workers(n=1)
    router = _start_router(router_manager, prefill_url, decode_url)

    response = requests.post(
        f"{router.url}/v1/completions",
        json={"model": "mock", "prompt": "hello", "max_tokens": 1},
        timeout=10,
    )

    assert response.status_code == 500
    assert "Prefill request failed" in response.text


@pytest.mark.integration
def test_encoder_only_decode_failure_status_is_preserved(router_manager, mock_workers):
    _, [prefill_url], _ = mock_workers(n=1)
    _, [decode_url], _ = mock_workers(n=1, args=["--status-code", "503"])
    router = _start_router(router_manager, prefill_url, decode_url)

    response = requests.post(
        f"{router.url}/v1/completions",
        json={"model": "mock", "prompt": "hello", "max_tokens": 1},
        timeout=10,
    )

    assert response.status_code == 503

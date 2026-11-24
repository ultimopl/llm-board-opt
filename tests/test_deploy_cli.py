import pytest
from board_opt.deploy import generate_serve_script, check_active_gpu_procs
from board_opt.tuner import TunedRuntimeConfig
from board_opt.builder import ToolchainPaths
from board_opt.models import EvalMetrics


def test_generate_serve_script_contains_tuned_flags(tmp_path):
    rt = TunedRuntimeConfig(
        "q8_0", "q4_0", 256, 32768, True, 0, 40,
        {"GGML_VK_MAX_NODES_PER_SUBMIT": "1024"},
        EvalMetrics(330.0, 43.0, 6.38, 12, 12, 11000),
    )
    tc = ToolchainPaths("/bin/lb", "/bin/lp", "/bin/lq", "/bin/llama-server", "/bin/gc", "/bin/rt")
    script = generate_serve_script("/models/opt.gguf", rt, tc, "0.0.0.0", 8080, str(tmp_path / "serve.sh"))
    text = open(script, encoding="utf-8").read()
    assert "GGML_VK_MAX_NODES_PER_SUBMIT=1024" in text
    assert "-ctk q8_0 -ctv q4_0" in text
    assert "--cache-ram 0" in text
    assert "--jinja" in text


def test_single_gpu_guard_aborts_on_unmanaged_proc():
    with pytest.raises(RuntimeError, match="Active GPU process"):
        check_active_gpu_procs(["1234 /usr/bin/llama-bench -m foo.gguf"])

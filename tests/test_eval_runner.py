import json
from board_opt.eval_runner import build_calibration_file, load_eval_tasks, grade_response


def test_build_calibration_combines_langs_and_code(tmp_path):
    out = build_calibration_file(["en", "pt"], ["python"], str(tmp_path / "cal.txt"))
    txt = open(out, encoding="utf-8").read()
    assert len(txt) > 3000
    assert "def " in txt
    assert "português" in txt


def test_grade_python_timeout_does_not_hang():
    task = {
        "id": "inf",
        "domain": "python",
        "grader": "python",
        "test_code": "assert solve() == 1",
    }
    ok, reason = grade_response(
        task, "```python\ndef solve():\n    while True: pass\n```", timeout_s=1
    )
    assert ok is False
    assert "timeout" in reason.lower()


def test_load_custom_eval_file(tmp_path):
    p = tmp_path / "custom.json"
    p.write_text(
        json.dumps([
            {"id": "c1", "domain": "custom", "prompt": "2+2?", "grader": "exact", "expected": "4"}
        ])
    )
    tasks = load_eval_tasks(["reasoning"], ["en", "pt"], eval_file=str(p), quick=True)
    assert any(t["id"] == "c1" for t in tasks)
    assert any(t["domain"] == "pt" for t in tasks)

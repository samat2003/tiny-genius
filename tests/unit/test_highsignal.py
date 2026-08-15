"""High-signal rubric: CP stdin/stdout is KEEP; utility I/O is REJECT."""

from __future__ import annotations

from tiny_genius.data.highsignal import load_highsignal_config, score_document


def test_cp_input_print_is_kept() -> None:
    cfg = load_highsignal_config()
    doc = {
        "doc_id": "cp1",
        "source_id": "codecontests_plus",
        "domain": "python",
        "problem_id": "1_A",
        "text": (
            "Read two integers and print their sum.\n\n"
            "a, b = map(int, input().split())\n"
            "print(a + b)\n"
        ),
        "solution": "a, b = map(int, input().split())\nprint(a + b)\n",
        "language_tag": "python",
    }
    verdict = score_document(doc, cfg)
    assert verdict["accept"] is True
    assert "cp_stdin_stdout_keep" in verdict["reasons"]
    assert "utility_io" not in verdict["reasons"]


def test_file_io_tutorial_is_rejected() -> None:
    cfg = load_highsignal_config()
    doc = {
        "doc_id": "io1",
        "source_id": "pythonbook",
        "domain": "python",
        "text": (
            "This chapter shows how to use pathlib.\n"
            "from pathlib import Path\n"
            "Path('out.txt').write_text('hi')\n"
        ),
        "language_tag": "python",
    }
    verdict = score_document(doc, cfg)
    assert verdict["accept"] is False
    assert "utility_io" in verdict["reasons"] or "library_tutorial" in verdict["reasons"]


def test_cpp_is_rejected() -> None:
    cfg = load_highsignal_config()
    doc = {
        "doc_id": "cpp1",
        "source_id": "apps",
        "domain": "python",
        "text": "#include <bits/stdc++.h>\nint main() { return 0; }\n",
    }
    assert score_document(doc, cfg)["accept"] is False

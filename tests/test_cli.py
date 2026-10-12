"""Tests for the GeminiScore CLI."""

import csv
import json
import os
import pytest
from gemini_score.cli import main
from gemini_score.similarity import _pseudo_embed


def make_mock_client():
    def mock_embed(texts, is_embed=True):
        return [_pseudo_embed(t) for t in texts]

    return mock_embed


def test_cli_version(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_text_files(tmp_path, monkeypatch, capsys):
    pred_path = tmp_path / "preds.txt"
    ref_path = tmp_path / "refs.txt"

    pred_path.write_text("The cat is on the mat.\n", encoding="utf-8")
    ref_path.write_text("The cat is on the mat.\n", encoding="utf-8")

    from gemini_score import core

    original_init = core.GeminiScore.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["client"] = make_mock_client()
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(core.GeminiScore, "__init__", patched_init)

    ret = main([
        "--pred", str(pred_path),
        "--ref", str(ref_path),
        "--format", "score-only"
    ])
    assert ret == 0
    captured = capsys.readouterr()
    score = float(captured.out.strip())
    assert score >= 99.0


def test_cli_csv_file(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "data.csv"
    out_path = tmp_path / "results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction", "reference"])
        writer.writerow(["The cat is on the mat.", "The cat is on the mat."])

    from gemini_score import core

    original_init = core.GeminiScore.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["client"] = make_mock_client()
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(core.GeminiScore, "__init__", patched_init)

    ret = main([
        "--file", str(csv_path),
        "--output", str(out_path),
        "--format", "json"
    ])
    assert ret == 0
    assert os.path.exists(out_path)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "score" in data
    assert data["score"] >= 99.0


def test_cli_json_file(tmp_path, monkeypatch, capsys):
    json_path = tmp_path / "data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([
            {"pred": "The cat is on the mat.", "ref": "The cat is on the mat."}
        ], f)

    from gemini_score import core

    original_init = core.GeminiScore.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["client"] = make_mock_client()
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(core.GeminiScore, "__init__", patched_init)

    ret = main([
        "--file", str(json_path),
        "--pred-col", "pred",
        "--ref-col", "ref",
        "--format", "score-only"
    ])
    assert ret == 0
    captured = capsys.readouterr()
    score = float(captured.out.strip())
    assert score >= 99.0


def test_cli_missing_args(capsys):
    ret = main([])
    assert ret == 1

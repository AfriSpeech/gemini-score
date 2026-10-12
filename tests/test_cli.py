"""Tests for the GeminiScore CLI."""

import csv
import json
import os
import pytest
from gemini_score.cli import main


def make_mock_client():
    def mock_client(prompt, is_json=False):
        if is_json:
            return '["The cat is on the mat."]'
        return "The cat is on the mat."

    return mock_client


def test_cli_list_languages(capsys):
    ret = main(["--list-languages"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Total African languages supported via afriso:" in captured.out
    assert "Yoruba" in captured.out


def test_cli_info_language(capsys):
    ret = main(["--info-language", "yor"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Language Name : Yoruba" in captured.out
    assert "ISO 639-3     : yor" in captured.out


def test_cli_text_files(tmp_path, monkeypatch, capsys):
    pred_path = tmp_path / "preds.txt"
    ref_path = tmp_path / "refs.txt"

    pred_path.write_text("Ologbo naa wa lori akete.\n", encoding="utf-8")
    ref_path.write_text("The cat is on the mat.\n", encoding="utf-8")

    # Mock GeminiScore to use mock client
    from gemini_score import core

    original_init = core.GeminiScore.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["client"] = make_mock_client()
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(core.GeminiScore, "__init__", patched_init)

    ret = main([
        "--pred", str(pred_path),
        "--ref", str(ref_path),
        "--language", "Yoruba",
        "--format", "score-only"
    ])
    assert ret == 0
    captured = capsys.readouterr()
    score = float(captured.out.strip())
    assert score > 90.0


def test_cli_csv_file(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "data.csv"
    out_path = tmp_path / "results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction", "reference", "language"])
        writer.writerow(["Ologbo naa wa lori akete.", "The cat is on the mat.", "yor"])

    from gemini_score import core

    original_init = core.GeminiScore.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["client"] = make_mock_client()
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(core.GeminiScore, "__init__", patched_init)

    ret = main([
        "--file", str(csv_path),
        "--lang-col", "language",
        "--output", str(out_path),
        "--format", "json"
    ])
    assert ret == 0
    assert os.path.exists(out_path)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "score" in data
    assert data["score"] > 90.0


def test_cli_json_file(tmp_path, monkeypatch, capsys):
    json_path = tmp_path / "data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([
            {"pred": "Ologbo naa wa lori akete.", "ref": "The cat is on the mat."}
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
        "--language", "yor",
        "--format", "score-only"
    ])
    assert ret == 0
    captured = capsys.readouterr()
    score = float(captured.out.strip())
    assert score > 90.0

from __future__ import annotations

import json

from pre.train.cli import train_main


def test_pre_train_cli_outputs_shape_checks(  # type: ignore[no-untyped-def]
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "pre-train",
            "--dataset",
            "nyc_taxi",
            "--model",
            "dummy",
            "--horizon",
            "12",
            "--context-length",
            "72",
        ],
    )
    train_main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["summary"]["shape_checks_passed"] is True
    assert payload["summary"]["batch_shapes"]["train"]["targets"][1] == 12

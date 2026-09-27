import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "prepare_generated_flowtrack_fid_fvd_pairs.py"
SPEC = importlib.util.spec_from_file_location("prepare_generated_flowtrack_fid_fvd_pairs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_front_rows_preserve_split_and_token_order(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"token": "a", "query_view_group": "front"},
        {"token": "a", "query_view_group": "rear"},
        {"token": "b", "query_view_group": "front"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected = MODULE.read_front_rows(manifest, "train")

    assert [row["token"] for row in selected] == ["a", "b"]
    assert {row["dataset_split"] for row in selected} == {"train"}


def test_flat_names_follow_historical_fvd_parser_contract():
    name = MODULE.flat_name(17, "token_value", 9, "CAM_FRONT_RIGHT")

    assert name == "00000017_token_value_f09_CAM_FRONT_RIGHT"


def test_render_source_uses_nonoverlapping_four_frame_window(tmp_path):
    path = MODULE.render_source_path(
        tmp_path, "heldout", "token", 15, "CAM_FRONT"
    )

    assert path == (
        tmp_path
        / "renders"
        / "heldout"
        / "token"
        / "window_12"
        / "frame_15_CAM_FRONT.png"
    )

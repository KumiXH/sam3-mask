from pathlib import Path

from PIL import Image

from sam3_mask.pairing.pairs import build_pairs, read_pair_scale
from sam3_mask.scanner.filesystem import scan_images


def test_scan_images_returns_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "LQ"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "1.png").write_bytes(b"fake")

    results = scan_images(root, [".png"])

    assert [item.relative_path.as_posix() for item in results] == ["a/b/1.png"]


def test_build_pairs_uses_strict_relative_path(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    (lq_root / "x").mkdir(parents=True)
    (hr_root / "x").mkdir(parents=True)
    (hr_root / "y").mkdir(parents=True)
    (lq_root / "x" / "1.png").write_bytes(b"lq")
    (hr_root / "x" / "1.png").write_bytes(b"hr")
    (hr_root / "y" / "1.png").write_bytes(b"wrong")

    pairs, missing_lq, missing_hr = build_pairs(
        scan_images(lq_root, [".png"]),
        scan_images(hr_root, [".png"]),
    )

    assert len(pairs) == 1
    assert pairs[0].relative_path.as_posix() == "x/1.png"
    assert missing_lq == [Path("y/1.png")]
    assert missing_hr == []


def test_build_pairs_is_case_sensitive_on_all_platforms(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    (lq_root / "x").mkdir(parents=True)
    (hr_root / "x").mkdir(parents=True)
    (lq_root / "x" / "Foo.png").write_bytes(b"lq")
    (hr_root / "x" / "foo.png").write_bytes(b"hr")

    pairs, missing_lq, missing_hr = build_pairs(
        scan_images(lq_root, [".png"]),
        scan_images(hr_root, [".png"]),
    )

    assert pairs == []
    assert missing_lq == [Path("x/foo.png")]
    assert missing_hr == [Path("x/Foo.png")]


def test_read_pair_scale_reports_two_x(tmp_path: Path) -> None:
    lq_path = tmp_path / "lq.png"
    hr_path = tmp_path / "hr.png"
    Image.new("RGB", (8, 8), "white").save(lq_path)
    Image.new("RGB", (16, 16), "white").save(hr_path)

    scale_x, scale_y = read_pair_scale(lq_path, hr_path)

    assert scale_x == 2.0
    assert scale_y == 2.0

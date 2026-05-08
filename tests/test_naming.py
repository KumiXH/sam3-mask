from sam3_mask.utils.naming import build_object_stem, normalize_label


def test_build_object_stem_flattens_parent_dirs() -> None:
    stem = build_object_stem("a/b/1.png", "face", 1)

    assert stem == "a_b_1_face_01"


def test_build_object_stem_flattens_windows_style_parent_dirs() -> None:
    stem = build_object_stem(r"a\b\1.png", "face", 1)

    assert stem == "a_b_1_face_01"


def test_build_object_stem_handles_root_image() -> None:
    stem = build_object_stem("1.png", "green plant", 12)

    assert stem == "1_green_plant_12"


def test_normalize_label_keeps_safe_tokens() -> None:
    assert normalize_label(" Green Plant / Leaf ") == "green_plant_leaf"

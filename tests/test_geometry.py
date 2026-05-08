import numpy as np

from sam3_mask.utils.geometry import clamp_box, resize_mask, scale_box


def test_scale_box_maps_lq_to_hr() -> None:
    assert scale_box((1, 2, 3, 4), 2.0, 2.0) == (2, 4, 6, 8)


def test_scale_box_maps_fractional_scale_by_endpoints() -> None:
    assert scale_box((1, 0, 1, 1), 1.5, 1.5) == (2, 0, 1, 2)


def test_clamp_box_stays_in_bounds() -> None:
    assert clamp_box((-1, 3, 10, 8), width=8, height=8) == (0, 3, 8, 5)


def test_resize_mask_preserves_boolean_shape() -> None:
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True

    resized = resize_mask(mask, size=(4, 4))

    assert resized.shape == (4, 4)
    assert resized.dtype == bool

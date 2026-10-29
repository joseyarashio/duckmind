"""duck-eye's ``duck`` task (spec §4.4, §4.7; owner decision D4: the framework only, no weights).

The letterbox, the planar head and the suppression follow the upstream ``duck-detect`` crate; the
vectors below are its own test cases, ported as numbers. The ONNX path runs a tiny synthetic model
built here (never the official weights): its score is the mean of the red input channel, so it
also proves the input is RGB, 0…1, NCHW, and padded with the letterbox's grey.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from duckmind.eye import detect  # noqa: E402


def test_a_portrait_frame_is_padded_left_and_right() -> None:
    frame = np.full((8, 4, 3), 255, dtype=np.uint8)  # 4 wide, 8 tall
    out, fit = detect.letterbox_rgb(frame, 8)
    assert out.shape == (8, 8, 3)
    assert (fit.scale, fit.pad_x, fit.pad_y) == (1.0, 2.0, 0.0)
    assert out[0, 0].tolist() == [detect.PAD] * 3
    assert out[0, 2].tolist() == [255, 255, 255]
    assert out[0, 7].tolist() == [detect.PAD] * 3


def test_the_duck_frame_letterbox() -> None:
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(720, dtype=np.uint32)[None, :] % 256
    out, fit = detect.letterbox_rgb(frame)
    assert (fit.scale, fit.pad_x, fit.pad_y) == (0.25, 70.0, 0.0)
    # nearest neighbour from the fitted size: column x of the fit is source column 4x
    assert out[5, 70 + 3, 0] == 12
    assert out[5, 69].tolist() == [detect.PAD] * 3 and out[5, 250].tolist() == [detect.PAD] * 3


def test_the_head_is_planar_not_interleaved() -> None:
    raw = [10.0, 100.0, 20.0, 200.0, 4.0, 40.0, 6.0, 60.0, 0.9, 0.1]
    found = detect.decode(raw, detect.Letterbox(1.0, 0.0, 0.0), 0.5, 0.5)
    assert len(found) == 1
    assert found[0].score == pytest.approx(0.9)
    assert found[0].box == (8.0, 17.0, 12.0, 23.0)


def test_overlapping_candidates_collapse_and_map_back_to_the_frame() -> None:
    count = 5
    raw = np.zeros(count * detect.STRIDE, dtype=np.float32)
    for i in range(count):
        raw[i] = 160.0 + i
        raw[count + i] = 160.0
        raw[2 * count + i] = 40.0
        raw[3 * count + i] = 80.0
        raw[4 * count + i] = 0.9 - i * 0.01
    found = detect.decode(raw, detect.Letterbox(0.25, 40.0, 0.0), 0.5, 0.5)
    assert len(found) == 1
    assert found[0].box[0] == 400.0
    assert found[0].width == 160.0 and found[0].height == 320.0
    assert found[0].bearing(1280.0) < 0.0


def test_two_ducks_stay_two() -> None:
    count = 4
    raw = np.zeros(count * detect.STRIDE, dtype=np.float32)
    for i, centre in enumerate([50.0, 52.0, 250.0, 248.0]):
        raw[i] = centre
        raw[count + i] = 160.0
        raw[2 * count + i] = 30.0
        raw[3 * count + i] = 60.0
        raw[4 * count + i] = 0.8
    assert len(detect.decode(raw, detect.Letterbox(1.0, 0.0, 0.0), 0.5, 0.5)) == 2


def test_iou() -> None:
    assert detect.iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)
    assert detect.iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert detect.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_no_model_means_the_task_answers_none(tmp_path: Path) -> None:
    assert detect.open_detector(None) is None
    assert detect.open_detector(tmp_path / "missing.onnx") is None
    assert detect.look(None, np.zeros((4, 4, 3), np.uint8), lambda u, v: (0.0, 0.0)) is None
    # an .rknn needs the NPU runtime, which a Mac does not have
    (tmp_path / "duck.rknn").write_bytes(b"not a model")
    assert detect.open_detector(tmp_path / "duck.rknn") is None


def _tiny_model(path: Path, n: int = 3) -> None:
    """[1,3,320,320] 'images' → [1,5,n]: fixed boxes, score = mean of the red channel for
    candidate 0 (0 for the rest)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    boxes = np.zeros((1, 4, n), dtype=np.float32)
    boxes[0, :, 0] = [160.0, 160.0, 40.0, 80.0]  # cx cy w h in the letterbox
    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    const = helper.make_tensor("boxes", TensorProto.FLOAT, boxes.shape, boxes.ravel().tolist())
    pad = helper.make_tensor("padding", TensorProto.FLOAT, [1, 1, n - 1], [0.0] * (n - 1))
    nodes = [
        helper.make_node("Slice", ["images", "starts", "ends", "axes"], ["red"]),
        helper.make_node("ReduceMean", ["red"], ["mean"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Reshape", ["mean", "shape"], ["score0"]),
        helper.make_node("Concat", ["score0", "padding"], ["scores"], axis=2),
        helper.make_node("Concat", ["boxes", "scores"], ["output0"], axis=1),
    ]
    shape = helper.make_tensor("shape", TensorProto.INT64, [3], [1, 1, 1])
    graph = helper.make_graph(
        nodes,
        "tiny_duck",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, n])],
        initializer=[starts, ends, axes, const, pad, shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_the_onnx_path_with_a_tiny_synthetic_model(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    path = tmp_path / "tiny.onnx"
    _tiny_model(path)
    detector = detect.open_detector(path)
    assert detector is not None and detector.input_size == 320
    red = np.zeros((1280, 720, 3), dtype=np.uint8)
    red[..., 0] = 255
    found = detector.detect(red)
    assert len(found) == 1
    # the red channel's mean: the 180 fitted columns at 1.0, the 140 padded ones at 114/255
    assert found[0].score == pytest.approx((180 + 140 * 114 / 255) / 320, abs=1e-3)
    # (160 - 20 - 70) / 0.25 = 280; (160 - 40) / 0.25 = 480
    assert found[0].box == pytest.approx((280.0, 480.0, 440.0, 800.0))
    # BGR would have put the frame's red into the last channel: only the grey pad is left
    blue = red[..., ::-1].copy()
    assert detector.detect(blue) == []
    seen = detect.look(detector, red, lambda u, v: ((u - 360) / 1000.0, (v - 640) / 1000.0))
    assert seen is not None and len(seen) == 1
    assert seen[0].b_mrad == 0 and seen[0].e_mrad == 0
    assert 0 < seen[0].conf_ppm < 1_000_000
    assert seen[0].box_px == (280, 480, 440, 800)

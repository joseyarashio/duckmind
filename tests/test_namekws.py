import json

import pytest

np = pytest.importorskip("numpy")

from duckmind.namekws.features import (  # noqa: E402
    FeatureConfig,
    StreamingLogMel,
    log_mel,
    mel_filterbank,
)


def _tone(seconds: float, freq: float, rate: int = 16_000, level: float = 0.3):
    t = np.arange(round(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _burst(seconds: float, level: float, seed: int = 0, rate: int = 16_000):
    """Broadband noise: energy in every mel band, unlike a pure tone."""
    rng = np.random.default_rng(seed)
    return (level * rng.standard_normal(round(seconds * rate))).clip(-1, 1).astype(np.float32)


def test_one_second_is_ninety_eight_frames_of_forty_mels():
    config = FeatureConfig()
    assert config.window_frames == 98
    assert log_mel(_tone(1.0, 440.0), config).shape == (98, 40)
    bank = mel_filterbank(config)
    assert bank.shape == (40, 257) and (bank.sum(axis=1) > 0).all()


def test_streaming_features_match_the_whole_clip_exactly():
    rng = np.random.default_rng(3)
    audio = (0.1 * rng.standard_normal(16_000 * 2)).astype(np.float32)
    whole = log_mel(audio)
    stream = StreamingLogMel()
    pieces = []
    for size in (123, 7, 1600, 999, 4000, 13_000, 11_271):
        chunk, audio = audio[:size], audio[size:]
        pieces.append(stream.push(chunk))
    assert np.allclose(np.concatenate(pieces), whole[: sum(p.shape[0] for p in pieces)], atol=1e-5)


def test_int16_and_float_audio_give_the_same_features():
    noise = _burst(0.5, 0.1)
    as_int = np.round(noise * 32767).astype(np.int16)
    assert np.allclose(log_mel(noise), log_mel(as_int), atol=1e-3)


def test_the_feature_digest_changes_with_any_parameter():
    assert FeatureConfig().digest != FeatureConfig(n_mels=32).digest
    assert FeatureConfig().digest == FeatureConfig().digest


def _exported(tmp_path, seed: int = 0):
    """A small random NameTCN exported exactly as training exports it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from duckmind.namekws.detector import DetectionRule
    from duckmind.namekws.model import NameTCN
    from duckmind.namekws.train import export

    torch.manual_seed(seed)
    model = NameTCN(hidden=16, stacks=2, layers=3, outputs=1)
    model.set_normalization(torch.full((40,), -8.0), torch.full((40,), 3.0))
    with torch.no_grad():
        model.head.bias.fill_(0.0)  # posteriors near 0.5, so detections actually happen
        model.head.weight.normal_(0.0, 1.0)
    rule = DetectionRule(threshold=0.55, repeat_threshold=0.5)
    sidecar = export(
        model, tmp_path, rule, {"note": "test"}, {"name": "蛋黃", "pinyin": "dan4 huang2"}
    )
    return model, tmp_path / "name_kws.onnx", sidecar


def test_export_streams_exactly_and_the_sidecar_describes_it(tmp_path):
    from duckmind.namekws.detector import SIDECAR_FORMAT, read_sidecar, rule_from

    _, path, sidecar = _exported(tmp_path)
    assert sidecar["format"] == SIDECAR_FORMAT and sidecar["streaming_parity"] < 1e-3
    read = read_sidecar(path)
    assert read["name"] == "蛋黃"
    assert rule_from(read).threshold == pytest.approx(0.55)


def test_streaming_detection_matches_offline_scanning(tmp_path):
    from duckmind.namekws.detector import (
        DetectionRule,
        NameDetector,
        StreamingModel,
        open_session,
        read_sidecar,
        scan,
    )

    _, path, _ = _exported(tmp_path, seed=3)
    sidecar = read_sidecar(path)
    config = FeatureConfig(**sidecar["features"])
    audio = np.concatenate(
        [_burst(0.8, 0.4, seed=i) if i % 2 else _burst(0.8, 0.002, seed=i) for i in range(12)]
    )
    # Put the bar where this random stand-in actually crosses it, loud bursts or not.
    posteriors = StreamingModel(open_session(path), sidecar["cache_shape"]).push(
        log_mel(audio, config)
    )
    low, high = np.quantile(posteriors, [0.5, 0.99])
    rule = DetectionRule(threshold=float((low + high) / 2), smooth_frames=3, refractory_s=0.5)
    offline = scan(audio, open_session(path), config, rule, sidecar["cache_shape"])
    detector = NameDetector(path, rule)
    live: list[float] = []
    for start in range(0, audio.size, 1_600):
        live += detector.push(audio[start : start + 1_600])
    assert offline, "the stand-in model should fire somewhere"
    assert live == pytest.approx(offline)


def test_a_model_is_refused_when_its_bytes_or_features_changed(tmp_path):
    from duckmind.namekws.detector import read_sidecar

    _, path, _ = _exported(tmp_path)
    read_sidecar(path)
    sidecar_path = path.with_suffix(".json")
    original = sidecar_path.read_text()
    changed = json.loads(original)
    changed["features"]["n_mels"] = 64
    sidecar_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="feature"):
        read_sidecar(path)
    sidecar_path.write_text(original)
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="model bytes"):
        read_sidecar(path)


def test_the_robot_model_is_tiny_and_its_cache_covers_the_receptive_field():
    torch = pytest.importorskip("torch")
    from duckmind.namekws.model import NameTCN

    model = NameTCN()  # the shipped shape: hidden 32, 3 stacks × 4 layers, kernel 5
    assert sum(p.numel() for p in model.parameters()) < 40_000
    assert model.cache_frames == 180 and model.empty_cache(1).shape == (1, 32, 180)
    # Chunked with the cache equals one pass over the whole sequence.
    model.eval()
    features = torch.randn(1, 300, 40)
    with torch.no_grad():
        whole, _ = model(features)
        cache = model.empty_cache(1)
        parts = []
        for start in range(0, 300, 7):
            out, cache = model(features[:, start : start + 7], cache)
            parts.append(out)
    assert torch.allclose(torch.cat(parts, dim=1), whole, atol=1e-5)


def test_max_pool_loss_rewards_one_confident_frame_and_punishes_every_false_one():
    torch = pytest.importorskip("torch")
    from duckmind.namekws.train import max_pool_loss

    quiet = torch.full((1, 50, 1), -6.0)
    peak = quiet.clone()
    peak[0, 30, 0] = 6.0
    positive = torch.tensor([0])
    negative = torch.tensor([-1])
    assert max_pool_loss(peak, positive) < max_pool_loss(quiet, positive)
    assert max_pool_loss(quiet, negative) < max_pool_loss(peak, negative)


def test_the_threshold_keeps_dev_false_accepts_under_half_the_target():
    from duckmind.namekws.train import _detections, choose_rule

    rng = np.random.default_rng(0)
    hours = 1.0
    posteriors = rng.uniform(0.0, 0.3, int(hours * 3600 * 100)).astype(np.float32)
    posteriors[rng.integers(0, posteriors.size, 40)] = 0.97  # rare confident blips
    rule = choose_rule(posteriors, hours, target=10.0)
    assert len(_detections(posteriors, rule)) / hours <= 5.0


def test_audio_folders_assembled_from_links_are_read(tmp_path):
    from duckmind.namekws.data import audio_files

    corpus = tmp_path / "corpus" / "noise"
    corpus.mkdir(parents=True)
    (corpus / "a.wav").write_bytes(b"")
    (corpus / "b.txt").write_bytes(b"")
    linked = tmp_path / "noise"
    linked.mkdir()
    (linked / "musan_noise").symlink_to(corpus, target_is_directory=True)
    assert [p.name for p in audio_files(linked)] == ["a.wav"]


def test_negatives_that_say_the_whole_name_can_be_left_out(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    from duckmind.namekws.data import load_synth

    rows = [
        ("positive", "蛋黃！"),
        ("hard_negative", "蛋黃酥好好吃"),
        ("hard_negative", "淡黃色的衣服"),
    ]
    lines = []
    for index, (kind, text) in enumerate(rows):
        path = tmp_path / f"{index}.wav"
        soundfile.write(path, np.zeros(8000, dtype=np.float32) + 0.01, 16_000)
        lines.append(json.dumps({"path": path.name, "kind": kind, "text": text, "voice_id": "v"}))
    (tmp_path / "metadata.jsonl").write_text("\n".join(lines))
    assert len(load_synth(tmp_path)) == 3
    kept = load_synth(tmp_path, drop_negatives_with="蛋黃")
    assert [clip.kind for clip in kept.clips] == ["keyword", "hard_negative"]

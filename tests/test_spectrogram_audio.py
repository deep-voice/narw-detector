"""The render against scipy's spectrogram; resampling to 2 kHz; the training audio store."""
import numpy as np
import pytest
import torch
from scipy.signal import resample_poly, spectrogram

from conftest import SR, recording, write_wav
from narw_detector.audio import AudioStore, ratio, read_2k, resample
from narw_detector.model import DualStreamSed
from narw_detector.scan import window_probs
from narw_detector.spectrogram import RenderSpec, render, slotted_mag


def test_render_matches_scipy_per_slot():
    spec = RenderSpec()
    x = recording(60, [12.0, 31.5])
    f, _, s = spectrogram(x[:20_000], fs=SR, window="hann", nperseg=256, noverlap=192,
                          mode="magnitude")
    mag = slotted_mag(torch.from_numpy(x)[None], spec)[0].numpy()
    assert mag.shape == (65, 1854)
    np.testing.assert_allclose(mag[:, :309], s[f <= 500][::-1], rtol=1e-4, atol=1e-7)
    out = render(torch.from_numpy(x)[None], spec)
    assert out.shape == (1, 1, 65, 1854)
    assert abs(float(out.mean())) < 1e-4 and abs(float(out.std(unbiased=False)) - 1) < 1e-3


@pytest.mark.parametrize("sr", [8000, 44100, 48000, 64000, 96000])
def test_chunked_resampling_equals_resampling_the_whole_file(sr):
    x = np.random.default_rng(0).standard_normal(int(7.3 * sr)).astype(np.float32)
    up, down = ratio(sr)
    whole = resample_poly(x.astype(np.float64), up, down)
    np.testing.assert_allclose(resample(x, sr, chunk=sr), whole, atol=1e-5)


def test_a_64_khz_recording_reads_as_its_2_khz_version(tmp_path):
    """Same tones written at 64 kHz and at 2 kHz: same input to the model, same output."""
    torch.manual_seed(0)
    x2 = recording(60, [10.0, 33.0], sr=SR)
    x64 = recording(60, [10.0, 33.0], sr=64_000)
    a = read_2k(write_wav(tmp_path / "a.wav", x2, SR))
    b = read_2k(write_wav(tmp_path / "b.wav", x64, 64_000))
    assert len(a) == len(b) == 120_000
    net, spec = DualStreamSed(width=8).eval(), RenderSpec()
    xa, xb = render(torch.from_numpy(a)[None], spec), render(torch.from_numpy(b)[None], spec)
    calls = (xa > 2).float()                              # the upcall pixels of the 2 kHz render
    assert float(((xb > 2).float() * calls).sum() / calls.sum()) > 0.9
    pa = window_probs(net, torch.from_numpy(a)[None], spec)
    pb = window_probs(net, torch.from_numpy(b)[None], spec)
    np.testing.assert_allclose(pa, pb, atol=0.02)


def test_store_resamples_once_and_zero_pads_outside_the_file(tmp_path):
    p2 = write_wav(tmp_path / "in" / "two.wav", recording(12, [3.0]), SR)
    p48 = write_wav(tmp_path / "in" / "fortyeight.wav", recording(12, [3.0], sr=48_000), 48_000)
    store = AudioStore({"two.wav": p2, "fortyeight.wav": p48}, tmp_path / "cache")
    assert store.paths["two.wav"] == p2                              # read in place
    assert store.paths["fortyeight.wav"].parent == tmp_path / "cache"
    assert store.n_samples["fortyeight.wav"] == 24_000
    x = store.read("two.wav", -100, 300)
    assert (x[:100] == 0).all() and np.array_equal(x[100:], read_2k(p2)[:200])
    assert (store.read("two.wav", 23_990, 20)[10:] == 0).all()

"""Unit tests for shared rover hardware selection."""

from aries_common import detect


def _sources(monkeypatch, *, ybimu=False, bno055=False):
    monkeypatch.setattr(detect, "ybimu_available", lambda _port: ybimu)
    monkeypatch.setattr(detect, "bno055_available", lambda _port: bno055)


def test_auto_prefers_ybimu_then_bno055_then_none(monkeypatch):
    _sources(monkeypatch, ybimu=True, bno055=True)
    assert detect.resolve_imu_source("auto", "/bno", "/yb")[0] == "ybimu"

    _sources(monkeypatch, ybimu=False, bno055=True)
    assert detect.resolve_imu_source("auto", "/bno", "/yb")[0] == "bno055"

    _sources(monkeypatch, ybimu=False, bno055=False)
    assert detect.resolve_imu_source("auto", "/bno", "/yb")[0] == "none"


def test_explicit_imu_selection_fails_closed_when_missing(monkeypatch):
    _sources(monkeypatch, ybimu=False, bno055=False)
    assert detect.resolve_imu_source("ybimu", "/bno", "/yb")[0] == "none"
    assert detect.resolve_imu_source("bno055", "/bno", "/yb")[0] == "none"

"""Unit tests for shared rover hardware selection."""

from aries_common import detect


def _sources(monkeypatch, *, ybimu=False, bno055=False):
    monkeypatch.setattr(detect, "ybimu_available", lambda _port: ybimu)
    monkeypatch.setattr(detect, "bno055_available", lambda _port: bno055)


def test_auto_prefers_ybimu_then_bno055_then_picoscan(monkeypatch):
    _sources(monkeypatch, ybimu=True, bno055=True)
    assert detect.resolve_imu_source("auto", "/bno", True, "/yb")[0] == "ybimu"

    _sources(monkeypatch, ybimu=False, bno055=True)
    assert detect.resolve_imu_source("auto", "/bno", True, "/yb")[0] == "bno055"

    _sources(monkeypatch, ybimu=False, bno055=False)
    assert detect.resolve_imu_source("auto", "/bno", True, "/yb")[0] == "picoscan"


def test_explicit_imu_selection_fails_closed_when_missing(monkeypatch):
    _sources(monkeypatch, ybimu=False, bno055=False)
    assert detect.resolve_imu_source("ybimu", "/bno", True, "/yb")[0] == "none"
    assert detect.resolve_imu_source("bno055", "/bno", True, "/yb")[0] == "none"


def test_lidar_auto_requires_driver_and_reachable_sensor(monkeypatch):
    monkeypatch.setattr(
        detect,
        "package_exists",
        lambda package: package == "sick_scan_xd",
    )
    monkeypatch.setattr(detect, "lidar_reachable", lambda _ip: True)
    assert detect.resolve_lidar_enabled("auto", "192.0.2.1")

    monkeypatch.setattr(detect, "package_exists", lambda _package: False)
    assert not detect.resolve_lidar_enabled("auto", "192.0.2.1")

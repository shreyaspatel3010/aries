"""Unit tests for shared rover hardware selection."""

from aries_common import detect


def _imu(monkeypatch, present):
    monkeypatch.setattr(detect, "microstrain_available", lambda _port: present)


def test_auto_starts_microstrain_when_present(monkeypatch):
    _imu(monkeypatch, True)
    assert detect.resolve_imu_source("auto", "/dev/microstrain_main") == ("microstrain", True)


def test_auto_falls_back_to_wheel_odom_when_absent(monkeypatch):
    _imu(monkeypatch, False)
    assert detect.resolve_imu_source("auto", "/dev/microstrain_main") == ("none", False)


def test_explicit_selection_fails_closed_when_missing(monkeypatch):
    _imu(monkeypatch, False)
    assert detect.resolve_imu_source("microstrain", "/dev/microstrain_main")[0] == "none"
    assert detect.resolve_imu_source("true", "/dev/microstrain_main")[0] == "none"


def test_disabling_imu_never_starts_the_driver(monkeypatch):
    _imu(monkeypatch, True)
    for mode in ("false", "none", "odom_only", "wheel_odom"):
        assert detect.resolve_imu_source(mode, "/dev/microstrain_main")[0] == "none"


def test_microstrain_needs_both_device_node_and_driver(monkeypatch):
    monkeypatch.setattr(detect.Path, "exists", lambda _self: True)
    monkeypatch.setattr(
        detect, "package_exists", lambda pkg: pkg == "microstrain_inertial_driver"
    )
    assert detect.microstrain_available("/dev/microstrain_main")

    monkeypatch.setattr(detect, "package_exists", lambda _pkg: False)
    assert not detect.microstrain_available("/dev/microstrain_main")

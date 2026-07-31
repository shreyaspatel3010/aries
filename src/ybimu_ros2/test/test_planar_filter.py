import math

import pytest
from ybimu_ros2.planar_filter import PlanarGyroMagFilter


VALID_FIELD = (40.0e-6, 0.0, 20.0e-6)


def _raw_filter(**kwargs):
    """Build a filter that trusts the magnetometer without a hard-iron fit."""
    kwargs.setdefault('estimate_gyro_bias', False)
    kwargs.setdefault('estimate_hard_iron', False)
    kwargs.setdefault('magnetic_max_rate_rad_s', 0.0)
    return PlanarGyroMagFilter(**kwargs)


def _field_at(yaw, radius=40.0e-6, offset=(0.0, 0.0), vertical=20.0e-6):
    """Return the body-frame field a sensor at ``yaw`` would measure."""
    return (
        radius * math.cos(-yaw) + offset[0],
        radius * math.sin(-yaw) + offset[1],
        vertical,
    )


def test_stationary_orientation_does_not_move():
    filter_ = _raw_filter()
    outputs = [
        filter_.update(index * 0.02, (0.0, 0.0, 0.0), VALID_FIELD)
        for index in range(100)
    ]

    assert outputs[-1].yaw == pytest.approx(0.0, abs=1.0e-12)
    assert outputs[-1].quaternion == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_integrates_z_gyro_when_magnetic_field_is_rejected():
    filter_ = _raw_filter(magnetic_field_max_t=100.0e-6)
    bad_field = (500.0e-6, 0.0, 0.0)
    filter_.update(0.0, (0.0, 0.0, 1.0), bad_field)
    output = filter_.update(0.1, (0.0, 0.0, 1.0), bad_field)

    assert output.yaw == pytest.approx(0.1)
    assert not output.magnetic_correction_used


def test_magnetometer_slowly_corrects_yaw_without_acceleration_input():
    filter_ = _raw_filter(magnetic_correction_time_constant_s=1.0)
    filter_.update(0.0, (0.0, 0.0, 0.0), VALID_FIELD)
    output = filter_.update(0.1, (0.0, 0.0, 0.0), _field_at(0.2))

    expected = (1.0 - math.exp(-0.1)) * 0.2
    assert output.yaw == pytest.approx(expected)
    assert output.magnetic_correction_used


BAD_FIELD = (500.0e-6, 0.0, 0.0)
DT = 0.01  # 100 Hz, the rate the driver runs


def _bias_filter(**kwargs):
    """Build a filter that estimates bias with the magnetometer rejected.

    Bias behaviour is isolated from the magnetic correction so a yaw assertion
    reflects gyro integration alone.
    """
    kwargs.setdefault('estimate_gyro_bias', True)
    kwargs.setdefault('magnetic_field_max_t', 100.0e-6)
    kwargs.setdefault('estimate_hard_iron', False)
    kwargs.setdefault('magnetic_max_rate_rad_s', 0.0)
    return PlanarGyroMagFilter(**kwargs)


def _feed(filter_, samples, rate, start=0, temperature_c=None):
    """Drive the filter with a constant rate and return the last output."""
    output = None
    for index in range(start, start + samples):
        output = filter_.update(
            index * DT, rate, BAD_FIELD, temperature_c)
    return output


def test_bias_is_usable_within_a_fraction_of_a_second():
    """Startup must be effectively instant, not a multi-second window."""
    filter_ = _bias_filter(gyro_bias_stillness_rad_s=0.5)
    output = _feed(filter_, 50, (0.1, -0.2, 0.05))

    assert output.bias_confident, 'bias must converge in well under a second'
    assert filter_.gyro_bias == pytest.approx((0.1, -0.2, 0.05), abs=1.0e-6)
    assert output.angular_velocity == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-6)
    # 50 samples at 100 Hz is half a second.
    assert 50 * DT <= 0.5


def test_yaw_integrates_from_the_first_sample():
    """Yaw must never be frozen waiting for a calibration phase to finish."""
    filter_ = _bias_filter()
    filter_.update(0.0, (0.0, 0.0, 1.0), BAD_FIELD)
    output = filter_.update(DT, (0.0, 0.0, 1.0), BAD_FIELD)

    assert output.yaw == pytest.approx(DT), 'the first turn must not be dropped'


def test_motion_is_never_averaged_into_the_bias():
    """The original failure: baking a startup turn into the bias forever."""
    filter_ = _bias_filter(gyro_bias_stillness_rad_s=0.05)

    for index in range(200):
        rate = 0.4 if index % 2 else -0.4
        filter_.update(index * DT, (0.0, 0.0, rate), BAD_FIELD)
    assert filter_.gyro_bias == (0.0, 0.0, 0.0), 'motion must not be measured'
    assert not filter_.bias_confident

    output = _feed(filter_, 100, (0.0, 0.0, 0.01), start=200)
    assert filter_.gyro_bias[2] == pytest.approx(0.01, abs=1.0e-6)
    assert output.bias_confident, 'stillness after motion must relock quickly'


def test_large_constant_gyro_bias_is_learned_not_read_as_motion():
    """A stationary gyro carries its offset; that must not look like motion.

    This unit reads -0.27 rad/s at rest. Judging stillness by magnitude made
    calibration restart on every sample, so yaw never integrated and the
    heading sat frozen at zero.
    """
    filter_ = _bias_filter(gyro_bias_stillness_rad_s=0.08)
    output = _feed(filter_, 50, (0.05, -0.03, -0.27))

    assert output.bias_confident
    assert filter_.gyro_bias[2] == pytest.approx(-0.27, abs=1.0e-6)

    output = filter_.update(50 * DT, (0.05, -0.03, -0.27 + 0.5), BAD_FIELD)
    assert output.angular_velocity[2] == pytest.approx(0.5, abs=1.0e-3)


def test_uncertainty_regrows_during_motion_so_the_estimate_stays_adaptive():
    """A gain that decays to zero would trust a cold-start offset all mission."""
    filter_ = _bias_filter(gyro_bias_stillness_rad_s=0.05)
    _feed(filter_, 50, (0.0, 0.0, 0.02))
    converged = filter_.gyro_bias_uncertainty[2]
    assert filter_.bias_confident

    for index in range(50, 3050):
        rate = 0.4 if index % 2 else -0.4
        filter_.update(index * DT, (0.0, 0.0, rate), BAD_FIELD)
    assert filter_.gyro_bias_uncertainty[2] > converged, (
        'uncertainty must regrow while no stationary sample is available')

    # A bias that shifted during the drive is then picked up again. The shift is
    # kept inside the corrected-rate gate, which is what distinguishes a drifted
    # bias from a turn. Recovery is statistically weighted rather than instant:
    # the prior still holds real information, so the new value is approached over
    # a few seconds of stillness instead of being jumped to.
    partial = _feed(filter_, 100, (0.0, 0.0, 0.035), start=3050)
    assert partial.bias_confident
    assert 0.02 < filter_.gyro_bias[2] < 0.035, 'must move toward the new bias'

    _feed(filter_, 6000, (0.0, 0.0, 0.035), start=3150)
    assert filter_.gyro_bias[2] == pytest.approx(0.035, abs=1.0e-3)


def test_a_slow_steady_turn_is_not_absorbed_into_the_bias():
    """Dispersion is blind to a constant rate, so the gate has to catch it.

    Measured with the dispersion test alone: a 10 s turn at 0.05 rad/s was 85%
    absorbed into the bias within seconds and the reported rate fell to zero, so
    yaw stopped following a turn that was still happening.
    """
    filter_ = _bias_filter()
    _feed(filter_, 500, (0.0, 0.0, -0.275))
    assert filter_.bias_confident
    start_yaw = filter_.yaw

    output = _feed(filter_, 1000, (0.0, 0.0, -0.275 + 0.05), start=500)

    assert output.yaw - start_yaw == pytest.approx(0.5, abs=1.0e-3), (
        'the full turn must reach yaw')
    assert output.angular_velocity[2] == pytest.approx(0.05, abs=1.0e-3), (
        'the turn must still be reported as a rate')
    assert filter_.gyro_bias[2] == pytest.approx(-0.275, abs=1.0e-4), (
        'a steady turn must not move the bias')


def test_rate_gate_cannot_lock_the_estimator_out_forever():
    """A bias error beyond the gate must not freeze the estimate permanently.

    The gate widens with the interval it has been blocking, so a bias that is
    wrong by more than the gate is eventually admitted instead of deadlocking.
    """
    filter_ = _bias_filter(gyro_bias_relock_s=1.0)
    _feed(filter_, 500, (0.0, 0.0, 0.0))
    assert filter_.bias_confident

    # A bias jump far larger than the gate: every sample looks like motion.
    _feed(filter_, 3000, (0.0, 0.0, 0.5), start=500)
    assert filter_.gyro_bias[2] > 0.05, (
        'the widening gate must recover from a large bias error')


def test_temperature_change_admits_extra_doubt():
    """The barometer's temperature is the hardware's own drift indicator."""
    common = dict(gyro_bias_stillness_rad_s=0.05,
                  gyro_bias_temperature_sensitivity_rad_s_per_c=0.002)
    steady = _bias_filter(**common)
    warming = _bias_filter(**common)

    for filter_ in (steady, warming):
        _feed(filter_, 50, (0.0, 0.0, 0.02), temperature_c=25.0)

    # Both move for the same time; only one also warms up.
    for index in range(50, 1050):
        rate = 0.4 if index % 2 else -0.4
        steady.update(index * DT, (0.0, 0.0, rate), BAD_FIELD, 25.0)
        warming.update(index * DT, (0.0, 0.0, rate), BAD_FIELD,
                       25.0 + 10.0 * (index - 50) / 1000.0)

    assert (warming.gyro_bias_uncertainty[2]
            > steady.gyro_bias_uncertainty[2]), (
        'a thermal transient must raise uncertainty above time alone')


def test_disabled_bias_estimation_leaves_the_gyro_untouched():
    filter_ = _bias_filter(estimate_gyro_bias=False)
    output = _feed(filter_, 50, (0.1, -0.2, 0.05))

    assert filter_.gyro_bias == (0.0, 0.0, 0.0)
    assert output.bias_confident, 'nothing to converge when estimation is off'
    assert output.angular_velocity == pytest.approx((0.1, -0.2, 0.05))


def test_magnetic_correction_is_suspended_during_a_fast_turn():
    filter_ = _raw_filter(
        magnetic_correction_time_constant_s=1.0,
        magnetic_max_rate_rad_s=0.5,
    )
    filter_.update(0.0, (0.0, 0.0, 0.0), VALID_FIELD)
    output = filter_.update(0.1, (0.0, 0.0, 2.0), _field_at(0.2))

    assert not output.magnetic_correction_used
    assert output.yaw == pytest.approx(0.2)


def test_hard_iron_offset_is_estimated_from_a_full_turn():
    """A biased circle must yield its centre, not a swinging heading."""
    offset = (10.0e-6, -6.0e-6)
    filter_ = PlanarGyroMagFilter(
        estimate_gyro_bias=False,
        magnetic_correction_time_constant_s=1.0,
        magnetic_max_rate_rad_s=0.0,
    )

    for index in range(360):
        yaw = math.radians(index)
        filter_.update(index * 0.02, (0.0, 0.0, 0.0),
                       _field_at(yaw, offset=offset))

    assert filter_.hard_iron.calibrated
    assert filter_.hard_iron.offset[0] == pytest.approx(offset[0], abs=1.0e-7)
    assert filter_.hard_iron.offset[1] == pytest.approx(offset[1], abs=1.0e-7)
    assert filter_.hard_iron.field_norm == pytest.approx(40.0e-6, abs=1.0e-7)


def test_uncalibrated_magnetometer_is_not_trusted():
    filter_ = PlanarGyroMagFilter(
        estimate_gyro_bias=False, magnetic_max_rate_rad_s=0.0)
    filter_.update(0.0, (0.0, 0.0, 0.0), VALID_FIELD)
    output = filter_.update(0.1, (0.0, 0.0, 0.0), _field_at(0.2))

    assert not output.magnetic_correction_used
    assert output.yaw == pytest.approx(0.0)


def test_seeded_hard_iron_calibration_is_used_immediately():
    filter_ = PlanarGyroMagFilter(
        estimate_gyro_bias=False,
        magnetic_correction_time_constant_s=1.0,
        magnetic_max_rate_rad_s=0.0,
        estimate_hard_iron=False,
        hard_iron_offset_t=(10.0e-6, -6.0e-6, 0.0),
        hard_iron_field_norm_t=40.0e-6,
    )
    offset = (10.0e-6, -6.0e-6)
    filter_.update(0.0, (0.0, 0.0, 0.0), _field_at(0.0, offset=offset))
    output = filter_.update(0.1, (0.0, 0.0, 0.0), _field_at(0.2, offset=offset))

    assert output.magnetic_correction_used
    assert output.yaw == pytest.approx((1.0 - math.exp(-0.1)) * 0.2)


def test_field_off_the_calibrated_circle_is_rejected():
    filter_ = PlanarGyroMagFilter(
        estimate_gyro_bias=False,
        magnetic_max_rate_rad_s=0.0,
        estimate_hard_iron=False,
        hard_iron_field_norm_t=40.0e-6,
        magnetic_norm_tolerance=0.25,
    )
    filter_.update(0.0, (0.0, 0.0, 0.0), VALID_FIELD)
    output = filter_.update(0.1, (0.0, 0.0, 0.0), _field_at(0.2, radius=70.0e-6))

    assert not output.magnetic_correction_used


def test_long_gap_clamps_instead_of_discarding_the_turn():
    """A stall must not silently lose the rotation that happened in it."""
    bad_field = (500.0e-6, 0.0, 0.0)
    filter_ = _raw_filter(max_dt_s=0.2, magnetic_field_max_t=100.0e-6)
    filter_.update(0.0, (0.0, 0.0, 1.0), bad_field)
    output = filter_.update(5.0, (0.0, 0.0, 1.0), bad_field)

    assert output.yaw == pytest.approx(0.2)


def test_reset_can_preserve_heading_and_calibration():
    filter_ = _bias_filter(gyro_bias_stillness_rad_s=0.5)
    _feed(filter_, 50, (0.0, 0.0, 0.02))
    filter_.update(50 * DT, (0.0, 0.0, 1.0), BAD_FIELD)
    heading, bias = filter_.yaw, filter_.gyro_bias
    assert heading != 0.0

    filter_.reset(preserve_calibration=True)
    assert filter_.yaw == heading
    assert filter_.gyro_bias == bias

    filter_.reset()
    assert filter_.yaw == 0.0
    assert filter_.gyro_bias == (0.0, 0.0, 0.0)

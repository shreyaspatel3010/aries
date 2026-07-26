from aries_vision_grasp.inference import (
    CLOCK_DOMAIN_MISMATCH_AGE_SEC,
    classify_frame_age,
)

MAX_AGE = 2.0

# The wall-clock/sim-time pair that produced the original silent detector:
# Gazebo stamps frames with sim uptime while a use_sim_time:=false node reads
# the epoch, so every result looked ~1.78e9 s old.
WALL_NOW = 1784972858.16
SIM_STAMP = 28.58


def test_sim_stamps_on_a_wall_clock_node_are_a_mismatch_not_staleness():
    verdict, reason = classify_frame_age(
        WALL_NOW - SIM_STAMP, using_sim_time=False, max_age_sec=MAX_AGE
    )
    assert verdict == 'clock_mismatch'
    assert 'use_sim_time:=true' in reason


def test_wall_stamps_on_a_sim_time_node_are_a_mismatch():
    # Reverse direction: the age is hugely negative, so a bare `age > max_age`
    # test would admit a frame whose stamp is meaningless.
    verdict, reason = classify_frame_age(
        SIM_STAMP - WALL_NOW, using_sim_time=True, max_age_sec=MAX_AGE
    )
    assert verdict == 'clock_mismatch'
    assert 'use_sim_time:=false' in reason


def test_mismatch_verdict_ignores_the_sign_of_the_age():
    big = CLOCK_DOMAIN_MISMATCH_AGE_SEC * 2
    for age in (big, -big):
        verdict, _ = classify_frame_age(
            age, using_sim_time=False, max_age_sec=MAX_AGE
        )
        assert verdict == 'clock_mismatch'


def test_genuine_staleness_is_not_reclassified_as_a_clock_mismatch():
    verdict, reason = classify_frame_age(
        3.16, using_sim_time=True, max_age_sec=MAX_AGE
    )
    assert verdict == 'stale'
    assert 'inference_result_max_age_sec' in reason


def test_fresh_frame_passes():
    verdict, reason = classify_frame_age(
        0.16, using_sim_time=True, max_age_sec=MAX_AGE
    )
    assert verdict == 'ok'
    assert reason == ''


def test_stamp_slightly_in_the_future_still_passes():
    # Benign clock jitter puts a stamp a few ms ahead; only epoch-sized
    # magnitudes mean the two clocks are different timelines.
    verdict, _ = classify_frame_age(
        -0.005, using_sim_time=True, max_age_sec=MAX_AGE
    )
    assert verdict == 'ok'


def test_age_exactly_at_the_stale_bound_passes():
    verdict, _ = classify_frame_age(
        MAX_AGE, using_sim_time=True, max_age_sec=MAX_AGE
    )
    assert verdict == 'ok'

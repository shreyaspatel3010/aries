import math

import numpy as np
import pytest

from aries_vision_grasp.wrist_planning import (
    azimuth_variants,
    score_joint_path,
)


def test_azimuth_variants_keep_tool_approach_axis():
    base = np.array([
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    variants = azimuth_variants(base, 12)
    assert len(variants) == 12
    assert variants[0][0] == pytest.approx(0.0)
    for _, rotation in variants:
        assert rotation[:, 2] == pytest.approx(base[:, 2])
        assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-9)


def test_joint_score_prefers_short_joint6_branch():
    current = {'joint1': 0.0, 'joint6': 1.0}
    short_pre = {'joint1': 0.3, 'joint6': 1.2}
    short_grasp = {'joint1': 0.4, 'joint6': 1.25}
    long_pre = {'joint1': 0.1, 'joint6': -2.5}
    long_grasp = {'joint1': 0.2, 'joint6': -2.6}
    short = score_joint_path(
        current, short_pre, short_grasp, ['joint1', 'joint6']
    )
    long = score_joint_path(
        current, long_pre, long_grasp, ['joint1', 'joint6']
    )
    assert short[0] < long[0]
    assert short[1] < long[1]


def test_joint_score_does_not_wrap_bounded_joint6_across_pi():
    current = {'joint6': math.radians(170.0)}
    pre = {'joint6': math.radians(-170.0)}
    grasp = {'joint6': math.radians(-165.0)}
    _, travel, _ = score_joint_path(
        current, pre, grasp, ['joint6']
    )
    assert travel > math.radians(340.0)


def test_joint_score_penalizes_arriving_at_wrist_limit():
    current = {'joint6': 0.0}
    safe_pre = {'joint6': 2.0}
    safe_grasp = {'joint6': 2.1}
    edge_pre = {'joint6': 2.98}
    edge_grasp = {'joint6': 3.05}
    safe = score_joint_path(current, safe_pre, safe_grasp, ['joint6'])
    edge = score_joint_path(current, edge_pre, edge_grasp, ['joint6'])
    assert safe[0] < edge[0]
    assert edge[2] < 0.30


def test_joint_score_requires_complete_joint_maps():
    with pytest.raises(ValueError):
        score_joint_path(
            {'joint6': 0.0},
            {'joint6': 0.1},
            {},
            ['joint6'],
        )

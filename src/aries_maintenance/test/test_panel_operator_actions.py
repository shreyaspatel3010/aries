"""Regression tests for ROS 2 action goal-handle/result sequencing."""

from aries_maintenance.action_utils import (
    load_enabled_controls, log_operation_result, make_joint_trajectory,
    run_action,
)


class ImmediateFuture:
    def __init__(self, value=None, exception=None):
        self._value = value
        self._exception = exception

    def add_done_callback(self, callback):
        callback(self)

    def exception(self):
        return self._exception

    def result(self):
        return self._value


class GoalHandle:
    def __init__(self, result, accepted=True):
        self.accepted = accepted
        self._result = result

    def get_result_async(self):
        return ImmediateFuture(self._result)


class ActionClient:
    def __init__(self, handle):
        self.handle = handle
        self.used_async = False

    def send_goal_async(self, goal):
        self.used_async = True
        return ImmediateFuture(self.handle)


def test_action_waits_for_goal_handle_then_result():
    expected = object()
    client = ActionClient(GoalHandle(expected))
    result, error = run_action(client, object(), "test action", 1.0)
    assert client.used_async
    assert result is expected
    assert error == ""


def test_rejected_action_is_reported_without_reading_a_result():
    client = ActionClient(GoalHandle(object(), accepted=False))
    result, error = run_action(client, object(), "test action", 1.0)
    assert result is None
    assert error == "test action goal rejected"


class Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def error(self, message):
        self.messages.append(("error", message))


def test_failure_then_success_use_distinct_logger_methods():
    logger = Logger()
    log_operation_result(logger, "mcb_12", False, "not localised")
    log_operation_result(logger, "mcb_12", True, "operated")
    assert logger.messages == [
        ("error", "[mcb_12] not localised"),
        ("info", "[mcb_12] operated"),
    ]


def test_gripper_command_targets_trajectory_controller_joint():
    command = make_joint_trajectory(
        "gripper_gear_left_joint", -0.03, 0.75)
    assert command.joint_names == ["gripper_gear_left_joint"]
    assert list(command.points[0].positions) == [-0.03]
    assert command.points[0].time_from_start.sec == 0
    assert command.points[0].time_from_start.nanosec == 750_000_000
    # Zero is intentional: it means immediate execution across wall/sim clocks.
    assert command.header.stamp.sec == 0
    assert command.header.stamp.nanosec == 0


def test_enabled_controls_are_reloaded_from_yaml_in_task_order(tmp_path):
    config = tmp_path / "panel.yaml"
    config.write_text("""
panel_operator:
  ros__parameters:
    controls:
      mcb_0: false
      mcb_1: true
      push_button_0: true
""")
    enabled, resolved, modified = load_enabled_controls(
        config, "panel_operator", ["mcb_0", "mcb_1", "push_button_0"])
    assert enabled == ["mcb_1", "push_button_0"]
    assert resolved == config.resolve()
    assert modified == config.stat().st_mtime


def test_yaml_reload_rejects_incomplete_control_table(tmp_path):
    config = tmp_path / "panel.yaml"
    config.write_text("""
panel_operator:
  ros__parameters:
    controls:
      mcb_0: true
""")
    try:
        load_enabled_controls(
            config, "panel_operator", ["mcb_0", "mcb_1"])
    except ValueError as exc:
        assert "missing: mcb_1" in str(exc)
    else:
        raise AssertionError("an incomplete YAML control table was accepted")

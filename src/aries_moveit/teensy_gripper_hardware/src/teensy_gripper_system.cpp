#include "teensy_gripper_hardware/teensy_gripper_system.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/float32.hpp>

#include <algorithm>
#include <chrono>

namespace teensy_gripper_hardware
{

hardware_interface::CallbackReturn TeensyGripperSystem::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (info_.joints.size() != 1) {
    RCLCPP_ERROR(logger_, "TeensyGripperSystem expects exactly 1 joint, got %zu", info_.joints.size());
    return hardware_interface::CallbackReturn::ERROR;
  }

  joint_name_ = info_.joints[0].name;

  auto & p = info_.hardware_parameters;
  if (p.count("min_pos"))     min_pos_     = std::stod(p.at("min_pos"));
  if (p.count("max_pos"))     max_pos_     = std::stod(p.at("max_pos"));
  if (p.count("cmd_topic"))   cmd_topic_   = p.at("cmd_topic");
  if (p.count("state_topic")) state_topic_ = p.at("state_topic");
  if (p.count("enable_anti_backtrack")) {
    const std::string v = p.at("enable_anti_backtrack");
    anti_backtrack_enabled_ = (v == "true" || v == "True" || v == "1");
  }
  RCLCPP_INFO(logger_, "Anti-backtrack filter %s",
              anti_backtrack_enabled_ ? "ENABLED" : "disabled");

  cmd_pos_   = min_pos_;
  state_pos_ = min_pos_;
  state_vel_ = 0.0;

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> TeensyGripperSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> res;
  res.emplace_back(joint_name_, hardware_interface::HW_IF_POSITION, &state_pos_);
  res.emplace_back(joint_name_, hardware_interface::HW_IF_VELOCITY, &state_vel_);
  return res;
}

std::vector<hardware_interface::CommandInterface> TeensyGripperSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> res;
  res.emplace_back(joint_name_, hardware_interface::HW_IF_POSITION, &cmd_pos_);
  return res;
}

hardware_interface::CallbackReturn TeensyGripperSystem::on_activate(const rclcpp_lifecycle::State &)
{
  // Create a dedicated node for micro-ROS bridge communication
  comm_node_ = rclcpp::Node::make_shared("teensy_gripper_comm");

  // Publisher: sends normalized position command to Teensy
  // BEST_EFFORT + depth 1: fire-and-forget, always latest command, no retransmission.
  // RELIABLE+depth10 (default) can build a 10-message (100 ms) queue backlog,
  // causing the Teensy to replay stale commands after MoveIt reports success.
  auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
  cmd_pub_ = comm_node_->create_publisher<std_msgs::msg::Float32>(cmd_topic_, cmd_qos);

  // Subscriber: receives normalized position state from Teensy.
  // BEST EFFORT to match the firmware's publisher, which had to leave the
  // reliable stream because acknowledging 100 Hz over serial stalled it. This
  // is not a free choice on either side: a BEST_EFFORT publisher and a
  // RELIABLE subscriber are incompatible and never match, so changing one end
  // without the other silences the topic completely.
  state_sub_ = comm_node_->create_subscription<std_msgs::msg::Float32>(
    state_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
    [this](const std_msgs::msg::Float32::SharedPtr msg) {
      latest_state_.store(msg->data);
      last_state_ns_.store(
        std::chrono::steady_clock::now().time_since_epoch().count());
      state_received_.store(true);
    });

  // Spin the communication node in a background thread
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(comm_node_);
  spin_thread_ = std::thread([this]() { executor_->spin(); });

  // Wait for the first state message from the Teensy (up to 5 s).
  // This ensures that cmd_pos_ / state_pos_ are initialised to the ACTUAL
  // servo position (stored in EEPROM on the Teensy) rather than min_pos_.
  // Without this wait, the JointTrajectoryController reads state = min_pos_
  // (open) at activation and holds there, which immediately commands the
  // servo to OPEN — causing the visible "close → open → close" sequence.
  {
    constexpr int kTimeoutMs = 5000;
    constexpr int kPollMs    = 10;
    int elapsed = 0;
    RCLCPP_INFO(logger_, "Waiting for initial Teensy state (timeout %d s)…", kTimeoutMs / 1000);
    while (!state_received_.load() && elapsed < kTimeoutMs) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
      elapsed += kPollMs;
    }
    if (state_received_.load()) {
      const double normalized = std::clamp(static_cast<double>(latest_state_.load()), 0.0, 1.0);
      state_pos_ = min_pos_ + normalized * (max_pos_ - min_pos_);
      cmd_pos_   = state_pos_;   // JTC will hold at the actual position
      RCLCPP_INFO(logger_, "Initial Teensy state received: %.4f (normalized %.3f)", state_pos_, normalized);
    } else {
      RCLCPP_WARN(logger_, "Teensy not connected after %d s — defaulting to min_pos (%.4f). "
        "Gripper may move to OPEN on first activation.", kTimeoutMs / 1000, min_pos_);
    }
  }

  last_written_pos_   = cmd_pos_;
  servo_pos_          = state_pos_;
  backtrack_detected_ = false;
  backtrack_hold_pos_ = state_pos_;
  last_resend_check_ns_.store(
    std::chrono::steady_clock::now().time_since_epoch().count());
  last_publish_ns_.store(
    std::chrono::steady_clock::now().time_since_epoch().count());

  RCLCPP_INFO(logger_, "TeensyGripperSystem active — cmd: %s  state: %s",
    cmd_topic_.c_str(), state_topic_.c_str());
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn TeensyGripperSystem::on_deactivate(const rclcpp_lifecycle::State &)
{
  if (executor_) {
    executor_->cancel();
  }
  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }
  cmd_pub_.reset();
  state_sub_.reset();
  executor_.reset();
  comm_node_.reset();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type TeensyGripperSystem::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  // Echo mode: Teensy's USE_SERVO_FEEDBACK=false means it simply echoes back
  // whatever cmd it last received.  Using that echo (published at 20 Hz) as
  // state_pos_ introduced up to 50 ms of lag: after a close trajectory
  // completes (cmd ≈ 1.0), the planning scene still showed the position from
  // 50 ms earlier (≈ 0.021), causing MoveIt to re-plan from that stale state
  // and send a second trajectory that started at 0.021 — physically snapping
  // the gripper back to near-open then re-closing.
  //
  // Fix: state_pos_ tracks servo_pos_ (the anti-backtrack-filtered command),
  // giving the planning scene zero lag and making it immune to sudden cmd
  // backtracks.  The Teensy state subscriber is still active for:
  //   • on_activate() initial position bootstrap
  //   • write() disconnect detection via last_state_ns_ / state_received_
  if (state_received_.load()) {
    state_pos_ = servo_pos_;
  }
  state_vel_ = 0.0;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type TeensyGripperSystem::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (!cmd_pub_) {
    return hardware_interface::return_type::ERROR;
  }

  cmd_pos_ = std::clamp(cmd_pos_, min_pos_, max_pos_);

  // Anti-backtrack filter ────────────────────────────────────────────────────
  // A sudden large decrease in cmd_pos_ indicates MoveIt re-planned a
  // trajectory from a stale (near-open) planning-scene state while the gripper
  // was already closed.  During hold the servo stays closed; once cmd climbs
  // back within the resync threshold of the hold position we resume normal
  // tracking.
  //
  // Both thresholds MUST be fractions of the joint range, not absolute numbers.
  // They were hardcoded 0.05 and 0.01 when this axis was 0.09 wide (56 % and
  // 11 % of stroke, hence the old "~56 %" comment).  The joint is now revolute,
  // min_pos -1.57 to max_pos 0.07 = 1.64 rad, which silently reduced them to
  // 3 % and 0.6 % and turned the guard into the failure it was meant to
  // prevent: joint_limits.yaml permits 10 rad/s and the loop runs at 80 Hz, so
  // one ordinary trajectory step is up to 0.125 rad — past 0.05 — and every
  // brisk OPEN trips the filter.  It then pins servo_pos_ at the closed
  // position and waits for cmd to come back within 0.01 rad of it, which an
  // opening trajectory never does, so the gripper stays shut for the rest of
  // the session with a single WARN as the only trace.  Observed in the field as
  // "cmd jumped 0.0700 -> 0.0200".  The units in the old messages said "m" for
  // what have always been radians here, which hid it further.
  //
  // DISABLED BY DEFAULT, and re-tuning it is not the answer. The signal it
  // keys on — a large negative step in cmd_pos_ — is exactly what a
  // legitimate full OPEN looks like: closed to open IS 0.07 -> -1.57, the
  // entire 1.64 rad stroke, in whatever step the trajectory happens to
  // produce. A bad replan and a good open are the same measurement, so no
  // threshold separates them; raising it from 3 % to 56 % of stroke only
  // moved which opens got swallowed. And the misfire is expensive: the servo
  // is pinned until cmd returns within the resync band, which an opening
  // trajectory need never do, so the gripper silently stops obeying.
  //
  // The stale-planning-scene snap this was built for was already fixed at
  // source — read() now reports servo_pos_ with zero lag (see the note
  // there), so MoveIt no longer replans from a near-open stale state. This is
  // a workaround for a bug that is gone. Set the hardware param
  // enable_anti_backtrack="true" in the URDF to bring it back.
  const double range = max_pos_ - min_pos_;
  const double backtrack_threshold = 0.56 * range;
  const double resync_threshold    = 0.11 * range;

  if (!anti_backtrack_enabled_) {
    backtrack_detected_ = false;   // falls through to plain tracking below
  } else if (!backtrack_detected_ && (servo_pos_ - cmd_pos_) > backtrack_threshold) {
    backtrack_detected_  = true;
    backtrack_hold_pos_  = servo_pos_;
    RCLCPP_WARN(logger_,
      "Anti-backtrack: cmd jumped %.4f -> %.4f rad (> %.4f rad); holding servo at %.4f rad "
      "until cmd resynchronises",
      servo_pos_, cmd_pos_, backtrack_threshold, servo_pos_);
  }

  if (backtrack_detected_) {
    if (std::abs(cmd_pos_ - backtrack_hold_pos_) < resync_threshold) {
      RCLCPP_INFO(logger_,
        "Anti-backtrack: cmd resynchronised at %.4f rad — resuming normal tracking", cmd_pos_);
      backtrack_detected_ = false;
      servo_pos_ = backtrack_hold_pos_;
    } else {
      servo_pos_ = backtrack_hold_pos_;   // hold; fall through to deadband/keepalive
    }
  } else {
    servo_pos_ = cmd_pos_;
  }
  // ──────────────────────────────────────────────────────────────────────────

  using clock = std::chrono::steady_clock;
  const auto now_ns = clock::now().time_since_epoch().count();

  // Before the Teensy has ever sent a state message, silently track cmd_pos_
  // so that the first connection does not replay whatever stale value the
  // controller initialised with (avoids ghost open/close on startup).
  if (!state_received_.load()) {
    // Silent until now, which hid a whole class of failure: if the Teensy is
    // not publishing when the plugin activates, this branch swallows EVERY
    // command for the rest of the session and nothing anywhere says so. The
    // gripper simply never moves. Seen when the board registers its micro-ROS
    // entities and then stops executing — /teensy_gripper and /gripper/state
    // both exist, the topic just never carries a message, and the agent keeps
    // the stale entities alive so everything looks connected.
    if (now_ns - last_disconnect_warn_ns_ > 5'000'000'000LL) {
      last_disconnect_warn_ns_ = now_ns;
      RCLCPP_WARN(logger_,
        "Never received /gripper/state — no command has EVER been sent to the servo. "
        "The board is not publishing: check that %s has a live publisher "
        "(ros2 topic hz %s), then reset or reflash the Teensy.",
        state_topic_.c_str(), state_topic_.c_str());
    }
    last_written_pos_ = cmd_pos_;
    return hardware_interface::return_type::OK;
  }

  // While the Teensy is disconnected (state older than 2 s): mark for resend
  // on the next reconnect but do not publish now.
  //
  // This branch used to return OK in silence, which made a mid-run agent death
  // invisible. read() keeps state_pos_ = servo_pos_ once state_received_ has
  // latched, so the JTC still sees perfect tracking and reports every gripper
  // goal as succeeded while nothing at all reaches the servo — the gripper just
  // stops responding with no error in any log. Say so, throttled, because this
  // runs in the control loop.
  const auto age_ns = now_ns - last_state_ns_.load();
  if (age_ns >= 2'000'000'000LL) {
    if (now_ns - last_disconnect_warn_ns_ > 2'000'000'000LL) {
      last_disconnect_warn_ns_ = now_ns;
      RCLCPP_WARN(logger_,
        "No /gripper/state for %.1f s — Teensy session is down; gripper commands are "
        "NOT reaching the servo. Check the micro_ros_agent process and the board.",
        static_cast<double>(age_ns) * 1e-9);
    }
    last_written_pos_ = -1.0;   // force resend on reconnect
    return hardware_interface::return_type::OK;
  }

  // Periodic retry: if the Teensy has not confirmed the target after 1 s,
  // force a resend.  With USE_SERVO_FEEDBACK=false the Teensy echoes the
  // commanded value immediately, so this only fires when the command was
  // genuinely lost during a brief reconnect.
  const bool at_target = std::abs(state_pos_ - cmd_pos_) < 0.002;
  if (!at_target && (now_ns - last_resend_check_ns_.load() > 1'000'000'000LL)) {
    last_resend_check_ns_.store(now_ns);
    last_written_pos_ = -1.0;   // fall through to publish below
  }

  // Keepalive: republish every 20 ms to ensure the final trajectory target is
  // always delivered promptly.  With the 0.0005 m deadband, per-cycle steps at
  // slow trajectory speeds fall below the threshold; without a short keepalive
  // the last command is withheld until the next forced send, causing the servo
  // to stall 0.0005 m short of target for up to 500 ms after the JTC reports
  // success.  20 ms is fast enough to be imperceptible while still avoiding a
  // continuous 200 Hz command flood during the hold phase.
  if (now_ns - last_publish_ns_.load() > 20'000'000LL) {
    last_written_pos_ = -1.0;   // fall through to publish below
  }

  // Dead-band: skip if servo position hasn't changed meaningfully.
  if (std::abs(servo_pos_ - last_written_pos_) < 0.0005) {
    return hardware_interface::return_type::OK;
  }

  const float normalized = (range > 0.0) ? static_cast<float>((servo_pos_ - min_pos_) / range) : 0.0f;

  std_msgs::msg::Float32 msg;
  msg.data = normalized;
  cmd_pub_->publish(msg);
  last_written_pos_ = servo_pos_;
  last_publish_ns_.store(now_ns);

  return hardware_interface::return_type::OK;
}

}  // namespace teensy_gripper_hardware

PLUGINLIB_EXPORT_CLASS(teensy_gripper_hardware::TeensyGripperSystem, hardware_interface::SystemInterface)

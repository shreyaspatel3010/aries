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

  // Subscriber: receives normalized position state from Teensy
  state_sub_ = comm_node_->create_subscription<std_msgs::msg::Float32>(
    state_topic_, 10,
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
  // was already closed.  Threshold 0.05 m is ~56 % of full range — far above
  // normal trajectory step (~0.001 m/cycle at 100 Hz) but catches the observed
  // 0.998 → 0.021 snap.  During hold the servo stays closed; once cmd climbs
  // back within kResyncThreshold of the hold position we resume normal tracking.
  constexpr double kBacktrackThreshold = 0.05;   // m
  constexpr double kResyncThreshold    = 0.01;   // m

  if (!backtrack_detected_ && (servo_pos_ - cmd_pos_) > kBacktrackThreshold) {
    backtrack_detected_  = true;
    backtrack_hold_pos_  = servo_pos_;
    RCLCPP_WARN(logger_,
      "Anti-backtrack: cmd jumped %.4f → %.4f m; holding servo at %.4f m until cmd resynchronises",
      servo_pos_, cmd_pos_, servo_pos_);
  }

  if (backtrack_detected_) {
    if (std::abs(cmd_pos_ - backtrack_hold_pos_) < kResyncThreshold) {
      RCLCPP_INFO(logger_,
        "Anti-backtrack: cmd resynchronised at %.4f m — resuming normal tracking", cmd_pos_);
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
    last_written_pos_ = cmd_pos_;
    return hardware_interface::return_type::OK;
  }

  // While the Teensy is disconnected (state older than 2 s): mark for resend
  // on the next reconnect but do not publish now.
  const auto age_ns = now_ns - last_state_ns_.load();
  if (age_ns >= 2'000'000'000LL) {
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

  const double range = max_pos_ - min_pos_;
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

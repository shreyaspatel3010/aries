#include "st3215_gripper_hardware/st3215_gripper_system.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace st3215_gripper_hardware
{

namespace
{
constexpr double STEPS_PER_RAD = STEPS_PER_REV / (2.0 * M_PI);   // 651.8986

// Jaw gap per radian of pinion, mm.  Both racks move, so it is twice the pitch
// radius: 21 teeth on a 3.0000 mm rack pitch gives r = 21*3/(2*pi) mm.
//
// FOURTH COPY of that pitch radius.  The others are aries/urdf/gripper_st3215.xacro,
// aries_vision_grasp/fourbar.py and aries_moveit/scripts/gripper_arc_visualizer.py,
// and src/aries/test/test_gripper_st3215.py cross-checks all of them against each
// other.  It is copied rather than shared because those are XML and Python and this
// is a ros2_control plugin that must not gain a dependency on either.
constexpr double JAW_MM_PER_RAD = 2000.0 * 0.01002676;

double param_double(const hardware_interface::HardwareInfo & info,
                    const std::string & key, double fallback)
{
  const auto it = info.hardware_parameters.find(key);
  return it == info.hardware_parameters.end() ? fallback : std::stod(it->second);
}

int param_int(const hardware_interface::HardwareInfo & info,
              const std::string & key, int fallback)
{
  const auto it = info.hardware_parameters.find(key);
  return it == info.hardware_parameters.end() ? fallback : std::stoi(it->second);
}

std::string param_string(const hardware_interface::HardwareInfo & info,
                         const std::string & key, const std::string & fallback)
{
  const auto it = info.hardware_parameters.find(key);
  return it == info.hardware_parameters.end() ? fallback : it->second;
}

bool param_bool(const hardware_interface::HardwareInfo & info,
                const std::string & key, bool fallback)
{
  const auto it = info.hardware_parameters.find(key);
  if (it == info.hardware_parameters.end()) { return fallback; }
  return it->second == "true" || it->second == "True" || it->second == "1";
}
}  // namespace

int ST3215GripperSystem::steps_from_rad(double q) const
{
  const double dir = invert_ ? -1.0 : 1.0;
  return static_cast<int>(std::lround(closed_steps_ + dir * (q - max_pos_) * STEPS_PER_RAD));
}

double ST3215GripperSystem::rad_from_steps(int steps) const
{
  const double dir = invert_ ? -1.0 : 1.0;
  return max_pos_ + dir * (steps - closed_steps_) / STEPS_PER_RAD;
}

hardware_interface::CallbackReturn
ST3215GripperSystem::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
      hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (info_.joints.size() != 1) {
    RCLCPP_FATAL(logger_, "expected exactly one joint, got %zu", info_.joints.size());
    return hardware_interface::CallbackReturn::ERROR;
  }
  joint_name_ = info_.joints[0].name;

  port_ = param_string(info_, "port", port_);
  baud_ = param_int(info_, "baud", baud_);
  servo_id_ = static_cast<uint8_t>(param_int(info_, "servo_id", servo_id_));
  closed_steps_ = param_int(info_, "closed_steps", closed_steps_);
  invert_ = param_bool(info_, "invert", invert_);
  min_pos_ = param_double(info_, "min_pos", min_pos_);
  max_pos_ = param_double(info_, "max_pos", max_pos_);
  accel_ = param_int(info_, "accel", accel_);
  goal_speed_ = param_int(info_, "goal_speed", goal_speed_);
  torque_limit_ = param_int(info_, "torque_limit", torque_limit_);
  io_rate_hz_ = param_double(info_, "io_rate_hz", io_rate_hz_);
  timeout_ms_ = param_int(info_, "timeout_ms", timeout_ms_);

  squeeze_relax_ = param_bool(info_, "squeeze_relax", squeeze_relax_);
  stall_error_rad_ = param_double(info_, "stall_error_rad", stall_error_rad_);
  stall_speed_rad_ = param_double(info_, "stall_speed_rad", stall_speed_rad_);
  stall_hold_s_ = param_double(info_, "stall_hold_s", stall_hold_s_);
  relax_bias_rad_ = param_double(info_, "relax_bias_rad", relax_bias_rad_);
  max_slew_rad_per_s_ = param_double(info_, "max_slew_rad_per_s", max_slew_rad_per_s_);

  publish_diagnostics_ = param_bool(info_, "publish_diagnostics", publish_diagnostics_);
  telemetry_rate_hz_ = param_double(info_, "telemetry_rate_hz", telemetry_rate_hz_);
  warn_temp_margin_c_ = param_double(info_, "warn_temp_margin_c", warn_temp_margin_c_);
  warn_volt_margin_v_ = param_double(info_, "warn_volt_margin_v", warn_volt_margin_v_);
  warn_current_frac_ = param_double(info_, "warn_current_frac", warn_current_frac_);
  warn_load_frac_ = param_double(info_, "warn_load_frac", warn_load_frac_);

  if (min_pos_ >= max_pos_) {
    RCLCPP_FATAL(logger_, "min_pos %.4f must be below max_pos %.4f", min_pos_, max_pos_);
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Start held at the open end, matching what the mock and Teensy backends
  // report before their first command.  A gripper that comes up believing it
  // is closed makes MoveIt plan its first open as a huge unexpected motion.
  // on_activate() replaces both of these with the servo's real position.
  cmd_pos_.store(min_pos_);
  state_pos_.store(min_pos_);
  last_reported_pos_ = min_pos_;

  // Telemetry node. Wrapped because a failure here must not cost the gripper:
  // node creation throws if the component name is not a legal ROS name, or if
  // rclcpp is not initialised (a plain unit test constructing the component).
  // Losing the display is a nuisance; losing the gripper is not.
  if (publish_diagnostics_) {
    std::string node_name = info_.name.empty() ? "st3215_gripper" : info_.name;
    for (char & c : node_name) {
      if (!std::isalnum(static_cast<unsigned char>(c))) { c = '_'; }
    }
    if (std::isdigit(static_cast<unsigned char>(node_name.front()))) {
      node_name.insert(node_name.begin(), '_');
    }
    try {
      tel_node_ = std::make_shared<rclcpp::Node>(node_name + "_telemetry");
      // Default (reliable) QoS, which is what every other /diagnostics
      // publisher and every off-the-shelf reader - rqt_robot_monitor,
      // diagnostic_aggregator - uses. A best-effort publisher here would
      // silently fail to match all of them. It cannot block the io thread:
      // keep-last depth 5 overwrites rather than waiting.
      diag_pub_ = tel_node_->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/diagnostics", rclcpp::QoS(5));
    } catch (const std::exception & e) {
      RCLCPP_WARN(logger_, "no gripper telemetry: %s", e.what());
      tel_node_.reset();
      diag_pub_.reset();
    }
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> ST3215GripperSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> ifaces;
  // The atomics are the source of truth, but StateInterface wants a raw
  // double*, so read() copies them into these members every cycle.
  ifaces.emplace_back(joint_name_, hardware_interface::HW_IF_POSITION, &pos_iface_);
  ifaces.emplace_back(joint_name_, hardware_interface::HW_IF_VELOCITY, &vel_iface_);
  ifaces.emplace_back(joint_name_, hardware_interface::HW_IF_EFFORT, &eff_iface_);
  return ifaces;
}

std::vector<hardware_interface::CommandInterface> ST3215GripperSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> ifaces;
  ifaces.emplace_back(joint_name_, hardware_interface::HW_IF_POSITION, &cmd_iface_);
  return ifaces;
}

void ST3215GripperSystem::inhibit(const std::string & reason)
{
  inhibit_reason_ = reason;
  command_inhibited_.store(true);
  RCLCPP_FATAL(logger_, "GRIPPER DISABLED: %s", reason.c_str());
  RCLCPP_FATAL(logger_,
               "The gripper will report its position but accept no commands. "
               "Everything else - the arm, the rover - keeps running.");
}

std::string ST3215GripperSystem::probe_bus()
{
  // IDs 1..30 covers the factory default and any hand-assigned one; a full
  // 0..253 sweep would take most of a minute at this timeout for no real gain.
  const std::vector<int> bauds{baud_, 500000, 115200, 250000, 57600};
  std::string found;
  for (size_t i = 0; i < bauds.size(); ++i) {
    if (i > 0) {
      bus_.close();
      if (!bus_.open(port_, bauds[i], timeout_ms_)) { continue; }
    }
    std::string ids;
    for (uint8_t id = 1; id <= 30; ++id) {
      if (bus_.ping(id)) { ids += (ids.empty() ? "" : ", ") + std::to_string(id); }
    }
    if (!ids.empty()) {
      found += (found.empty() ? "" : "; ") + std::to_string(bauds[i]) +
               " baud: id " + ids;
      break;      // one answering baud is the answer; stop hunting
    }
  }
  bus_.close();
  if (found.empty()) {
    return " NOTHING answered on any ID at any of 1M/500k/250k/115200/57600 baud. "
           "The adapter enumerated (the port opened), so this is the servo side: "
           "check that the bus servo has its own 6-12 V supply - USB powers the "
           "adapter's logic only - and that the 3-pin lead is seated.";
  }
  return " Something DID answer, at " + found +
         ". Set servo_id:= / servo_bus_baud:= to match, or re-address the servo.";
}

hardware_interface::CallbackReturn
ST3215GripperSystem::on_activate(const rclcpp_lifecycle::State &)
{
  // The stroke has to fit inside the servo's single turn without crossing the
  // 4095 -> 0 encoder seam, or the gripper stops partway through an open with
  // nothing to show for it.
  const int open_steps = steps_from_rad(min_pos_);
  if (open_steps < POS_MIN || open_steps > POS_MAX ||
      closed_steps_ < POS_MIN || closed_steps_ > POS_MAX)
  {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "closed_steps=%d with invert=%s puts the open end at step %d, outside the "
             "usable %d..%d. The %.3f rad stroke needs %d steps and the servo has one turn. "
             "Relaunch with gripper_closed_steps:=<value> or gripper_servo_invert:=true.",
             closed_steps_, invert_ ? "true" : "false", open_steps, POS_MIN, POS_MAX,
             max_pos_ - min_pos_,
             static_cast<int>(std::lround((max_pos_ - min_pos_) * STEPS_PER_RAD)));
    inhibit(msg);
    return start_inhibited();
  }

  if (!bus_.open(port_, baud_, timeout_ms_)) {
    inhibit("cannot open " + port_ + ": " + bus_.last_error() +
            ". Check the adapter is plugged in and that you are in the dialout group; "
            "scripts/setup_system.sh installs the udev rule that makes the symlink.");
    return start_inhibited();
  }

  // Spread the attempts over ~0.5 s rather than firing five back to back: a
  // servo that was just powered, or an adapter that has just enumerated, is not
  // ready within a few milliseconds.
  bool alive = false;
  for (int attempt = 0; attempt < 10 && !alive; ++attempt) {
    alive = bus_.ping(servo_id_);
    if (!alive) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
  }
  if (!alive) {
    // Do not just report the failure - go and find out WHY, while the port is
    // already open and nothing else is using it. "No reply from id 1" is the
    // same message whether the servo is at id 3, at another baud, or unpowered,
    // and those have completely different fixes.
    const std::string probe = probe_bus();
    inhibit("no reply from servo id " + std::to_string(servo_id_) + " on " + port_ +
            " at " + std::to_string(baud_) + " baud." + probe);
    return start_inhibited();
  }

  // The servo's own protection limits, before anything else: every DANGER and
  // CUTOFF line the telemetry publishes is measured against these, and the
  // inhibit paths below publish too.
  read_protection_limits();

  // Wheel mode ignores goal positions ENTIRELY and reports no error for it, so
  // the whole gripper would look healthy and never move.  Read, never write:
  // REG_MODE is EPROM.
  uint8_t mode = 0;
  if (bus_.read8(servo_id_, REG_MODE, mode) && mode != 0) {
    // The one write made while inhibited, and it is a STOP, not a command: in
    // wheel mode REG_GOAL_SPEED is a continuous velocity, so a non-zero value
    // left over from a bench session has the pinion running at its end stop
    // right now. Zero it, then touch nothing else.
    bus_.write16(servo_id_, REG_GOAL_SPEED, 0);
    inhibit("servo " + std::to_string(servo_id_) + " is in mode " + std::to_string(mode) +
            " (wheel), where goal POSITIONS are silently ignored and goal SPEED becomes a "
            "continuous velocity - so commanding it would spin the pinion into its end "
            "stop, not do nothing. Fix it once with "
            "`python3 scripts/st3215_test.py --mode position`; this component will not "
            "write EPROM. Its goal speed has been zeroed in case it was already running.");
    return start_inhibited();
  }

  // WHERE THE JAWS ARE, before anything is allowed to command them.
  //
  // This is a hard requirement, not a nicety.  Both the command the controller
  // will send and the goal this component writes are seeded from it, and
  // without it the very first write() would drive the jaws to whatever the
  // command interface happens to hold.  Measured on the bench emulator: with
  // no controller loaded at all, the servo walked from closed (+0.07) to
  // exactly q = 0 - the C++ default of the command member - because write()
  // acted on a value nothing had ever set.  On the rover that is the gripper
  // moving 1.4 mm off closed the instant the stack comes up, and it would drop
  // anything it was holding.  Hence the NaN default on cmd_iface_, the isfinite
  // guard in write(), and this seed.
  uint16_t raw_pos = 0;
  bool located = false;
  for (int attempt = 0; attempt < 10 && !located; ++attempt) {
    located = bus_.read16(servo_id_, REG_PRESENT_POSITION, raw_pos);
  }
  if (!located) {
    inhibit("servo " + std::to_string(servo_id_) + " answers a ping but not a position "
            "read, so there is no way to tell where the jaws are. Commanding them blind "
            "would move them by an unknown amount from an unknown place.");
    return start_inhibited();
  }
  // NOT clamped into [min_pos_, max_pos_]. Where the gripper IS parked is a
  // fact; clamping it to the command band and then commanding that would walk
  // the jaws to the band edge the moment the stack starts.
  const double q = rad_from_steps(static_cast<int>(raw_pos));
  state_pos_.store(q);
  cmd_pos_.store(q);            // hold station; do not lurch on the first write
  cmd_iface_ = q;
  slewed_goal_ = q;
  slew_primed_ = true;
  last_reported_pos_ = q;
  have_state_.store(true);
  RCLCPP_INFO(logger_, "servo %u at step %u -> q = %.4f rad (gap %.1f mm)",
              servo_id_, raw_pos, q, JAW_MM_PER_RAD * (max_pos_ - q));

  // Only written when explicitly asked for.  On this servo the acceleration
  // register silently caps top speed as well as ramping, and it is what
  // produced a 44 deg overshoot on the bench, so 0 means "leave whatever is
  // stored alone" rather than "no acceleration limit".
  if (accel_ > 0) { bus_.write8(servo_id_, REG_ACCELERATION, static_cast<uint8_t>(accel_)); }
  if (torque_limit_ > 0) {
    bus_.write16(servo_id_, REG_TORQUE_LIMIT, static_cast<uint16_t>(torque_limit_));
  }
  // ORDER MATTERS: goal first, THEN torque.
  //
  // The servo keeps its goal position across a torque-off/torque-on cycle, and
  // across the whole session that wrote it. Enabling torque with a stale goal
  // makes the horn SNAP to wherever the last run left it - reported from the
  // bench as "when starting it goes to the close position", which is exactly
  // what a stale closed-end goal looks like. Writing the present position as
  // the goal first makes the enable a no-op hold.
  if (goal_speed_ > 0) {
    bus_.write16(servo_id_, REG_GOAL_SPEED, static_cast<uint16_t>(goal_speed_));
  }
  bus_.write16(servo_id_, REG_GOAL_POSITION, raw_pos);
  bus_.write8(servo_id_, REG_TORQUE_ENABLE, 1);

  running_.store(true);
  io_thread_ = std::thread(&ST3215GripperSystem::io_loop, this);
  RCLCPP_INFO(logger_, "ST3215 gripper up on %s, servo %u, closed_steps %d%s",
              port_.c_str(), servo_id_, closed_steps_, invert_ ? " (inverted)" : "");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
ST3215GripperSystem::start_inhibited()
{
  // Activation succeeded so the arm keeps running, but there is nothing safe to
  // command. The loop runs either way: with the port open it still reads, so
  // /joint_states carries the true jaw position rather than an echo, and with
  // the port shut every bus call returns false harmlessly. It runs even then
  // because it is what publishes the telemetry, and a gripper whose adapter is
  // unplugged is the case where the operator most needs to be told why - an
  // RViz panel that simply goes blank looks like a display bug, not a fault.
  running_.store(true);
  io_thread_ = std::thread(&ST3215GripperSystem::io_loop, this);
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
ST3215GripperSystem::on_deactivate(const rclcpp_lifecycle::State &)
{
  running_.store(false);
  if (io_thread_.joinable()) { io_thread_.join(); }
  if (bus_.is_open()) {
    // Leave torque ON.  Cutting it drops whatever the gripper is holding, and
    // deactivate happens on controller switches as well as on shutdown.
    bus_.close();
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

void ST3215GripperSystem::read_protection_limits()
{
  // EPROM, so this is read once and never polled: these do not change unless
  // somebody writes them, and this component never does.
  uint8_t t = 0, hi = 0, lo = 0;
  uint16_t amps = 0;
  if (bus_.read8(servo_id_, REG_MAX_TEMPERATURE, t)) { max_temp_c_ = t; }
  if (bus_.read8(servo_id_, REG_MAX_VOLTAGE, hi)) { max_volt_v_ = hi * VOLTAGE_LSB_V; }
  if (bus_.read8(servo_id_, REG_MIN_VOLTAGE, lo)) { min_volt_v_ = lo * VOLTAGE_LSB_V; }
  if (bus_.read16(servo_id_, REG_PROTECTION_CURRENT, amps)) {
    protect_current_ma_ = amps * CURRENT_LSB_MA;
  }
  RCLCPP_INFO(logger_,
              "servo protection limits: temp <= %.0f C, voltage %.1f-%.1f V, current <= %.0f mA "
              "(-1 = the servo did not answer that register)",
              max_temp_c_, min_volt_v_, max_volt_v_, protect_current_ma_);
}

void ST3215GripperSystem::publish_telemetry()
{
  if (!diag_pub_) { return; }

  // Its own buffer per call, not a shared one: several of the strings below
  // are built by concatenating two num() results in one expression, and a
  // shared buffer would make that depend on evaluation order.
  auto num = [](const char * fmt, double v) {
    char b[64];
    std::snprintf(b, sizeof(b), fmt, v);
    return std::string(b);
  };

  diagnostic_msgs::msg::DiagnosticStatus st;
  st.name = (info_.name.empty() ? std::string("gripper") : info_.name) + ": ST3215 servo";
  st.hardware_id = port_ + " id " + std::to_string(static_cast<int>(servo_id_));

  const bool inhibited = command_inhibited_.load();
  const uint32_t misses = consecutive_failures_.load();
  // Half a second of silence at any sane io_rate is many missed polls in a row,
  // not one unlucky USB frame.
  const bool comms_lost = misses > static_cast<uint32_t>(std::max(1.0, io_rate_hz_ * 0.5));
  const uint8_t status = state_status_.load();
  const double q = state_pos_.load();
  const double gap_mm = JAW_MM_PER_RAD * (max_pos_ - q);
  const double temp = state_temp_.load();
  const double volt = state_volt_.load();
  const double current = std::fabs(state_current_ma_.load());
  const double load = std::fabs(state_load_frac_.load());
  const bool moving = state_moving_.load();

  // DANGER: within a margin of a limit the SERVO enforces. A limit that did not
  // read back is left out rather than compared against a made-up number.
  std::vector<std::string> dangers;
  if (max_temp_c_ > 0 && temp >= max_temp_c_ - warn_temp_margin_c_) {
    dangers.push_back(num("temperature %.0f C", temp) + num(" (cutoff %.0f)", max_temp_c_));
  }
  if (max_volt_v_ > 0 && volt >= max_volt_v_ - warn_volt_margin_v_) {
    dangers.push_back(num("voltage %.1f V high", volt) + num(" (cutoff %.1f)", max_volt_v_));
  }
  if (min_volt_v_ > 0 && volt > 0.0 && volt <= min_volt_v_ + warn_volt_margin_v_) {
    dangers.push_back(num("voltage %.1f V low", volt) + num(" (cutoff %.1f)", min_volt_v_));
  }
  if (protect_current_ma_ > 0 && current >= warn_current_frac_ * protect_current_ma_) {
    dangers.push_back(num("current %.0f mA", current) +
                      num(" (cutoff %.0f)", protect_current_ma_));
  }
  // Not a servo limit: a gripper squeezing something sits here indefinitely.
  // Worth showing because it is what heats the winding in the first place.
  if (!moving && load >= warn_load_frac_) {
    dangers.push_back(num("holding at %.0f%% torque", 100.0 * load));
  }

  const std::string flags = status_flags(status);
  if (inhibited) {
    st.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    st.message = "CUTOFF - commands refused: " + inhibit_reason_;
  } else if (status != 0) {
    // The servo has tripped its own protection and unloaded torque.
    st.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    st.message = "CUTOFF - servo protection tripped: " + flags;
  } else if (comms_lost) {
    st.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    st.message = "CUTOFF - no reply from the servo; the readings below are the last ones seen";
  } else if (!dangers.empty()) {
    st.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    st.message = "DANGER - ";
    for (size_t i = 0; i < dangers.size(); ++i) {
      st.message += (i ? "; " : "") + dangers[i];
    }
  } else {
    st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    st.message = moving ? "moving" : "holding";
  }

  auto kv = [&st](const std::string & key, const std::string & value) {
    diagnostic_msgs::msg::KeyValue e;
    e.key = key;
    e.value = value;
    st.values.push_back(e);
  };

  // Pose, three ways: the joint value MoveIt plans in, the jaw opening an
  // operator can compare against the object in front of the camera, and the
  // raw servo step the bench scripts speak.
  kv("position_rad", num("%.4f", q));
  kv("position_deg", num("%.1f", q * 180.0 / M_PI));
  kv("gap_mm", num("%.1f", gap_mm));
  kv("position_steps", std::to_string(state_steps_.load()));
  kv("command_rad", num("%.4f", cmd_pos_.load()));
  kv("command_gap_mm", num("%.1f", JAW_MM_PER_RAD * (max_pos_ - cmd_pos_.load())));
  // What was actually written, which is NOT the command whenever the slew
  // limiter or squeeze-relax is in the way. The gap between the two is the
  // whole reason a stalled close does not look like a fault.
  kv("goal_written_rad", num("%.4f", written_goal_.load()));
  kv("velocity_rad_s", num("%.3f", state_vel_.load()));
  kv("moving", moving ? "true" : "false");
  kv("squeeze_relax_active", relaxing_.load() ? "true" : "false");

  kv("current_ma", num("%.0f", state_current_ma_.load()));
  kv("voltage_v", num("%.1f", volt));
  kv("temperature_c", num("%.0f", temp));
  kv("load_percent", num("%.0f", 100.0 * state_load_frac_.load()));
  kv("effort_nm", num("%.3f", state_eff_.load()));

  char hex[8];
  std::snprintf(hex, sizeof(hex), "0x%02X", status);
  kv("status_byte", hex);
  kv("status_flags", flags.empty() ? "none" : flags);

  // The limits every DANGER line above is measured against, so the display can
  // be read without knowing what the servo was configured with.
  kv("limit_temperature_c", max_temp_c_ > 0 ? num("%.0f", max_temp_c_) : "unknown");
  kv("limit_voltage_min_v", min_volt_v_ > 0 ? num("%.1f", min_volt_v_) : "unknown");
  kv("limit_voltage_max_v", max_volt_v_ > 0 ? num("%.1f", max_volt_v_) : "unknown");
  kv("limit_current_ma", protect_current_ma_ > 0 ? num("%.0f", protect_current_ma_) : "unknown");

  kv("commands_inhibited", inhibited ? "true" : "false");
  if (inhibited) { kv("inhibit_reason", inhibit_reason_); }
  kv("comms", comms_lost ? "no reply" : "ok");
  kv("read_failures", std::to_string(read_failures_.load()));
  kv("bad_checksums", std::to_string(bus_.bad_checksums()));
  kv("timeouts", std::to_string(bus_.timeouts()));

  diagnostic_msgs::msg::DiagnosticArray msg;
  msg.header.stamp = tel_node_->now();
  msg.status.push_back(st);
  diag_pub_->publish(msg);
}

void ST3215GripperSystem::io_loop()
{
  using clock = std::chrono::steady_clock;
  const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, io_rate_hz_));
  auto next = clock::now();

  int last_sent = INT32_MIN;
  auto last_refresh = clock::now();
  auto stalled_since = clock::time_point::min();
  // Not time_point::min(): differencing that against now() overflows the
  // duration. A bool costs nothing and says what is meant.
  auto last_telemetry = clock::now();
  bool telemetry_sent = false;
  uint64_t consecutive_failures = 0;

  while (running_.load()) {
    next += std::chrono::duration_cast<clock::duration>(period);

    // ---- read -------------------------------------------------------
    // Position, speed, load, voltage, temperature, status, moving and current
    // are consecutive registers, so ONE block read costs one round trip
    // instead of six. See TELEMETRY_BLOCK_LEN.
    uint8_t blk[TELEMETRY_BLOCK_LEN];
    double measured = state_pos_.load();
    if (bus_.read_block(servo_id_, TELEMETRY_BLOCK_START, TELEMETRY_BLOCK_LEN, blk)) {
      consecutive_failures = 0;
      auto u16 = [&blk](uint8_t reg) {
        return static_cast<uint16_t>(blk[tel(reg)] | (blk[tel(reg) + 1] << 8));
      };
      const uint16_t raw_pos = u16(REG_PRESENT_POSITION);
      const uint16_t raw_spd = u16(REG_PRESENT_SPEED);
      const uint16_t raw_load = u16(REG_PRESENT_LOAD);
      const uint16_t raw_curr = u16(REG_PRESENT_CURRENT);

      measured = rad_from_steps(static_cast<int>(raw_pos));
      const double dir = invert_ ? -1.0 : 1.0;
      state_pos_.store(measured);
      state_steps_.store(static_cast<int>(raw_pos));
      state_vel_.store(dir * sign_magnitude(raw_spd, 15) / STEPS_PER_RAD);
      // Present load is a 0-1000 PWM duty with its sign at bit 10, NOT a
      // torque reading. Scaled by the datasheet stall torque it is a usable
      // proxy for grip effort and nothing more; do not calibrate against it.
      const double load_frac = sign_magnitude(raw_load, 10) / LOAD_FULL_SCALE;
      state_load_frac_.store(dir * load_frac);
      state_eff_.store(dir * load_frac * 2.94);
      // Current is a real measurement in mA, and unlike load it is the number
      // the servo's own overcurrent protection acts on.
      state_current_ma_.store(sign_magnitude(raw_curr, 15) * CURRENT_LSB_MA);
      state_volt_.store(blk[tel(REG_PRESENT_VOLTAGE)] * VOLTAGE_LSB_V);
      state_temp_.store(blk[tel(REG_PRESENT_TEMPERATURE)]);
      state_status_.store(blk[tel(REG_STATUS)]);
      state_moving_.store(blk[tel(REG_MOVING)] != 0);
      have_state_.store(true);
    } else {
      ++consecutive_failures;
      read_failures_.fetch_add(1);
    }
    consecutive_failures_.store(static_cast<uint32_t>(consecutive_failures));

    // ---- telemetry --------------------------------------------------
    // BEFORE the inhibit and no-command early exits below, not after: an
    // inhibited gripper is exactly when somebody is looking at the display to
    // find out what is wrong with it.
    if (diag_pub_ && telemetry_rate_hz_ > 0.0) {
      const auto since = std::chrono::duration<double>(clock::now() - last_telemetry).count();
      if (!telemetry_sent || since >= 1.0 / telemetry_rate_hz_) {
        last_telemetry = clock::now();
        telemetry_sent = true;
        publish_telemetry();
      }
    }

    // ---- write ------------------------------------------------------
    if (command_inhibited_.load()) {
      // Say it again on a throttle. A gripper that reports a position but
      // ignores every command is otherwise indistinguishable from a working
      // one right up until the moment it is needed.
      RCLCPP_ERROR_THROTTLE(logger_, *clock_, 5000,
                            "GRIPPER DISABLED, no command is being sent: %s",
                            inhibit_reason_.c_str());
      std::this_thread::sleep_until(next);
      if (clock::now() - next > std::chrono::seconds(1)) { next = clock::now(); }
      continue;
    }
    // Until a controller has actually written a command, send NOTHING. The
    // goal was set to the measured position in on_activate(), so the servo is
    // already holding station; writing a clamped cmd_pos_ here would drag a
    // gripper parked outside the command band to its edge at every startup.
    if (!have_command_.load()) {
      std::this_thread::sleep_until(next);
      if (clock::now() - next > std::chrono::seconds(1)) { next = clock::now(); }
      continue;
    }
    double goal = std::clamp(cmd_pos_.load(), min_pos_, max_pos_);

    // Squeeze-relax: if the servo has been commanded past what it can reach
    // and has stopped trying to get there, it is holding an object and
    // grinding. Re-aim just past where it actually is.
    bool relaxing = false;
    if (squeeze_relax_ && have_state_.load()) {
      const bool pressing = (goal - measured) > stall_error_rad_;   // +q closes
      const bool still = std::fabs(state_vel_.load()) < stall_speed_rad_;
      if (pressing && still) {
        if (stalled_since == clock::time_point::min()) { stalled_since = clock::now(); }
        const double held = std::chrono::duration<double>(clock::now() - stalled_since).count();
        if (held > stall_hold_s_) {
          goal = std::min(goal, measured + relax_bias_rad_);
          relaxing = true;
        }
      } else {
        stalled_since = clock::time_point::min();
      }
    }
    relaxing_.store(relaxing);

    // Rate-limit the goal we actually write. See max_slew_rad_per_s_.
    if (!slew_primed_) { slewed_goal_ = have_state_.load() ? measured : goal; slew_primed_ = true; }
    const double max_delta = max_slew_rad_per_s_ / std::max(1.0, io_rate_hz_);
    slewed_goal_ += std::clamp(goal - slewed_goal_, -max_delta, max_delta);
    goal = slewed_goal_;
    written_goal_.store(goal);

    const int steps = std::clamp(steps_from_rad(goal), POS_MIN, POS_MAX);
    const auto now = clock::now();
    const bool stale = std::chrono::duration<double>(now - last_refresh).count() > 0.5;
    if (steps != last_sent || stale) {
      // Goal speed goes with the goal: writing position alone leaves whatever
      // speed was last set, and on a fresh servo that is 0, which on STS means
      // "as fast as possible".
      if (goal_speed_ > 0) {
        bus_.write16(servo_id_, REG_GOAL_SPEED, static_cast<uint16_t>(goal_speed_));
      }
      if (bus_.write16(servo_id_, REG_GOAL_POSITION, static_cast<uint16_t>(steps))) {
        last_sent = steps;
        last_refresh = now;
      }
    }

    if (consecutive_failures == 50) {
      RCLCPP_WARN(logger_,
                  "no reply from servo %u for 50 cycles (%llu bad checksums, %llu timeouts "
                  "since start). Holding the last known position.",
                  servo_id_,
                  static_cast<unsigned long long>(bus_.bad_checksums()),
                  static_cast<unsigned long long>(bus_.timeouts()));
    }

    std::this_thread::sleep_until(next);
    if (clock::now() - next > std::chrono::seconds(1)) { next = clock::now(); }
  }
}

hardware_interface::return_type
ST3215GripperSystem::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  // Never hand the controller a position that the servo has not confirmed.
  // Until the first successful poll the command is echoed, which is what the
  // mock and Teensy backends do and what keeps MoveIt's first plan sane.
  pos_iface_ = have_state_.load() ? state_pos_.load() : cmd_pos_.load();
  vel_iface_ = have_state_.load() ? state_vel_.load() : 0.0;
  eff_iface_ = state_eff_.load();
  last_reported_pos_ = pos_iface_;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type
ST3215GripperSystem::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (std::isfinite(cmd_iface_)) {
    const double want = std::clamp(cmd_iface_, min_pos_, max_pos_);
    // A controller echoing back the position we seeded is not a command. Only
    // a value that actually differs releases the io thread to start writing,
    // so simply starting the stack never moves the gripper.
    if (!have_command_.load() && std::fabs(cmd_iface_ - state_pos_.load()) > 1e-6) {
      have_command_.store(true);
    }
    cmd_pos_.store(want);
  }
  return hardware_interface::return_type::OK;
}

}  // namespace st3215_gripper_hardware

PLUGINLIB_EXPORT_CLASS(st3215_gripper_hardware::ST3215GripperSystem,
                       hardware_interface::SystemInterface)

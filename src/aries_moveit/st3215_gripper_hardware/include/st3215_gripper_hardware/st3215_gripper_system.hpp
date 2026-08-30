#pragma once

// ros2_control system for the SECONDARY (rack-and-pinion) gripper, whose
// ST3215 bus servo hangs off the rover PC's own USB adapter rather than off
// the Teensy.
//
// WHAT MAKES THIS DIFFERENT FROM teensy_gripper_hardware
//
//   * The joint value IS the servo shaft angle.  The servo direct-drives the
//     pinion, so there is no gearbox, no normalisation and no calibration
//     curve: q [rad] maps to steps by one multiply.  The Teensy gripper takes
//     a normalised 0-1 float and the mapping lives in firmware.
//
//   * There is real position feedback.  The Teensy gripper has no position
//     sensor at all - its /joint_states echoes the command back, which is why
//     every stall and contact check on that gripper is sim-only and why an
//     empty close reports as a success.  This servo reports present position,
//     speed and load, so the state interfaces here are measurements.
//
//   * It owns the serial port.  The I/O runs on its own thread rather than in
//     read()/write(), because those are called from the controller manager's
//     80 Hz loop that also drives the ARM: a servo that stops answering must
//     not be able to stall the arm's control cycle.
//
// SERVO ZERO, AND THE OFFSET THIS GRIPPER DEPENDS ON
//
// MEASURED 2026-08-29 on the assembled gripper, torque off, hand-swept stop to
// stop in one motion with the position unwrapped: full close 3520, full open
// 489, 3031 steps of travel.  So closed_steps 3520, invert FALSE (opening
// DECREASES the count), open end 541 as commanded with 50 steps held off the
// stop.  Those are the launch defaults.
//
// THE SERVO CARRIES A POSITION-CORRECTION OFFSET AND THIS CALIBRATION NEEDS IT.
// Register 31 is set to 1232 (85 as delivered).  Without it the stroke straddles
// the 4095/0 encoder seam, and a single-turn servo in position mode CANNOT hold
// a goal across that: told to move from one side to the other it takes the
// direct numeric path, which is the long way round - backwards through the whole
// mechanism at full torque.  No value of closed_steps or invert fixes it, and it
// is not something this component can work around, because the servo chooses the
// path.  If the servo is swapped or factory-reset, set reg31 again first.
//
// The stroke is 3056 steps at the joint limit out of a usable 3995 (50..4045),
// so the zero is not free to sit anywhere: closed_steps must be at least 3106
// with invert false.  on_activate() checks this and INHIBITS commanding if it
// does not hold, because the failure otherwise looks like the gripper stopping
// mid-open for no reason.
//
// To re-measure, torque off and sweep by hand:
//
//     python3 scripts/st3215_test.py --monitor
//
// and read the step at each stop.  THE STEP NUMBERS ALONE CANNOT TELL YOU WHICH
// STOP IS WHICH - both assignments are valid arithmetic and leave the same
// margin - so pair each reading with a look at the jaws.  Getting it backwards
// yields a gripper that opens on close and closes on open, with nothing in any
// log to say so.  It happened once here, on the strength of an apparent button
// inversion that turned out to be the stale-goal startup snap below.
//
// MODE IS NOT WRITTEN HERE.  REG_MODE is an EPROM register: writing it wears
// the cell and persists across power cycles.  on_activate() READS it, and if
// the servo is in wheel mode it INHIBITS commanding - see below.  In wheel mode
// goal positions are silently ignored while REG_GOAL_SPEED becomes a continuous
// velocity command, so writing this component's normal goal pair would spin the
// pinion into its end stop rather than do nothing.  Fix it once with
// `python3 scripts/st3215_test.py --mode position`.
//
// A GRIPPER FAULT MUST NOT TAKE THE ARM DOWN.
//
// Returning ERROR from on_activate() does exactly that: ros2_control throws
// `Failed to set the initial state of the component` out of the controller
// manager's constructor, the WHOLE ros2_control_node dies, and with it the arm
// trajectory controller, the joint state broadcaster and every consumer of
// /joint_states.  Observed on the rover: one servo left in wheel mode from a
// bench session left the operator with no arm, no gripper and a spawner looping
// on "waiting for service /controller_manager/list_controllers".
//
// So every fault that would make commanding unsafe or useless now ACTIVATES
// and latches `command_inhibited_` instead.  Inhibited means: keep reading and
// reporting the true position, never write a goal, and log the reason and the
// fix on a throttle so it cannot be mistaken for a working gripper.  The arm
// keeps working.

// LIVE TELEMETRY
//
// The servo reports its own supply voltage, winding temperature, current and
// protection status, and it is the ONLY part of the arm that does. None of
// that fits through a ros2_control state interface that a stock broadcaster
// would publish, so this component owns a small rclcpp::Node and publishes a
// diagnostic_msgs/DiagnosticArray on /diagnostics at telemetry_rate_hz.
// scripts/gripper_status_overlay.py in aries_moveit turns that into the RViz
// text overlay; anything else that wants the numbers (a checker, a log) reads
// the same topic.
//
// The DANGER and CUTOFF thresholds are the SERVO'S OWN, read out of EPROM at
// activation (max temperature, max/min input voltage, protection current) -
// they are not constants invented here, because the servo unloads torque on
// its own limits and not on ours. DANGER is a margin short of one of them;
// CUTOFF is REG_STATUS actually non-zero, which means the servo has already
// tripped, or this component having inhibited commanding.

#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>

#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "st3215_gripper_hardware/sts_bus.hpp"

namespace st3215_gripper_hardware
{

class ST3215GripperSystem : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(ST3215GripperSystem)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  hardware_interface::return_type read(const rclcpp::Time &, const rclcpp::Duration &) override;
  hardware_interface::return_type write(const rclcpp::Time &, const rclcpp::Duration &) override;

private:
  void io_loop();
  int steps_from_rad(double q) const;
  double rad_from_steps(int steps) const;
  /// Latch a fault that makes commanding unsafe or pointless. Activation still
  /// succeeds so the rest of the stack survives; nothing is ever written.
  void inhibit(const std::string & reason);
  /// Finish activation after inhibit(): keep reading if the port is open, but
  /// never write. Returns SUCCESS so the controller manager survives.
  hardware_interface::CallbackReturn start_inhibited();
  /// After a failed ping: sweep the bus for any servo, at this baud and then at
  /// the other common ones, and describe what was found. Turns "no reply from
  /// id 1" into something that says whether the servo is at another ID, another
  /// baud, or not powered at all.
  std::string probe_bus();
  /// Read the servo's own protection limits out of EPROM. Best effort: a limit
  /// that does not answer is left unknown and the display says so rather than
  /// comparing against a guess.
  void read_protection_limits();
  /// One DiagnosticArray on /diagnostics. Called from the io thread, including
  /// while inhibited - a gripper that refuses commands is precisely when the
  /// operator needs to see why.
  void publish_telemetry();

  std::string joint_name_;

  // --- parameters -----------------------------------------------------
  std::string port_{"/dev/aries_servo_bus"};
  int baud_{1000000};
  uint8_t servo_id_{1};
  int closed_steps_{3000};
  bool invert_{false};
  double min_pos_{-4.065};
  double max_pos_{0.07};
  int accel_{0};              // 0 = leave the servo's stored value alone
  int goal_speed_{2000};
  int torque_limit_{0};       // 0 = do not write
  double io_rate_hz_{100.0};
  // SLEW LIMIT on the goal actually written to the servo, rad/s.
  //
  // Nothing else bounds how far one write can move this gripper. A wrong
  // closed_steps, a mis-parsed invert, or a bench script writing a raw goal can
  // all ask for a jump of most of a revolution, and the servo will take it at
  // full speed - which on a rack and pinion means driving the jaws through
  // their stops. That happened on the bench: a goal write produced ~960 steps
  // of travel per command and wrapped the encoder, and the pinion appears to
  // have skipped against the racks.
  //
  // 3.0 rad/s still crosses the whole 1.87 rad stroke in 0.62 s, so it costs
  // nothing in normal use; it only removes the ability to make one enormous
  // uncommanded move. The controller's own trajectory is unaffected - it never
  // asks for anything this fast.
  double max_slew_rad_per_s_{3.0};
  double slewed_goal_{0.0};
  bool slew_primed_{false};
  // 25 ms, NOT the 1 ms a 1 Mbaud round trip suggests. The wire time for a
  // 6-byte request and a 6-byte reply is ~120 us, but this is a USB CDC-ACM
  // bridge (a CH343), and the host schedules bulk transfers when it feels like
  // it - a few milliseconds is normal and an occasional scheduling hiccup is
  // longer. scripts/st3215_test.py, which works, allows 50 ms.
  //
  // This started at 5 ms and that was wrong: it answered on one run and not the
  // next, which is exactly how a marginal deadline shows up. It costs nothing
  // when the servo is healthy, because a read returns as soon as the bytes
  // arrive; it is only the ceiling.
  int timeout_ms_{25};

  // --- squeeze-relax --------------------------------------------------
  // A closed command drives to gap 0 by design, so gripping anything leaves a
  // permanent position error the servo answers with full torque.  On the v2
  // gripper a 30 mm probe leaves about 20 deg of unclosable error, and the
  // servo sits there stalled and hot.  When the joint has been commanded past
  // where it can actually reach and has stopped moving, back the goal off to
  // the measured position plus a small bias: the same grip force, a fraction
  // of the current, and the SEARCH is untouched - only the hold is.
  bool squeeze_relax_{true};
  double stall_error_rad_{0.03};
  double stall_speed_rad_{0.05};
  double stall_hold_s_{0.35};
  double relax_bias_rad_{0.015};

  // --- telemetry ------------------------------------------------------
  bool publish_diagnostics_{true};
  double telemetry_rate_hz_{5.0};
  // How close to the servo's OWN limit counts as danger. Small enough that
  // "danger" means the trip is close, wide enough to give the operator time
  // to back off: the ST3215 climbs roughly 1 deg C every few seconds when
  // stalled, so 8 deg C is tens of seconds of warning.
  double warn_temp_margin_c_{8.0};
  double warn_volt_margin_v_{0.5};
  double warn_current_frac_{0.8};
  // Sustained load with no motion, as a fraction of the 0-1000 duty. Not a
  // servo limit - it is the squeeze-relax condition seen from outside, and it
  // is the state a gripper holding an object actually sits in.
  double warn_load_frac_{0.7};

  // Servo protection limits, read from EPROM at activation. Negative means the
  // read failed and nothing is compared against it.
  double max_temp_c_{-1.0};
  double max_volt_v_{-1.0};
  double min_volt_v_{-1.0};
  double protect_current_ma_{-1.0};

  // --- shared with the I/O thread -------------------------------------
  std::atomic<double> cmd_pos_{0.0};
  std::atomic<double> state_pos_{0.0};
  std::atomic<double> state_vel_{0.0};
  std::atomic<double> state_eff_{0.0};
  std::atomic<int> state_steps_{0};
  std::atomic<double> state_load_frac_{0.0};   // signed, -1..1 of full duty
  std::atomic<double> state_volt_{0.0};
  std::atomic<double> state_temp_{0.0};
  std::atomic<double> state_current_ma_{0.0};
  std::atomic<uint8_t> state_status_{0};
  std::atomic<bool> state_moving_{false};
  std::atomic<bool> relaxing_{false};
  std::atomic<double> written_goal_{0.0};
  std::atomic<bool> have_state_{false};
  std::atomic<bool> command_inhibited_{false};
  // False until a controller writes a command that differs from the position
  // seeded at activation. While false the io thread writes nothing at all, so
  // bringing the stack up cannot move the gripper.
  std::atomic<bool> have_command_{false};
  std::string inhibit_reason_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> read_failures_{0};
  // Reads failed back to back. The total above says how flaky the link has
  // ever been; this says whether the servo is answering RIGHT NOW, which is
  // what separates "warm" from "unplugged" on the display.
  std::atomic<uint32_t> consecutive_failures_{0};

  StsBus bus_;
  std::thread io_thread_;
  // Our own node, because a hardware component is not given one in Jazzy. It
  // is never spun: a publisher works without an executor, and this component
  // has nothing to receive.
  rclcpp::Node::SharedPtr tel_node_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::Logger logger_{rclcpp::get_logger("ST3215GripperSystem")};
  // RCLCPP_*_THROTTLE needs a clock it can keep state against, and the io
  // thread has no node to borrow one from.
  std::shared_ptr<rclcpp::Clock> clock_{std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME)};

  // StateInterface and CommandInterface hold raw double pointers, so the
  // atomics above cannot be exported directly. read()/write() copy across
  // once per control cycle, which is also the only place the two sides meet.
  double pos_iface_{0.0};
  double vel_iface_{0.0};
  double eff_iface_{0.0};
  // NaN, NOT zero, and write() drops any non-finite value. An unclaimed
  // command interface - the window between activation and a controller being
  // spawned, or any period with no controller running - otherwise hands
  // write() a perfectly finite 0.0 that nothing ever asked for. Measured: the
  // jaws walked from closed to q = 0 on their own with no controller loaded.
  // on_activate() overwrites this with the servo's measured position.
  double cmd_iface_{std::numeric_limits<double>::quiet_NaN()};
  double last_reported_pos_{0.0};
};

}  // namespace st3215_gripper_hardware

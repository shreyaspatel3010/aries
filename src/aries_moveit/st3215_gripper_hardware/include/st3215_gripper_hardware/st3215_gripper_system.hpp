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
// SERVO ZERO, AND WHY IT IS A PARAMETER
//
// The gripper needs 4.135 rad (236.9 deg, 2695 steps) to go from closed to
// fully open, out of the servo's 360 deg single-turn range.  It fits, but only
// with the zero placed deliberately.  ``closed_steps`` is the raw step count at
// q = +0.07 rad, jaws touching.  To calibrate:
//
//     python3 scripts/st3215_test.py --monitor      # torque off, back-drive
//     close the jaws by hand until they just touch, read the step count,
//     put that number in closed_steps.
//
// Check it: closed_steps - 2695 must stay above 50 with invert false (or
// closed_steps + 2695 below 4045 with invert true), or the servo runs into its
// own encoder seam partway through the stroke.  on_activate() checks this and
// INHIBITS commanding if it does not hold (see below), because the failure
// otherwise appears as the gripper stopping mid-open for no visible reason.
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

#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <thread>

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
  int timeout_ms_{5};

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

  // --- shared with the I/O thread -------------------------------------
  std::atomic<double> cmd_pos_{0.0};
  std::atomic<double> state_pos_{0.0};
  std::atomic<double> state_vel_{0.0};
  std::atomic<double> state_eff_{0.0};
  std::atomic<bool> have_state_{false};
  std::atomic<bool> command_inhibited_{false};
  std::string inhibit_reason_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> read_failures_{0};

  StsBus bus_;
  std::thread io_thread_;
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

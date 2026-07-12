#pragma once

#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <std_msgs/msg/float32.hpp>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace teensy_gripper_hardware
{

class TeensyGripperSystem : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(TeensyGripperSystem)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  hardware_interface::return_type read(const rclcpp::Time &, const rclcpp::Duration &) override;
  hardware_interface::return_type write(const rclcpp::Time &, const rclcpp::Duration &) override;

private:
  std::string joint_name_;
  double min_pos_{0.0};
  double max_pos_{0.08925};

  double cmd_pos_{0.0};
  double state_pos_{0.0};
  double state_vel_{0.0};

  // Dead-band: track last written position to skip redundant sends
  double last_written_pos_{-1.0};

  // Anti-backtrack filter: prevents sudden large backward cmd jumps from
  // reaching the servo (e.g., when MoveIt re-plans from a stale planning-scene
  // state and sends a trajectory that starts near-open while the gripper is
  // closed).  servo_pos_ is the filtered value actually sent to the Teensy and
  // fed back to the planning scene via state_pos_.
  double servo_pos_{0.0};
  bool   backtrack_detected_{false};
  double backtrack_hold_pos_{0.0};

  rclcpp::Logger logger_{rclcpp::get_logger("TeensyGripperSystem")};

  // micro-ROS bridge: topic names (configurable via URDF hardware params)
  std::string cmd_topic_{"/gripper/cmd"};
  std::string state_topic_{"/gripper/state"};

  // ROS 2 communication node (runs in a background thread)
  rclcpp::Node::SharedPtr comm_node_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr cmd_pub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr state_sub_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  std::thread spin_thread_;

  // Latest normalized state received from Teensy via /gripper/state
  std::atomic<float> latest_state_{0.0f};
  std::atomic<bool>  state_received_{false};
  // Timestamp of the last received state message (for staleness check)
  std::atomic<std::chrono::steady_clock::time_point::rep> last_state_ns_{0};
  // Timestamp of last periodic-retry check (for dropped-command recovery)
  std::atomic<std::chrono::steady_clock::time_point::rep> last_resend_check_ns_{0};
  // Timestamp of last publish (keepalive — prevents servo idle-detach)
  std::atomic<std::chrono::steady_clock::time_point::rep> last_publish_ns_{0};
};
}  // namespace teensy_gripper_hardware
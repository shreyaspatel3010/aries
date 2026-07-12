#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

namespace
{
const size_t ROS_QUEUE_SIZE = 10;
const size_t TRAJECTORY_QUEUE_SIZE = 1;

template<typename T>
T getOrDeclareParameter(const rclcpp::Node::SharedPtr &node, const std::string &name, const T &default_value)
{
  if (!node->has_parameter(name))
  {
    return node->declare_parameter<T>(name, default_value);
  }

  T value;
  node->get_parameter(name, value);
  return value;
}

std::string contactPairString(const collision_detection::CollisionResult &result)
{
  if (result.contacts.empty())
  {
    return "unknown links";
  }

  const auto &first_pair = *result.contacts.begin();
  return first_pair.first.first + " <-> " + first_pair.first.second;
}
}  // namespace

class ServoCollisionGuard
{
public:
  ServoCollisionGuard()
  {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    nh_ = std::make_shared<rclcpp::Node>("servo_collision_guard", options);

    input_topic_ = getOrDeclareParameter<std::string>(nh_, "input_topic", "servo_guard/input_joint_trajectory");
    output_topic_ = getOrDeclareParameter<std::string>(
      nh_,
      "output_topic",
      "rebel_arm_trajectory_controller/joint_trajectory");
    joint_state_topic_ = getOrDeclareParameter<std::string>(nh_, "joint_state_topic", "joint_states");
    status_topic_ = getOrDeclareParameter<std::string>(nh_, "status_topic", "/arm_joystick/status");
    group_name_ = getOrDeclareParameter<std::string>(nh_, "group_name", "arm_with_gripper");
    min_self_distance_ = getOrDeclareParameter<double>(nh_, "min_self_distance", 0.015);
    distance_tolerance_ = getOrDeclareParameter<double>(nh_, "distance_tolerance", 0.001);
    interpolation_steps_ = getOrDeclareParameter<int>(nh_, "interpolation_steps", 6);
    hold_time_ = getOrDeclareParameter<double>(nh_, "hold_time", 0.05);

    min_self_distance_ = std::max(0.0, min_self_distance_);
    distance_tolerance_ = std::max(0.0, distance_tolerance_);
    interpolation_steps_ = std::max(1, interpolation_steps_);
    hold_time_ = std::max(0.01, hold_time_);

    robot_model_loader::RobotModelLoader loader(nh_, "robot_description");
    robot_model_ = loader.getModel();
    if (!robot_model_)
    {
      throw std::runtime_error("ServoCollisionGuard failed to load robot_description");
    }
    const auto &variable_names = robot_model_->getVariableNames();
    variable_names_.insert(variable_names.begin(), variable_names.end());

    planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_);
    current_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
    current_state_->setToDefaultValues();
    current_state_->update(true);

    trajectory_pub_ = nh_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      output_topic_, TRAJECTORY_QUEUE_SIZE);
    status_pub_ = nh_->create_publisher<std_msgs::msg::String>(status_topic_, ROS_QUEUE_SIZE);

    joint_state_sub_ = nh_->create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_, ROS_QUEUE_SIZE,
      std::bind(&ServoCollisionGuard::jointStateCallback, this, std::placeholders::_1));

    trajectory_sub_ = nh_->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      input_topic_, TRAJECTORY_QUEUE_SIZE,
      std::bind(&ServoCollisionGuard::trajectoryCallback, this, std::placeholders::_1));

    publishStatus("Servo collision guard ready");
    RCLCPP_INFO(
      nh_->get_logger(),
      "Servo collision guard forwarding '%s' -> '%s' with %.3f m self-distance margin",
      input_topic_.c_str(), output_topic_.c_str(), min_self_distance_);
  }

  void spin()
  {
    rclcpp::spin(nh_);
  }

private:
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i)
    {
      if (isKnownVariable(msg->name[i]))
      {
        current_state_->setVariablePosition(msg->name[i], msg->position[i]);
      }
    }
    current_state_->update(true);
    have_state_ = true;
  }

  void trajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
  {
    moveit::core::RobotState state(robot_model_);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_state_)
      {
        RCLCPP_WARN_THROTTLE(
          nh_->get_logger(), *nh_->get_clock(), 2000,
          "Blocking Servo trajectory: no joint_states received yet");
        return;
      }
      state = *current_state_;
    }

    std::string reason;
    if (!trajectoryIsSafe(*msg, state, reason))
    {
      publishHoldTrajectory(*msg, state);
      publishStatus("Arm command blocked: " + reason);
      RCLCPP_WARN_THROTTLE(
        nh_->get_logger(), *nh_->get_clock(), 1000,
        "Blocked Servo trajectory: %s", reason.c_str());
      return;
    }

    trajectory_msgs::msg::JointTrajectory command = *msg;
    command.header.stamp = nh_->now();
    trajectory_pub_->publish(command);
  }

  bool trajectoryIsSafe(
    const trajectory_msgs::msg::JointTrajectory &trajectory,
    const moveit::core::RobotState &start_state,
    std::string &reason) const
  {
    if (trajectory.joint_names.empty() || trajectory.points.empty())
    {
      reason = "empty Servo trajectory";
      return false;
    }

    moveit::core::RobotState previous_state(start_state);
    double previous_distance = selfDistance(previous_state);

    for (size_t point_index = 0; point_index < trajectory.points.size(); ++point_index)
    {
      const auto &point = trajectory.points[point_index];
      if (point.positions.size() != trajectory.joint_names.size())
      {
        reason = "Servo trajectory missing joint positions";
        return false;
      }

      moveit::core::RobotState target_state(previous_state);
      for (size_t joint_index = 0; joint_index < trajectory.joint_names.size(); ++joint_index)
      {
        const std::string &joint_name = trajectory.joint_names[joint_index];
        if (!isKnownVariable(joint_name))
        {
          reason = "unknown joint '" + joint_name + "'";
          return false;
        }
        target_state.setVariablePosition(joint_name, point.positions[joint_index]);
      }
      target_state.update(true);

      for (int step = 1; step <= interpolation_steps_; ++step)
      {
        const double t = static_cast<double>(step) / static_cast<double>(interpolation_steps_);
        moveit::core::RobotState sample_state(robot_model_);
        previous_state.interpolate(target_state, t, sample_state);
        sample_state.update(true);

        if (!stateIsSafe(sample_state, previous_distance, reason))
        {
          reason += " at Servo point " + std::to_string(point_index);
          return false;
        }

        previous_distance = selfDistance(sample_state);
      }

      previous_state = target_state;
    }

    return true;
  }

  bool stateIsSafe(
    const moveit::core::RobotState &state,
    double previous_distance,
    std::string &reason) const
  {
    collision_detection::CollisionRequest collision_request;
    collision_request.group_name = group_name_;
    collision_request.contacts = true;
    collision_request.max_contacts = 1;
    collision_request.max_contacts_per_pair = 1;

    collision_detection::CollisionResult collision_result;
    planning_scene_->checkSelfCollision(
      collision_request, collision_result, state, planning_scene_->getAllowedCollisionMatrix());

    if (collision_result.collision)
    {
      reason = "self collision " + contactPairString(collision_result);
      return false;
    }

    const double distance = selfDistance(state);
    if (!std::isfinite(distance))
    {
      return true;
    }

    const bool entering_safety_margin =
      distance < min_self_distance_ && distance < previous_distance - distance_tolerance_;
    if (entering_safety_margin)
    {
      std::ostringstream stream;
      stream << "self collision margin " << distance << " m < " << min_self_distance_ << " m";
      reason = stream.str();
      return false;
    }

    return true;
  }

  double selfDistance(const moveit::core::RobotState &state) const
  {
    collision_detection::DistanceRequest distance_request;
    collision_detection::DistanceResult distance_result;
    distance_request.group_name = group_name_;
    distance_request.acm = &planning_scene_->getAllowedCollisionMatrix();
    distance_request.distance_threshold = min_self_distance_;
    distance_request.enable_signed_distance = true;
    distance_request.enableGroup(robot_model_);

    planning_scene_->getCollisionEnv()->distanceSelf(distance_request, distance_result, state);
    return distance_result.minimum_distance.distance;
  }

  bool isKnownVariable(const std::string &name) const
  {
    return variable_names_.find(name) != variable_names_.end();
  }

  void publishHoldTrajectory(
    const trajectory_msgs::msg::JointTrajectory &reference,
    const moveit::core::RobotState &state)
  {
    if (reference.joint_names.empty())
    {
      return;
    }

    trajectory_msgs::msg::JointTrajectory hold;
    hold.header.stamp = nh_->now();
    hold.joint_names = reference.joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.reserve(reference.joint_names.size());
    point.velocities.assign(reference.joint_names.size(), 0.0);
    for (const std::string &joint_name : reference.joint_names)
    {
      if (!isKnownVariable(joint_name))
      {
        return;
      }
      point.positions.push_back(state.getVariablePosition(joint_name));
    }
    point.time_from_start = rclcpp::Duration::from_seconds(hold_time_);
    hold.points.push_back(point);

    trajectory_pub_->publish(hold);
  }

  void publishStatus(const std::string &status) const
  {
    if (!status_pub_)
    {
      return;
    }

    std_msgs::msg::String msg;
    msg.data = status;
    status_pub_->publish(msg);
  }

  rclcpp::Node::SharedPtr nh_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_sub_;

  moveit::core::RobotModelPtr robot_model_;
  planning_scene::PlanningScenePtr planning_scene_;
  moveit::core::RobotStatePtr current_state_;
  std::set<std::string> variable_names_;

  mutable std::mutex state_mutex_;
  bool have_state_ = false;

  std::string input_topic_;
  std::string output_topic_;
  std::string joint_state_topic_;
  std::string status_topic_;
  std::string group_name_;
  double min_self_distance_;
  double distance_tolerance_;
  int interpolation_steps_;
  double hold_time_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try
  {
    ServoCollisionGuard guard;
    guard.spin();
  }
  catch (const std::exception &error)
  {
    RCLCPP_FATAL(rclcpp::get_logger("servo_collision_guard"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/collision_detection/collision_common.hpp>

namespace
{
constexpr size_t ROS_QUEUE_SIZE = 10;

const std::array<std::string, 6> DEFAULT_ARM_JOINTS = {
  "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
};

double clampAbs(double value, double limit)
{
  return std::clamp(value, -std::abs(limit), std::abs(limit));
}
}

class RebelSmoothSafeJoystick
{
public:
  RebelSmoothSafeJoystick()
  {
    nh_ = std::make_shared<rclcpp::Node>("rebel_servo_teleop_gamepad");

    loadParams();

    // Before initMoveItModel(): it reports load failures through publishStatus(),
    // which dereferences status_pub_.
    status_pub_ = nh_->create_publisher<std_msgs::msg::String>(
      status_topic_, ROS_QUEUE_SIZE);

    initMoveItModel();

    arm_pub_ = nh_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      arm_command_topic_, ROS_QUEUE_SIZE);

    gripper_pub_ = nh_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      gripper_command_topic_, ROS_QUEUE_SIZE);

    joy_sub_ = nh_->create_subscription<sensor_msgs::msg::Joy>(
      joy_topic_, ROS_QUEUE_SIZE,
      std::bind(&RebelSmoothSafeJoystick::joyCallback, this, std::placeholders::_1));

    joint_state_sub_ = nh_->create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_, ROS_QUEUE_SIZE,
      std::bind(&RebelSmoothSafeJoystick::jointStateCallback, this, std::placeholders::_1));

    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / std::max(1.0, command_rate_hz_)));

    timer_ = nh_->create_wall_timer(period, std::bind(&RebelSmoothSafeJoystick::timerCallback, this));

    last_joy_time_ = nh_->now();
    last_publish_time_ = nh_->now();
    last_slew_time_ = nh_->now();
    last_gripper_update_ = nh_->now();

    publishStatus(
      "FINAL SAFE SMOOTH ReBeL joystick ready: joystick silent when idle so RViz planner can control arm.  RB toggles Cartesian/Direct-Joint, "
      "80 Hz velocity-only JTC, MoveIt self-collision guard active.");
  }

  void spin()
  {
    rclcpp::spin(nh_);
  }

private:
  enum class Mode
  {
    CARTESIAN,
    DIRECT_JOINT
  };

  template<typename T>
  T declareGet(const std::string &name, const T &default_value)
  {
    nh_->declare_parameter<T>(name, default_value);
    T value = default_value;
    nh_->get_parameter(name, value);
    return value;
  }

  void loadParams()
  {
    joy_topic_ = declareGet<std::string>("joy_topic", "joy");
    joint_state_topic_ = declareGet<std::string>("joint_state_topic", "joint_states");

    arm_command_topic_ = declareGet<std::string>(
      "arm_command_topic", "rebel_arm_trajectory_controller/joint_trajectory");

    gripper_command_topic_ = declareGet<std::string>(
      "gripper_command_topic", "rebel_gripper_controller/joint_trajectory");

    gripper_joint_name_ = declareGet<std::string>(
      "gripper_joint_name", "gripper_gear_left_joint");

    status_topic_ = declareGet<std::string>("status_topic", "/arm_joystick/status");

    planning_group_ = declareGet<std::string>("planning_group", "igus_rebel_arm");
    planning_link_ = declareGet<std::string>("planning_link", "gripper_tcp");

    std::vector<std::string> default_names(DEFAULT_ARM_JOINTS.begin(), DEFAULT_ARM_JOINTS.end());
    auto names = declareGet<std::vector<std::string>>("joint_names", default_names);
    if (names.size() == 6)
    {
      for (size_t i = 0; i < 6; ++i)
      {
        arm_joint_names_[i] = names[i];
      }
    }

    auto default_min = std::vector<double>{-3.1241, -1.4835, -1.39626, -3.12414, -1.65806, -3.12414};
    auto default_max = std::vector<double>{ 3.1241,  2.4435,  2.61799,  3.12414,  1.65806,  3.12414};

    auto mins = declareGet<std::vector<double>>("joint_min_positions", default_min);
    auto maxs = declareGet<std::vector<double>>("joint_max_positions", default_max);
    if (mins.size() == 6 && maxs.size() == 6)
    {
      for (size_t i = 0; i < 6; ++i)
      {
        joint_min_[i] = mins[i];
        joint_max_[i] = maxs[i];
      }
    }

    command_rate_hz_ = declareGet<double>("command_publish_rate_hz", 80.0);
    joy_timeout_sec_ = declareGet<double>("joy_timeout_sec", 0.35);
    deadzone_ = declareGet<double>("deadzone", 0.04);

    linear_scale_ = declareGet<double>("linear_scale", 0.10);
    angular_scale_ = declareGet<double>("angular_scale", 0.18);
    direct_joint_scale_ = declareGet<double>("joint_scale", 0.28);

    max_joint_velocity_ = declareGet<double>("max_joint_velocity", 0.28);
    max_joint_accel_ = declareGet<double>("max_joint_accel", 1.8);
    max_joint_decel_ = declareGet<double>("max_joint_decel", 16.0);
    joint_velocity_deadband_ = declareGet<double>("joint_velocity_deadband", 0.004);

    dls_lambda_ = declareGet<double>("dls_lambda", 0.10);
    joint_limit_margin_ = declareGet<double>("joint_limit_margin", 0.05);
    collision_preview_sec_ = declareGet<double>("collision_preview_sec", 0.16);

    velocity_point_1_sec_ = declareGet<double>("velocity_point_1_sec", 0.040);
    velocity_point_2_sec_ = declareGet<double>("velocity_point_2_sec", 0.080);
    stop_zero_cycles_total_ = declareGet<int>("stop_zero_cycles", 4);

    constant_speed_mode_ = declareGet<bool>("constant_speed_mode", false);

    button_enable_ = declareGet<int>("button_enable", 5);                    // RB
    button_arm_mode_toggle_ = declareGet<int>("button_arm_mode_toggle", 5);  // RB
    button_rover_enable_ = declareGet<int>("button_rover_enable", 4);        // LB
    arm_toggle_mode_ = declareGet<bool>("arm_toggle_mode", true);

    button_gripper_open_ = declareGet<int>("button_gripper_open", 2);
    button_gripper_close_ = declareGet<int>("button_gripper_close", 1);
    button_gripper_toggle_ = declareGet<int>("button_gripper_toggle", 0);

    axis_linear_x_ = declareGet<int>("axis_linear_x", 1);
    axis_linear_y_ = declareGet<int>("axis_linear_y", 0);
    axis_linear_z_ = declareGet<int>("axis_linear_z", 7);
    axis_angular_x_ = declareGet<int>("axis_angular_x", 3);
    axis_angular_y_ = declareGet<int>("axis_angular_y", 4);
    axis_angular_z_ = declareGet<int>("axis_angular_z", 6);

    axis_joint1_ = declareGet<int>("axis_joint1", 0);
    axis_joint2_ = declareGet<int>("axis_joint2", 1);
    axis_joint3_ = declareGet<int>("axis_joint3", 4);
    axis_joint4_ = declareGet<int>("axis_joint4", 3);
    axis_joint5_ = declareGet<int>("axis_joint5", 6);
    axis_joint6_ = declareGet<int>("axis_joint6", 7);

    gripper_speed_ = declareGet<double>("gripper_speed", 2.0);
    gripper_open_position_ = declareGet<double>("gripper_open_position", -1.57);
    gripper_closed_position_ = declareGet<double>("gripper_closed_position", 0.07);
    gripper_trajectory_duration_ = declareGet<double>("gripper_trajectory_duration", 0.10);
    max_gripper_command_step_ = declareGet<double>("max_gripper_command_step", 0.04);

    command_rate_hz_ = std::clamp(command_rate_hz_, 30.0, 100.0);
    deadzone_ = std::clamp(deadzone_, 0.0, 0.25);
    linear_scale_ = std::clamp(linear_scale_, 0.01, 0.20);
    angular_scale_ = std::clamp(angular_scale_, 0.02, 0.80);
    direct_joint_scale_ = std::clamp(direct_joint_scale_, 0.01, 0.60);

    max_joint_velocity_ = std::clamp(max_joint_velocity_, 0.05, 0.60);
    max_joint_accel_ = std::clamp(max_joint_accel_, 0.2, 12.0);
    max_joint_decel_ = std::clamp(max_joint_decel_, 0.2, 30.0);
    joint_velocity_deadband_ = std::clamp(joint_velocity_deadband_, 0.0, 0.05);

    dls_lambda_ = std::clamp(dls_lambda_, 0.001, 1.0);
    joint_limit_margin_ = std::clamp(joint_limit_margin_, 0.0, 0.20);
    collision_preview_sec_ = std::clamp(collision_preview_sec_, 0.04, 0.40);
    velocity_point_1_sec_ = std::clamp(velocity_point_1_sec_, 0.015, 0.10);
    velocity_point_2_sec_ = std::clamp(velocity_point_2_sec_, velocity_point_1_sec_ + 0.01, 0.20);
    stop_zero_cycles_total_ = std::clamp(stop_zero_cycles_total_, 1, 8);

    if (gripper_open_position_ > gripper_closed_position_)
    {
      std::swap(gripper_open_position_, gripper_closed_position_);
    }

    commanded_gripper_position_ = gripper_open_position_;
    current_gripper_position_ = gripper_open_position_;
  }

  void initMoveItModel()
  {
    try
    {
      robot_model_loader_ = std::make_unique<robot_model_loader::RobotModelLoader>(
        nh_, "robot_description");

      robot_model_ = robot_model_loader_->getModel();
      if (!robot_model_)
      {
        publishStatus("MoveIt model unavailable: robot_description not loaded");
        return;
      }

      joint_model_group_ = robot_model_->getJointModelGroup(planning_group_);
      if (!joint_model_group_)
      {
        publishStatus("MoveIt group unavailable: " + planning_group_);
        return;
      }

      tip_link_model_ = robot_model_->getLinkModel(planning_link_);
      if (!tip_link_model_)
      {
        publishStatus("MoveIt tip link unavailable: " + planning_link_);
        return;
      }

      robot_state_ = std::make_unique<moveit::core::RobotState>(robot_model_);
      robot_state_->setToDefaultValues();

      planning_scene_ = std::make_unique<planning_scene::PlanningScene>(robot_model_);

      moveit_ready_ = true;
    }
    catch (const std::exception &e)
    {
      publishStatus(std::string("MoveIt init failed: ") + e.what());
      moveit_ready_ = false;
    }
  }

  void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(joy_mutex_);
    latest_joy_ = *msg;
    have_latest_joy_ = true;
    last_joy_time_ = nh_->now();
  }

  void timerCallback()
  {
    sensor_msgs::msg::Joy msg;
    {
      std::lock_guard<std::mutex> lock(joy_mutex_);

      if (!have_latest_joy_)
      {
        publishZeroIfNeeded();
        return;
      }

      if (joy_timeout_sec_ > 0.0 &&
          (nh_->now() - last_joy_time_).seconds() > joy_timeout_sec_)
      {
        hardZeroStop();
        return;
      }

      msg = latest_joy_;
    }

    processJoy(msg);
  }

  void processJoy(const sensor_msgs::msg::Joy &msg)
  {
    const bool rover_enabled = buttonPressed(msg, button_rover_enable_);
    const bool rb_pressed = buttonPressed(msg, button_enable_);
    const bool rb_toggle_pressed = buttonPressed(msg, button_arm_mode_toggle_);
    const bool rb_rising_edge = rb_toggle_pressed && !previous_rb_toggle_pressed_;

    if (rover_enabled)
    {
      previous_rb_toggle_pressed_ = rb_toggle_pressed;
      previous_gripper_toggle_pressed_ = false;
      hardZeroStop();
      publishStatus("LB rover mode active: arm blocked");
      return;
    }

    if (arm_toggle_mode_ && rb_rising_edge)
    {
      toggleMode();
    }

    previous_rb_toggle_pressed_ = rb_toggle_pressed;

    if (!rb_pressed)
    {
      previous_gripper_toggle_pressed_ = false;
      hardZeroStop();
      return;
    }

    updateGripper(msg);

    std::array<double, 6> qdot{};
    bool have_command = false;

    if (active_mode_ == Mode::CARTESIAN)
    {
      have_command = computeCartesianCommand(msg, qdot);
    }
    else
    {
      have_command = computeDirectJointCommand(msg, qdot);
    }

    if (!have_command || allZero(qdot))
    {
      hardZeroStop();
      return;
    }

    applyJointLimitProtection(qdot);
    limitVelocityAndAcceleration(qdot);

    if (!selfCollisionSafe(qdot))
    {
      std::string reason;

      if (!scaleToSafeVelocity(qdot, reason))
      {
        hardZeroStop();
        publishStatus("MoveIt self-collision guard: motion blocked (" + reason + ")");
        return;
      }
    }

    if (allZero(qdot))
    {
      hardZeroStop();
      return;
    }

    active_motion_ = true;
    stop_zero_cycles_left_ = 0;
    publishVelocityOnlyTrajectory(qdot, false);
  }

  void toggleMode()
  {
    if (!arm_mode_initialized_)
    {
      active_mode_ = Mode::CARTESIAN;
      arm_mode_initialized_ = true;
    }
    else
    {
      active_mode_ = active_mode_ == Mode::CARTESIAN ? Mode::DIRECT_JOINT : Mode::CARTESIAN;
    }

    last_output_vel_.fill(0.0);
    active_motion_ = false;
    stop_zero_cycles_left_ = stop_zero_cycles_total_;
    publishZeroVelocityTrajectory(true);

    if (active_mode_ == Mode::CARTESIAN)
    {
      publishStatus("ARM MODE: Cartesian direct MoveIt IK + self-collision guard, 0.10 m/s");
    }
    else
    {
      publishStatus("ARM MODE: Direct joint jog + self-collision guard");
    }
  }

  bool computeCartesianCommand(const sensor_msgs::msg::Joy &msg, std::array<double, 6> &qdot)
  {
    if (!moveit_ready_ || !have_joint_state_)
    {
      publishStatus("Waiting for MoveIt model and joint_states");
      return false;
    }

    Eigen::Matrix<double, 6, 1> twist;
    twist << linear_scale_ * axisValue(msg, axis_linear_x_),
             linear_scale_ * axisValue(msg, axis_linear_y_),
             linear_scale_ * axisValue(msg, axis_linear_z_),
             angular_scale_ * axisValue(msg, axis_angular_x_),
             angular_scale_ * axisValue(msg, axis_angular_y_),
             angular_scale_ * axisValue(msg, axis_angular_z_);

    if (twist.norm() < 1e-8)
    {
      return false;
    }

    std::array<double, 6> pos{};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      pos = joint_pos_;
    }

    for (size_t i = 0; i < 6; ++i)
    {
      robot_state_->setVariablePosition(arm_joint_names_[i], pos[i]);
    }
    robot_state_->update(true);

    Eigen::MatrixXd jacobian;
    const bool ok = robot_state_->getJacobian(
      joint_model_group_, tip_link_model_, Eigen::Vector3d::Zero(), jacobian);

    if (!ok || jacobian.rows() != 6 || jacobian.cols() < 6)
    {
      publishStatus("Jacobian unavailable");
      return false;
    }

    const Eigen::MatrixXd jj_t =
      jacobian * jacobian.transpose() +
      (dls_lambda_ * dls_lambda_) * Eigen::MatrixXd::Identity(6, 6);

    const Eigen::VectorXd result =
      jacobian.transpose() * jj_t.ldlt().solve(twist);

    if (result.size() < 6 || !result.allFinite())
    {
      return false;
    }

    for (size_t i = 0; i < 6; ++i)
    {
      qdot[i] = result[static_cast<int>(i)];
    }

    return true;
  }

  bool computeDirectJointCommand(const sensor_msgs::msg::Joy &msg, std::array<double, 6> &qdot)
  {
    if (!have_joint_state_)
    {
      publishStatus("Waiting for joint_states");
      return false;
    }

    qdot = {
      direct_joint_scale_ * axisValue(msg, axis_joint1_),
      direct_joint_scale_ * axisValue(msg, axis_joint2_),
      direct_joint_scale_ * axisValue(msg, axis_joint3_),
      direct_joint_scale_ * axisValue(msg, axis_joint4_),
      direct_joint_scale_ * axisValue(msg, axis_joint5_),
      direct_joint_scale_ * axisValue(msg, axis_joint6_)
    };

    return !allZero(qdot);
  }

  void applyJointLimitProtection(std::array<double, 6> &qdot)
  {
    std::array<double, 6> pos{};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_joint_state_)
      {
        qdot.fill(0.0);
        return;
      }
      pos = joint_pos_;
    }

    for (size_t i = 0; i < 6; ++i)
    {
      if (pos[i] <= joint_min_[i] + joint_limit_margin_ && qdot[i] < 0.0)
      {
        qdot[i] = 0.0;
      }
      if (pos[i] >= joint_max_[i] - joint_limit_margin_ && qdot[i] > 0.0)
      {
        qdot[i] = 0.0;
      }
    }
  }

  void limitVelocityAndAcceleration(std::array<double, 6> &target)
  {
    const rclcpp::Time now = nh_->now();
    double dt = (now - last_slew_time_).seconds();

    if (dt <= 0.0 || dt > 0.20)
    {
      dt = 1.0 / command_rate_hz_;
    }

    last_slew_time_ = now;

    double max_abs = 0.0;
    for (double v : target)
    {
      max_abs = std::max(max_abs, std::abs(v));
    }

    if (max_abs > max_joint_velocity_)
    {
      const double scale = max_joint_velocity_ / max_abs;
      for (double &v : target)
      {
        v *= scale;
      }
    }

    for (size_t i = 0; i < 6; ++i)
    {
      const double desired = clampAbs(target[i], max_joint_velocity_);
      const bool accelerating = std::abs(desired) > std::abs(last_output_vel_[i]);
      const double limit = (accelerating ? max_joint_accel_ : max_joint_decel_) * dt;
      target[i] = last_output_vel_[i] + clampAbs(desired - last_output_vel_[i], limit);

      if (std::abs(target[i]) < joint_velocity_deadband_)
      {
        target[i] = 0.0;
      }
    }

    last_output_vel_ = target;
  }

  struct GuardVerdict
  {
    bool checked = false;      // false when the guard could not run at all
    bool in_bounds = true;
    bool collision = false;
    double penetration = 0.0;  // summed contact depth, for "is it getting worse"
    std::string reason;

    bool safe() const
    {
      return in_bounds && !collision;
    }
  };

  // Evaluates the pose reached by holding qdot for collision_preview_sec_.
  // want_details fills in the contact pair and penetration depth; it costs a
  // full contact enumeration, so the 80 Hz happy path leaves it off and lets
  // the collision check bail out at the first contact.
  GuardVerdict evaluateGuard(const std::array<double, 6> &qdot, bool want_details = false)
  {
    GuardVerdict verdict;

    if (!moveit_ready_ || !planning_scene_)
    {
      return verdict;
    }

    std::array<double, 6> pos{};
    std::map<std::string, double> external;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_joint_state_)
      {
        // Not a contact, so the "already touching" escape below cannot fire.
        verdict.reason = "no joint_states yet";
        verdict.in_bounds = false;
        verdict.checked = true;
        return verdict;
      }
      pos = joint_pos_;
      external = external_joint_pos_;
    }

    // Non-arm joints first (gripper opening, rover suspension), then the arm
    // preview on top. Mimic joints are driven by their master, never directly.
    for (const auto &entry : external)
    {
      const auto *jm = robot_model_->getJointModel(entry.first);
      if (jm && jm->getVariableCount() == 1 && !jm->getMimic())
      {
        robot_state_->setJointPositions(jm, &entry.second);
      }
    }

    for (size_t i = 0; i < 6; ++i)
    {
      const double next = std::clamp(
        pos[i] + qdot[i] * collision_preview_sec_, joint_min_[i], joint_max_[i]);
      robot_state_->setVariablePosition(arm_joint_names_[i], next);
    }

    robot_state_->update(true);
    verdict.checked = true;

    if (!robot_state_->satisfiesBounds(joint_model_group_))
    {
      verdict.in_bounds = false;

      for (size_t i = 0; i < 6; ++i)
      {
        const auto *jm = robot_model_->getJointModel(arm_joint_names_[i]);
        const double value = robot_state_->getVariablePosition(arm_joint_names_[i]);

        if (jm && !jm->satisfiesPositionBounds(&value))
        {
          verdict.reason += (verdict.reason.empty() ? "out of bounds: " : ", ") +
                            arm_joint_names_[i] + "=" + std::to_string(value);
        }
      }

      return verdict;
    }

    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;
    req.group_name = planning_group_;
    req.contacts = want_details;
    req.max_contacts = want_details ? 4 : 1;
    req.max_contacts_per_pair = 1;

    planning_scene_->checkSelfCollision(req, res, *robot_state_);
    verdict.collision = res.collision;

    for (const auto &contact : res.contacts)
    {
      if (verdict.reason.empty())
      {
        verdict.reason = contact.first.first + " vs " + contact.first.second;
      }

      for (const auto &point : contact.second)
      {
        verdict.penetration += std::abs(point.depth);
      }
    }

    if (verdict.collision && verdict.reason.empty())
    {
      verdict.reason = "self-collision";
    }

    return verdict;
  }

  bool selfCollisionSafe(const std::array<double, 6> &qdot)
  {
    const GuardVerdict verdict = evaluateGuard(qdot);
    return !verdict.checked || verdict.safe();
  }

  bool scaleToSafeVelocity(std::array<double, 6> &qdot, std::string &reason)
  {
    const std::array<double, 4> scales = {0.75, 0.50, 0.30, 0.15};

    for (double scale : scales)
    {
      std::array<double, 6> test{};
      for (size_t i = 0; i < 6; ++i)
      {
        test[i] = qdot[i] * scale;
      }

      const GuardVerdict verdict = evaluateGuard(test);

      if (!verdict.checked || verdict.safe())
      {
        qdot = test;
        publishStatus("MoveIt self-collision guard: velocity scaled");
        return true;
      }
    }

    // Nothing was safe. If the arm is already standing in the collision, every
    // preview inherits it and blocking would trap the operator with no way to
    // jog out - the escape has to come from the joystick, not only from RViz.
    // Allow the slowest step that does not push deeper into the contact.
    std::array<double, 6> slowest{};
    for (size_t i = 0; i < 6; ++i)
    {
      slowest[i] = qdot[i] * scales.back();
    }

    const GuardVerdict last = evaluateGuard(slowest, true);
    const GuardVerdict current = evaluateGuard({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, true);

    if (current.checked && current.collision && last.checked && last.collision &&
        last.penetration <= current.penetration + 1e-6)
    {
      qdot = slowest;

      publishStatus("MoveIt self-collision guard: already touching (" + current.reason +
                    "), allowing slow motion away");
      return true;
    }

    reason = last.checked ? last.reason : std::string("guard unavailable");
    qdot.fill(0.0);
    return false;
  }

  void publishVelocityOnlyTrajectory(const std::array<double, 6> &vel, bool force)
  {
    if (!force && !publishGateReady())
    {
      return;
    }

    std::array<double, 6> pos{};

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_joint_state_)
      {
        return;
      }
      pos = joint_pos_;
    }

    // IMPORTANT:
    // Jazzy joint_trajectory_controller can crash for this setup if the
    // trajectory contains velocities but no positions. Always send both.
    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = nh_->now();
    traj.joint_names.assign(arm_joint_names_.begin(), arm_joint_names_.end());

    trajectory_msgs::msg::JointTrajectoryPoint p1;
    trajectory_msgs::msg::JointTrajectoryPoint p2;

    p1.positions.resize(6);
    p2.positions.resize(6);
    p1.velocities.resize(6);
    p2.velocities.resize(6);

    for (size_t i = 0; i < 6; ++i)
    {
      p1.positions[i] = std::clamp(
        pos[i] + vel[i] * velocity_point_1_sec_,
        joint_min_[i], joint_max_[i]);

      p2.positions[i] = std::clamp(
        pos[i] + vel[i] * velocity_point_2_sec_,
        joint_min_[i], joint_max_[i]);

      p1.velocities[i] = vel[i];
      p2.velocities[i] = vel[i];
    }

    p1.time_from_start = rclcpp::Duration::from_seconds(velocity_point_1_sec_);
    p2.time_from_start = rclcpp::Duration::from_seconds(velocity_point_2_sec_);

    traj.points.push_back(p1);
    traj.points.push_back(p2);

    arm_pub_->publish(traj);
  }


  void publishZeroVelocityTrajectory(bool force)
  {
    std::array<double, 6> zero{};
    publishVelocityOnlyTrajectory(zero, force);
  }

  bool publishGateReady()
  {
    const rclcpp::Time now = nh_->now();
    const double min_period = 1.0 / std::max(1.0, command_rate_hz_);

    if ((now - last_publish_time_).seconds() < min_period * 0.90)
    {
      return false;
    }

    last_publish_time_ = now;
    return true;
  }

  void hardZeroStop()
  {
    // RViz-safe idle behavior:
    // Publish zero velocity only for a short stop window after joystick motion,
    // then become silent so MoveIt/RViz can control the same trajectory controller.
    const bool was_commanding =
      active_motion_ || !allZero(last_output_vel_) || stop_zero_cycles_left_ > 0;

    if (!was_commanding)
    {
      return;
    }

    if ((active_motion_ || !allZero(last_output_vel_)) && stop_zero_cycles_left_ <= 0)
    {
      stop_zero_cycles_left_ = stop_zero_cycles_total_;
    }

    active_motion_ = false;
    last_output_vel_.fill(0.0);

    if (stop_zero_cycles_left_ > 0)
    {
      publishZeroVelocityTrajectory(true);
      --stop_zero_cycles_left_;
    }
  }


  void publishZeroIfNeeded()
  {
    if (stop_zero_cycles_left_ > 0)
    {
      publishZeroVelocityTrajectory(true);
      --stop_zero_cycles_left_;
    }
  }

  bool allZero(const std::array<double, 6> &v) const
  {
    for (double x : v)
    {
      if (std::abs(x) > 1e-8)
      {
        return false;
      }
    }
    return true;
  }

  void updateGripper(const sensor_msgs::msg::Joy &msg)
  {
    const bool open_pressed = buttonPressed(msg, button_gripper_open_);
    const bool close_pressed = buttonPressed(msg, button_gripper_close_);
    const bool toggle_pressed = buttonPressed(msg, button_gripper_toggle_);

    const rclcpp::Time now = nh_->now();
    double dt = (now - last_gripper_update_).seconds();

    if (dt <= 0.0 || dt > 0.5)
    {
      dt = 1.0 / std::max(1.0, command_rate_hz_);
    }

    last_gripper_update_ = now;

    const double max_step = std::min(
      max_gripper_command_step_,
      std::max(0.001, gripper_speed_ * dt));

    if (open_pressed != close_pressed)
    {
      const double direction = close_pressed ? 1.0 : -1.0;
      publishGripper(commanded_gripper_position_ + direction * max_step,
                     gripper_trajectory_duration_);
      previous_gripper_toggle_pressed_ = toggle_pressed;
      return;
    }

    if (toggle_pressed && !previous_gripper_toggle_pressed_)
    {
      const double midpoint = 0.5 * (gripper_open_position_ + gripper_closed_position_);
      const double ref = have_gripper_position_ ? current_gripper_position_ : commanded_gripper_position_;
      const double target = ref <= midpoint ? gripper_closed_position_ : gripper_open_position_;
      const double distance = std::abs(target - commanded_gripper_position_);
      const double duration = std::max(
        gripper_trajectory_duration_,
        distance / std::max(0.001, gripper_speed_));
      publishGripper(target, duration);
    }

    previous_gripper_toggle_pressed_ = toggle_pressed;
  }

  void publishGripper(double target, double duration)
  {
    commanded_gripper_position_ = std::clamp(
      target, gripper_open_position_, gripper_closed_position_);

    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = nh_->now();
    traj.joint_names = {gripper_joint_name_};

    trajectory_msgs::msg::JointTrajectoryPoint p;
    p.positions = {commanded_gripper_position_};
    p.time_from_start = rclcpp::Duration::from_seconds(std::max(0.05, duration));

    traj.points.push_back(p);
    gripper_pub_->publish(traj);
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);

    bool found_all = true;

    for (size_t j = 0; j < 6; ++j)
    {
      bool found = false;

      for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i)
      {
        if (msg->name[i] == arm_joint_names_[j])
        {
          joint_pos_[j] = msg->position[i];

          if (i < msg->velocity.size())
          {
            joint_vel_[j] = msg->velocity[i];
          }

          found = true;
          break;
        }
      }

      found_all = found_all && found;
    }

    if (found_all)
    {
      have_joint_state_ = true;
    }

    // Mirror every joint the model knows so the collision guard sees the real
    // gripper opening and rover suspension, not the RobotState defaults.
    // /joint_states arrives split across publishers (arm + gripper from the
    // broadcaster, wheels from the rover), so accumulate instead of replacing.
    if (robot_model_)
    {
      for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i)
      {
        if (std::isfinite(msg->position[i]) && robot_model_->hasJointModel(msg->name[i]))
        {
          external_joint_pos_[msg->name[i]] = msg->position[i];
        }
      }
    }

    for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i)
    {
      if (msg->name[i] == gripper_joint_name_)
      {
        current_gripper_position_ = std::clamp(
          msg->position[i], gripper_open_position_, gripper_closed_position_);

        if (!have_gripper_position_)
        {
          commanded_gripper_position_ = current_gripper_position_;
          have_gripper_position_ = true;
        }
      }
    }
  }

  double rawAxis(const sensor_msgs::msg::Joy &msg, int axis) const
  {
    if (axis < 0 || static_cast<size_t>(axis) >= msg.axes.size())
    {
      return 0.0;
    }

    return msg.axes[axis];
  }

  double axisValue(const sensor_msgs::msg::Joy &msg, int axis) const
  {
    const double raw = rawAxis(msg, axis);

    if (std::abs(raw) < deadzone_)
    {
      return 0.0;
    }

    const double sign = raw >= 0.0 ? 1.0 : -1.0;

    if (constant_speed_mode_)
    {
      return sign;
    }

    return sign * (std::abs(raw) - deadzone_) / std::max(1e-6, 1.0 - deadzone_);
  }

  bool buttonPressed(const sensor_msgs::msg::Joy &msg, int button) const
  {
    if (button < 0 || static_cast<size_t>(button) >= msg.buttons.size())
    {
      return false;
    }

    return msg.buttons[button] != 0;
  }

  void publishStatus(const std::string &text)
  {
    const rclcpp::Time now = nh_->now();

    if (text == last_status_ && (now - last_status_time_).seconds() < 1.0)
    {
      return;
    }

    last_status_ = text;
    last_status_time_ = now;

    if (status_pub_)
    {
      std_msgs::msg::String msg;
      msg.data = text;
      status_pub_->publish(msg);
    }

    RCLCPP_INFO_THROTTLE(
      nh_->get_logger(), *nh_->get_clock(), 1000, "%s", text.c_str());
  }

  rclcpp::Node::SharedPtr nh_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr gripper_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;

  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
  moveit::core::RobotModelPtr robot_model_;
  std::unique_ptr<moveit::core::RobotState> robot_state_;
  std::unique_ptr<planning_scene::PlanningScene> planning_scene_;
  const moveit::core::JointModelGroup *joint_model_group_ = nullptr;
  const moveit::core::LinkModel *tip_link_model_ = nullptr;
  bool moveit_ready_ = false;

  std::mutex joy_mutex_;
  sensor_msgs::msg::Joy latest_joy_;
  bool have_latest_joy_ = false;

  std::mutex state_mutex_;
  std::array<double, 6> joint_pos_{};
  std::array<double, 6> joint_vel_{};
  bool have_joint_state_ = false;

  // Every joint_states entry the robot model knows about, arm joints included.
  // The collision guard needs the gripper and rover joints too: with them left
  // at the RobotState defaults the guard checks a closed gripper on a robot
  // whose gripper is actually open, and blocks poses that are really clear.
  std::map<std::string, double> external_joint_pos_;

  std::array<std::string, 6> arm_joint_names_ = DEFAULT_ARM_JOINTS;
  std::array<double, 6> joint_min_ = {-3.1241, -1.4835, -1.39626, -3.12414, -1.65806, -3.12414};
  std::array<double, 6> joint_max_ = { 3.1241,  2.4435,  2.61799,  3.12414,  1.65806,  3.12414};

  std::array<double, 6> last_output_vel_{};

  std::string joy_topic_;
  std::string joint_state_topic_;
  std::string arm_command_topic_;
  std::string gripper_command_topic_;
  std::string gripper_joint_name_;
  std::string status_topic_;
  std::string planning_group_;
  std::string planning_link_;

  double command_rate_hz_ = 80.0;
  double joy_timeout_sec_ = 0.35;
  double deadzone_ = 0.04;
  double linear_scale_ = 0.10;
  double angular_scale_ = 0.18;
  double direct_joint_scale_ = 0.28;

  double max_joint_velocity_ = 0.28;
  double max_joint_accel_ = 1.8;
  double max_joint_decel_ = 16.0;
  double joint_velocity_deadband_ = 0.004;
  double dls_lambda_ = 0.10;
  double joint_limit_margin_ = 0.05;
  double collision_preview_sec_ = 0.16;
  double velocity_point_1_sec_ = 0.040;
  double velocity_point_2_sec_ = 0.080;
  bool constant_speed_mode_ = false;

  int stop_zero_cycles_total_ = 4;
  int stop_zero_cycles_left_ = 0;

  int button_enable_ = 5;
  int button_arm_mode_toggle_ = 5;
  int button_rover_enable_ = 4;
  bool arm_toggle_mode_ = true;

  int button_gripper_open_ = 2;
  int button_gripper_close_ = 1;
  int button_gripper_toggle_ = 0;

  int axis_linear_x_ = 1;
  int axis_linear_y_ = 0;
  int axis_linear_z_ = 7;
  int axis_angular_x_ = 3;
  int axis_angular_y_ = 4;
  int axis_angular_z_ = 6;

  int axis_joint1_ = 0;
  int axis_joint2_ = 1;
  int axis_joint3_ = 4;
  int axis_joint4_ = 3;
  int axis_joint5_ = 6;
  int axis_joint6_ = 7;

  bool arm_mode_initialized_ = false;
  bool previous_rb_toggle_pressed_ = false;
  Mode active_mode_ = Mode::CARTESIAN;
  bool active_motion_ = false;

  double gripper_speed_ = 2.0;
  double gripper_open_position_ = -1.57;
  double gripper_closed_position_ = 0.07;
  double gripper_trajectory_duration_ = 0.10;
  double max_gripper_command_step_ = 0.04;
  double commanded_gripper_position_ = -1.57;
  double current_gripper_position_ = -1.57;
  bool have_gripper_position_ = false;
  bool previous_gripper_toggle_pressed_ = false;

  rclcpp::Time last_joy_time_;
  rclcpp::Time last_publish_time_;
  rclcpp::Time last_slew_time_;
  rclcpp::Time last_gripper_update_;
  rclcpp::Time last_status_time_{0, 0, RCL_ROS_TIME};
  std::string last_status_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_unique<RebelSmoothSafeJoystick>();
  node->spin();
  rclcpp::shutdown();
  return 0;
}

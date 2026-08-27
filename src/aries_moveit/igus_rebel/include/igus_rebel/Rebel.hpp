#ifndef REBEL_HPP_
#define REBEL_HPP_

#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <math.h>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int16.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "igus_rebel_msgs/msg/digital_output.hpp"
#include "igus_rebel_msgs/srv/set_digital_output.hpp"
#include <hardware_interface/system_interface.hpp>

#include "igus_rebel/RebelSocket.hpp"
#include "igus_rebel/CriMessages.hpp"

// 1.0 = jog% numerically equals speed in deg/s (100% jog = 100 deg/s).
// igus Rebel 6DOF max joint speed ~100 deg/s, so scale=1.0 is correct.
// Do NOT use 2.0 here: that saturates jog at only 50 deg/s causing severe tracking lag.
#define JOINT_VELOCITY_SCALE 1.0
// Jog commands below this percentage of max speed are zeroed.
// Prevents tiny P-gain corrections at end-of-motion from fighting the
// robot's own internal servo hold, which causes visible jitter.
#define VELOCITY_DEADBAND_PCT 0.15f

using namespace hardware_interface;

namespace Igus
{
    class Rebel : public SystemInterface
    {
    public:
        enum class ControlMode
        {
            POSITION,
            VELOCITY
        };

    private:
        rclcpp::Node::SharedPtr node_;
        
        std::shared_ptr<RebelSocket> rebelSocket;
        CriMessages::Status currentStatus;

        // The arm's emergency stop, which is wired straight to the control box
        // and is invisible to ROS without this. CRI reports it in every status
        // message and the parser has always stored it -- it was simply never
        // published, so nothing downstream (the stack light above all) could
        // know the arm had been e-stopped.
        //
        // Two topics on purpose. The Bool is the interpreted answer everything
        // consumes; the Int32 is the raw CRI field, because the meaning of the
        // number is NOT established -- CriMessages.cpp still carries a "TODO:
        // process further to actual meaning" against it. Watch the raw topic
        // while pressing the button once, then set estop_pressed_value to
        // whatever it reads. Guessing silently would give the operator a red
        // light that means nothing.
        void PublishEStop(const CriMessages::Status &);

        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr estop_pub_;
        rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr estop_raw_pub_;
        int estop_pressed_value_;
        bool estop_published_;
        int last_estop_raw_;

        // Per-joint motor current, straight out of the CRI status message.
        // CriMessages has always parsed CURRENTJOINTS and nothing ever read it,
        // so the stack had no measure of how hard the arm was pushing -- the
        // first sign of a stall was the joint module tripping on 'Position lag'
        // or 'Overcurrent' and disabling the motors. Publishing it is what lets
        // teleop stop before the firmware does.
        //
        // Raw CRI units (mA on the Rebel), NOT newton-metres. It is also
        // mirrored into the effort state interface so it shows up in
        // /joint_states and can be plotted with the joint positions.
        void PublishLoad(const CriMessages::Status &);

        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_current_pub_;
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr fault_pub_;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fault_detail_pub_;
        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
        std::chrono::steady_clock::time_point last_current_publish_;
        double current_publish_period_;
        bool fault_published_;
        bool last_fault_;

        // Current commanded jog
        float j1, j2, j3, j4, j5, j6;
        ControlMode controlMode;

        bool continueAlive;
        bool continueMessage;
        std::thread aliveThread;
        std::thread messageThread;
        int aliveWaitMs;

        int current_ccnt;
        std::mutex cntLock;
        std::mutex aliveLock;

        // Zero-torque (hand-guiding) state is confirmed from the CRI reply.  The
        // alive thread keeps sending zero jog while this is true, even if a ROS
        // controller still has an old velocity command buffered.
        std::atomic<bool> handGuiding{false};
        std::mutex zeroTorqueLock;
        std::condition_variable zeroTorqueCondition;
        bool zeroTorqueAllowed{false};
        bool zeroTorqueEnabled{false};
        unsigned long zeroTorqueResponseCount{0};

        double vel_cmd[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double pos[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double last_pos[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double vel[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double eff[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        CriMessages::Kinstate lastKinstate;
        std::array<int, 16> lastErrorJoints;
        std::string kinstateMessage;

        std::unordered_map<int, std::string> unacknowledgedCommands;

        // ROS2 communication
        rclcpp::Service<igus_rebel_msgs::srv::SetDigitalOutput>::SharedPtr digital_output_srv_;
        rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr hand_guiding_srv_;

        // Thread functions
        void AliveThreadFunction();
        void MessageThreadFunction();

        // Other functions
        int Ccnt();
        void Command(const std::string &);
        void GetConfig(const std::string &);
        void SetControlMode(const ControlMode &);

        // Function to react to specific status values, to display warnings, error messages, etc.
        void ProcessStatus(const CriMessages::Status &);
        void ProcessZeroTorqueResponse(const std::string &);
        void SetUpRosHardwareInterface();

    public:
        const std::vector<std::string> JOINT_NAME = {
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

        // pi / 180
        const double degToRad = 0.0174532925199432957692369076848861271344287188854172545609719144;

        // IP & port
        const std::string ip = "192.168.3.11";
        const int port = 3920;

        Rebel();
        ~Rebel();

        void SetJog(const float &, const float &, const float &, const float &, const float &, const float &);
        void GetJoints(float &, float &, float &, float &, float &, float &);
        void SetDigitalOut(const int &, const bool &);

        // Interaction with hardware for ROS2
        CallbackReturn on_init(const HardwareInfo &hardware_info) override;
        CallbackReturn on_configure(const rclcpp_lifecycle::State &previous_state) override;
        CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;
        CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

        std::vector<StateInterface> export_state_interfaces() override;
        std::vector<CommandInterface> export_command_interfaces() override;
        return_type read(const rclcpp::Time &time, const rclcpp::Duration &period) override;
        return_type write(const rclcpp::Time &time, const rclcpp::Duration &period) override;

        void read();
        void write();

        void dio_callback(const std::shared_ptr<igus_rebel_msgs::srv::SetDigitalOutput::Request> request,
                          std::shared_ptr<igus_rebel_msgs::srv::SetDigitalOutput::Response> response);
        void hand_guiding_callback(
            const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
            std::shared_ptr<std_srvs::srv::SetBool::Response> response);

        // Clear a tripped joint module and re-enable the motors without
        // restarting the stack. After an overcurrent/position-lag trip the
        // Rebel leaves the motors disabled and every later jog is ignored in
        // silence, which reads as "the arm died".
        void reset_callback(
            const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
            std::shared_ptr<std_srvs::srv::Trigger::Response> response);

        void GetReferenceInfo();

        void Start();
        void Stop();
    };
}

#endif

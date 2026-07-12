#ifndef REBEL_HPP_
#define REBEL_HPP_

#include <thread>
#include <mutex>
#include <math.h>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int16.hpp"
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

        void GetReferenceInfo();

        void Start();
        void Stop();
    };
}

#endif
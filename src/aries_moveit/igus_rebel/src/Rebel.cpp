#include "rclcpp/rclcpp.hpp"
#include "igus_rebel/Rebel.hpp"
#include "igus_rebel/CriKeywords.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <utility>

namespace Igus
{

    //
    // Constructor(s) / Destructor(s)
    //
    Rebel::Rebel()
    {
    }

    Rebel::~Rebel()
    {
        Stop();
    }

    //
    // private functions
    //
    void Rebel::AliveThreadFunction()
    {
        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Starting to send ALIVEJOG");

        while (continueAlive)
        {
            std::ostringstream msg;
            msg << std::showpoint;
            msg << std::fixed;
            msg << std::setprecision(8);
            msg << "CRISTART " << Ccnt() << " ";
            msg << "ALIVEJOG ";
            {
                std::lock_guard<std::mutex> lockGuard(aliveLock);
                if (handGuiding.load())
                {
                    msg << 0.0f << " " << 0.0f << " " << 0.0f << " ";
                    msg << 0.0f << " " << 0.0f << " " << 0.0f << " ";
                }
                else
                {
                    msg << j1 << " " << j2 << " " << j3 << " ";
                    msg << j4 << " " << j5 << " " << j6 << " ";
                }
                msg << 0.0f << " " << 0.0f << " " << 0.0f << " ";
                msg << "CRIEND" << std::endl;
                rebelSocket->SendMessage(msg.str());
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(aliveWaitMs));
        }

        RCLCPP_WARN(rclcpp::get_logger("igus_rebel"), "Stopped to send ALIVEJOG");
    }

    void Rebel::MessageThreadFunction()
    {
        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Starting to process robot messages");

        while (continueMessage)
        {
            if (rebelSocket->HasMessage())
            {
                std::string msg = rebelSocket->GetMessage();

                CriMessages::MessageType type = CriMessages::CriMessage::GetMessageType(msg);

                switch (type)
                {
                case CriMessages::MessageType::STATUS:
                {
                    CriMessages::Status status = CriMessages::Status(msg);
                    // status.Print();
                    status.Log();
                    currentStatus = status;
                    ProcessStatus(currentStatus);
                    break;
                }

                case CriMessages::MessageType::RUNSTATE:
                {
                    break;
                }

                case CriMessages::MessageType::MESSAGE:
                {
                    CriMessages::Message message = CriMessages::Message(msg);
                    RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Rebel MESSAGE: %s", message.message.c_str());
                    break;
                }

                case CriMessages::MessageType::CMD:
                {
                    CriMessages::Command command = CriMessages::Command(msg);

                    if (command.command.rfind(CriKeywords::COMMAND_ZEROTORQUE, 0) == 0)
                    {
                        ProcessZeroTorqueResponse(command.command);
                    }

                    // Not sure if the ROS node should display these?
                    RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "CMD: %s", command.command.c_str());
                    break;
                }

                case CriMessages::MessageType::CONFIG:
                {
                    CriMessages::ConfigType configType = CriMessages::Config::GetConfigType(msg);

                    switch (configType)
                    {
                    case CriMessages::ConfigType::KINEMATICLIMITS:
                    {
                        CriMessages::KinematicLimits kinematicLimits = CriMessages::KinematicLimits(msg);
                        // kinematicLimits.Print();
                        break;
                    }
                    case CriMessages::ConfigType::UNKNOWN:
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Unknown config message: %s", msg.c_str());
                        break;
                    }
                    }

                    break;
                }

                case CriMessages::MessageType::INFO:
                {
                    CriMessages::Info info = CriMessages::Info(msg);
                    RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "INFO: %s", info.info.c_str());
                    break;
                }

                case CriMessages::MessageType::LOGMSG:
                {
                    CriMessages::LogMsg log = CriMessages::LogMsg(msg);

                    switch (log.logLevel)
                    {
                    case CriMessages::LogLevel::DEBUG:
                    {
                        RCLCPP_DEBUG(rclcpp::get_logger("igus_rebel"), "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
                        break;
                    }

                    case CriMessages::LogLevel::APP_INFO:
                    {
                        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "REBEL LOG (APP_INFO): %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
                        break;
                    }

                    case CriMessages::LogLevel::APP_ERROR:
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "REBEL LOG (APP_ERROR): %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
                        break;
                    }

                    case CriMessages::LogLevel::INFO:
                    {
                        // The Rebel is pretty chatty with its INFO level log messages, so I've set them to output only to the ROS DEBUG level.
                        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
                        break;
                    }

                    case CriMessages::LogLevel::WARN:
                    {
                        RCLCPP_WARN(rclcpp::get_logger("igus_rebel"), "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
                        break;
                    }

                    case CriMessages::LogLevel::ERROR:
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
                        break;
                    }

                    case CriMessages::LogLevel::FATAL:
                    {
                        RCLCPP_FATAL(rclcpp::get_logger("igus_rebel"), "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
                        break;
                    }

                    case CriMessages::LogLevel::UNKNOWN:
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "REBEL LOG (UNKNOWN LOG LEVEL): %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
                        break;
                    }
                    }

                    break;
                }

                case CriMessages::MessageType::VARIABLES:
                {
                    // CriMessages::Variables vars = CriMessages::Variables(msg);
                    break;
                }

                case CriMessages::MessageType::CMDERROR:
                {
                    CriMessages::CmdError error = CriMessages::CmdError(msg);

                    try
                    {
                        std::string command = unacknowledgedCommands.at(error.recjectedCmd);
                        unacknowledgedCommands.erase(error.recjectedCmd);
                        RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Rebel did not accept command: %s. Error message: %s", command.c_str(), error.error.c_str());
                    }
                    catch (const std::out_of_range &e)
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Rebel did not accept unknown command. Error message: %s (%d)", error.error.c_str(), error.recjectedCmd);
                    }
                    break;
                }

                case CriMessages::MessageType::CMDACK:
                {
                    CriMessages::CmdAck ack = CriMessages::CmdAck(msg);

                    try
                    {
                        std::string command = unacknowledgedCommands.at(ack.acceptedCmd);
                        unacknowledgedCommands.erase(ack.acceptedCmd);
                        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Rebel accepted command: %s", command.c_str());
                        break;
                    }
                    catch (const std::out_of_range &e)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("igus_rebel"), "Rebel accepted unknown command: %d", ack.acceptedCmd);
                        break;
                    }
                    break;
                }

                case CriMessages::MessageType::CYCLESTAT:
                {
                    CriMessages::Cyclestat cyclestat = CriMessages::Cyclestat(msg);
                    // Will only output this once every 2 minutes, because this is sent every 0.5 seconds.
                    RCLCPP_INFO_THROTTLE(rclcpp::get_logger("igus_rebel"), *node_->get_clock(), 120, "Rebel cycle statistics -- Cycletime: %d -- Workload: %d%%", cyclestat.cycletime, cyclestat.workload);
                    break;
                }

                case CriMessages::MessageType::UNKNOWN:
                {
                    RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "UNKNOW MESSAGE: %s", msg.c_str());
                    break;
                }

                case CriMessages::MessageType::OPINFO:
                {
                    break;
                }

                case CriMessages::MessageType::GSIG:
                {
                    break;
                }
                case CriMessages::MessageType::GRIPPERSTATE:
                {
                    break;
                }
                }
            }
        }

        RCLCPP_WARN(rclcpp::get_logger("igus_rebel"), "Stopped to process robot messages");
    }

    int Rebel::Ccnt()
    {
        std::lock_guard<std::mutex> lockGuard(cntLock);
        int current = current_ccnt;
        current_ccnt = (current_ccnt % 9999) + 1;
        return current;
    }

    void Rebel::SetDigitalOut(const int &output, const bool &is_on)
    {
        std::ostringstream cmd;
        cmd << CriKeywords::COMMAND_DOUT << " " << output << " " << (is_on ? "true" : "false");
        Command(cmd.str());
    }

    void Rebel::Command(const std::string &command)
    {
        int commandCount = Ccnt();
        std::ostringstream msg;
        msg << CriKeywords::START << " " << commandCount << " ";
        msg << CriKeywords::TYPE_CMD << " ";
        msg << command << " ";
        msg << CriKeywords::END << std::endl;

        unacknowledgedCommands[commandCount] = command;

        rebelSocket->SendMessage(msg.str());
    }

    void Rebel::GetConfig(const std::string &config)
    {
        std::ostringstream msg;
        msg << CriKeywords::START << " " << Ccnt() << " ";
        msg << CriKeywords::TYPE_CONFIG << " ";
        msg << config << " ";
        msg << CriKeywords::END << std::endl;

        rebelSocket->SendMessage(msg.str());
    }

    void Rebel::SetControlMode(const ControlMode &mode)
    {
        switch (mode)
        {
        case Rebel::ControlMode::POSITION:
        {
            {
                std::lock_guard<std::mutex> lockGuard(aliveLock);

                j1 = currentStatus.posJointCurrent.at(0);
                j2 = currentStatus.posJointCurrent.at(1);
                j3 = currentStatus.posJointCurrent.at(2);
                j4 = currentStatus.posJointCurrent.at(3);
                j5 = currentStatus.posJointCurrent.at(4);
                j6 = currentStatus.posJointCurrent.at(5);

                Command(CriKeywords::COMMAND_MOTIONTYPECARTBASE);
                controlMode = mode;
            }
            RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Rebel now controlled by position control.");
            break;
        }

        case Rebel::ControlMode::VELOCITY:
        {
            Command(CriKeywords::COMMAND_MOTIONTYPEJOINT);
            controlMode = mode;
            RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Rebel now controlled by velocity control.");
            break;
        }
        }
    }

    // Publish the arm's e-stop out of the CRI status. Only on change (and
    // once at startup), because ProcessStatus runs on every status message --
    // that is tens of hertz, and a latched condition does not need repeating.
    void Rebel::PublishEStop(const CriMessages::Status &status)
    {
        if (!estop_pub_)
        {
            return;
        }
        if (estop_published_ && status.eStop == last_estop_raw_)
        {
            return;
        }
        last_estop_raw_ = status.eStop;
        estop_published_ = true;

        std_msgs::msg::Int32 raw;
        raw.data = status.eStop;
        estop_raw_pub_->publish(raw);

        std_msgs::msg::Bool pressed;
        pressed.data = (status.eStop == estop_pressed_value_);
        estop_pub_->publish(pressed);

        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"),
                    "Arm e-stop: raw %d -> %s (pressed value %d)",
                    status.eStop, pressed.data ? "PRESSED" : "released",
                    estop_pressed_value_);
    }

    // Text for the joint error bitfield CRI reports per joint. Used both by the
    // change-triggered log below and by the /arm/fault_detail topic, so the two
    // can never drift apart.
    static std::string JointErrorText(int bits)
    {
        std::string text;
        const std::pair<CriMessages::ErrorJoint, const char *> names[] = {
            {CriMessages::ErrorJoint::TEMP, "Overtemperature"},
            {CriMessages::ErrorJoint::ESTOP_LOWV, "Supply too low: Is emergency button pressed?"},
            {CriMessages::ErrorJoint::MNE, "Motor not enabled"},
            {CriMessages::ErrorJoint::COM, "Communication watch dog"},
            {CriMessages::ErrorJoint::POS, "Position lag"},
            {CriMessages::ErrorJoint::ENC, "Encoder Error"},
            {CriMessages::ErrorJoint::OC, "Overcurrent"},
            {CriMessages::ErrorJoint::DRV, "DriveError/SVM"},
        };

        for (const auto &entry : names)
        {
            if (bits & static_cast<int>(entry.first))
            {
                text += (text.empty() ? "" : " ") + std::string("'") + entry.second + "'";
            }
        }

        return text;
    }

    // Per-joint motor current and the joint error bits, both of which arrive in
    // every CRI status message and were previously only logged (the errors) or
    // dropped entirely (the current).
    //
    // This is the signal a contact guard needs. The Rebel's joint modules run
    // their own closed loop: a jog they cannot follow makes their internal
    // setpoint run away from the real position until the module trips on
    // 'Position lag' or 'Overcurrent' and disables the motors. Current rises
    // well before that point, so anything watching this topic can back off
    // first -- which is the whole difference between "the arm stopped" and
    // "the arm tripped".
    void Rebel::PublishLoad(const CriMessages::Status &status)
    {
        if (joint_current_pub_)
        {
            const auto now = std::chrono::steady_clock::now();
            const double since = std::chrono::duration<double>(
                now - last_current_publish_).count();

            // Status arrives far faster than anyone needs to watch a contact
            // force, and this topic crosses the field link.
            if (since >= current_publish_period_)
            {
                last_current_publish_ = now;

                std_msgs::msg::Float64MultiArray currents;
                currents.data.resize(6);
                for (int i = 0; i < 6; ++i)
                {
                    currents.data[i] = static_cast<double>(status.currentjoints.at(i));
                }
                joint_current_pub_->publish(currents);
            }
        }

        // A joint is faulted when any of its error bits is set. That includes
        // 'Motor not enabled', which is exactly the state a trip leaves behind,
        // so this stays true until something calls the reset service.
        bool faulted = false;
        std::string detail;
        for (int i = 0; i < 6; ++i)
        {
            const int bits = status.errorJoints.at(i);
            if (bits == 0)
            {
                continue;
            }
            faulted = true;
            detail += (detail.empty() ? "" : ", ") + std::string("joint") +
                      std::to_string(i + 1) + ": " + JointErrorText(bits);
        }

        if (fault_published_ && faulted == last_fault_)
        {
            return;
        }
        last_fault_ = faulted;
        fault_published_ = true;

        if (fault_pub_)
        {
            std_msgs::msg::Bool msg;
            msg.data = faulted;
            fault_pub_->publish(msg);
        }
        if (fault_detail_pub_)
        {
            std_msgs::msg::String msg;
            msg.data = faulted ? detail : std::string("clear");
            fault_detail_pub_->publish(msg);
        }
        if (faulted)
        {
            RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"),
                         "Arm faulted [%s]; motors are disabled until "
                         "/arm/reset is called (RT+Y on the pad)",
                         detail.c_str());
        }
        else
        {
            RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Arm fault cleared");
        }
    }

    void Rebel::ProcessStatus(const CriMessages::Status &status)
    {
        PublishEStop(status);
        PublishLoad(status);

        CriMessages::Kinstate currentKinstate = status.kinstate;
        std::array<int, 16> currentErrorJoints = status.errorJoints;

        if (lastKinstate != currentKinstate)
        {

            if (lastKinstate != CriMessages::Kinstate::NO_ERROR)
            {
                RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Kinematics error resolved [%s]", kinstateMessage.c_str());
            }

            if (currentKinstate != CriMessages::Kinstate::NO_ERROR)
            {

                switch (status.kinstate)
                {
                case CriMessages::Kinstate::JOINT_LIMIT_MIN:
                {
                    kinstateMessage = "joint at minimum limit";
                    break;
                }

                case CriMessages::Kinstate::JOINT_LIMIT_MAX:
                {
                    kinstateMessage = "joint at maximum limit";
                    break;
                }

                case CriMessages::Kinstate::CARTESIAN_SINGULARITY_CENTER:
                {
                    kinstateMessage = "cartesian singularity (center)";
                    break;
                }

                case CriMessages::Kinstate::CARTESIAN_SINGULARITY_REACH:
                {
                    kinstateMessage = "cartesian singularity (reach)";
                    break;
                }

                case CriMessages::Kinstate::CARTESIAN_SINGULARITY_WRIST:
                {
                    kinstateMessage = "cartesian singularity (wrist)";
                    break;
                }

                case CriMessages::Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_1:
                {
                    kinstateMessage = "tool at virtual box limit 1";
                    break;
                }

                case CriMessages::Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_2:
                {
                    kinstateMessage = "tool at virtual box limit 2";
                    break;
                }

                case CriMessages::Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_3:
                {
                    kinstateMessage = "tool at virtual box limit 3";
                    break;
                }

                case CriMessages::Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_4:
                {
                    kinstateMessage = "tool at virtual box limit 4";
                    break;
                }

                case CriMessages::Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_5:
                {
                    kinstateMessage = "tool at virtual box limit 5";
                    break;
                }

                case CriMessages::Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_6:
                {
                    kinstateMessage = "tool at virtual box limit 6";
                    break;
                }

                case CriMessages::Kinstate::MOTION_NOT_ALLOWED:
                {
                    kinstateMessage = "motion not allowed";
                    break;
                }

                case CriMessages::Kinstate::UNKNOWN:
                {
                    kinstateMessage = "unknown error";
                    break;
                }

                case CriMessages::Kinstate::NO_ERROR:
                {
                    kinstateMessage = "no error";
                    break;
                }
                }

                RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Kinematics error [%s]", kinstateMessage.c_str());
            }
        }

        if (currentErrorJoints != lastErrorJoints)
        {

            // loop throught the 6 joint errors
            for (unsigned int i = 0; i < 6; i++)
            {
                int errorJoint = currentErrorJoints.at(i);

                if (errorJoint != lastErrorJoints.at(i))
                {
                    std::string errorMsg = JointErrorText(errorJoint);

                    if (errorMsg != "")
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Joint %i Error: [%s]", i, errorMsg.c_str());
                    }
                    else
                    {
                        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Joint %i Error: Cleared", i);
                    }
                }
            }
        }

        lastKinstate = currentKinstate;
        lastErrorJoints = currentErrorJoints;
    }

    void Rebel::ProcessZeroTorqueResponse(const std::string &command)
    {
        // Expected CRI response: "ZeroTorque <allowed> <enabled> [CRIEND]".
        std::istringstream stream(command);
        std::string keyword;
        std::string allowedText;
        std::string enabledText;
        stream >> keyword >> allowedText >> enabledText;

        auto parseBool = [](std::string value, bool &result) {
            std::transform(value.begin(), value.end(), value.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (value == "true" || value == "1")
            {
                result = true;
                return true;
            }
            if (value == "false" || value == "0")
            {
                result = false;
                return true;
            }
            return false;
        };

        bool allowed = false;
        bool enabled = false;
        if (keyword != CriKeywords::COMMAND_ZEROTORQUE ||
            !parseBool(allowedText, allowed) || !parseBool(enabledText, enabled))
        {
            RCLCPP_WARN(rclcpp::get_logger("igus_rebel"),
                "Could not parse ZeroTorque response: %s", command.c_str());
            return;
        }

        {
            std::lock_guard<std::mutex> lock(zeroTorqueLock);
            zeroTorqueAllowed = allowed;
            zeroTorqueEnabled = enabled;
            ++zeroTorqueResponseCount;
        }
        zeroTorqueCondition.notify_all();

        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"),
            "ZeroTorque available=%s active=%s",
            allowed ? "true" : "false", enabled ? "true" : "false");
    }

    //
    // public functions
    //
    void Rebel::SetJog(const float &joint1, const float &joint2, const float &joint3,
                       const float &joint4, const float &joint5, const float &joint6)
    {
        j1 = joint1;
        j2 = joint2;
        j3 = joint3;
        j4 = joint4;
        j5 = joint5;
        j6 = joint6;
    }

    void Rebel::GetJoints(float &joint1, float &joint2, float &joint3,
                          float &joint4, float &joint5, float &joint6)
    {
        joint1 = currentStatus.posJointCurrent.at(0);
        joint2 = currentStatus.posJointCurrent.at(1);
        joint3 = currentStatus.posJointCurrent.at(2);
        joint4 = currentStatus.posJointCurrent.at(3);
        joint5 = currentStatus.posJointCurrent.at(4);
        joint6 = currentStatus.posJointCurrent.at(5);
    }

    CallbackReturn Rebel::on_init(const HardwareInfo &)
    {
        rebelSocket = std::make_shared<RebelSocket>(ip, port, 200),
        j1 = 0.0f;
        j2 = 0.0f;
        j3 = 0.0f;
        j4 = 0.0f;
        j5 = 0.0f;
        j6 = 0.0f;
        controlMode = Rebel::ControlMode::VELOCITY;
        current_ccnt = 1;
        continueAlive = false;
        continueMessage = false;
        aliveWaitMs = 10;
        lastKinstate = CriMessages::Kinstate::NO_ERROR;
        kinstateMessage = "";
        node_ = std::make_shared<rclcpp::Node>("igus_rebel");

        // Latched: the e-stop state is a condition, not an event, so anything
        // that starts later -- the stack light, an operator's rviz -- must get
        // the current value rather than wait for the next change.
        auto estop_qos = rclcpp::QoS(1).reliable().transient_local();
        estop_pub_ = node_->create_publisher<std_msgs::msg::Bool>(
            "/arm/estop", estop_qos);
        estop_raw_pub_ = node_->create_publisher<std_msgs::msg::Int32>(
            "/arm/estop_raw", estop_qos);
        // Which raw CRI value means PRESSED. Not verified against hardware --
        // see the note in Rebel.hpp. Determine it once by watching
        // /arm/estop_raw while pressing the button, then set this.
        estop_pressed_value_ = node_->has_parameter("estop_pressed_value")
            ? node_->get_parameter("estop_pressed_value").as_int()
            : node_->declare_parameter<int>("estop_pressed_value", 0);
        estop_published_ = false;
        last_estop_raw_ = 0;

        // How hard the arm is pushing, and whether a joint module has tripped.
        // The fault is a condition, not an event, so it is latched the same way
        // the e-stop is: a teleop node that starts later still learns the arm
        // is sitting disabled.
        auto fault_qos = rclcpp::QoS(1).reliable().transient_local();
        joint_current_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/arm/joint_currents", rclcpp::QoS(10));
        fault_pub_ = node_->create_publisher<std_msgs::msg::Bool>(
            "/arm/fault", fault_qos);
        fault_detail_pub_ = node_->create_publisher<std_msgs::msg::String>(
            "/arm/fault_detail", fault_qos);
        const double current_rate = node_->has_parameter("current_publish_rate")
            ? node_->get_parameter("current_publish_rate").as_double()
            : node_->declare_parameter<double>("current_publish_rate", 20.0);
        current_publish_period_ = 1.0 / std::max(1.0, current_rate);
        last_current_publish_ = std::chrono::steady_clock::now();
        fault_published_ = false;
        last_fault_ = false;
        for (double &value : eff)
        {
            value = 0.0;
        }

        reset_srv_ = node_->create_service<std_srvs::srv::Trigger>(
            "/arm/reset",
            std::bind(&Rebel::reset_callback, this, std::placeholders::_1, std::placeholders::_2));

        digital_output_srv_ = node_->create_service<igus_rebel_msgs::srv::SetDigitalOutput>(
            "set_digital_output", std::bind(&Rebel::dio_callback, this, std::placeholders::_1, std::placeholders::_2));
        hand_guiding_srv_ = node_->create_service<std_srvs::srv::SetBool>(
            "~/set_hand_guiding",
            std::bind(&Rebel::hand_guiding_callback, this, std::placeholders::_1, std::placeholders::_2));
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn Rebel::on_configure(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Configuring Rebel hardware interface");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn Rebel::on_activate(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Activating Rebel hardware interface");
        Start();
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn Rebel::on_deactivate(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Deactivating Rebel hardware interface");
        Stop();
        return CallbackReturn::SUCCESS;
    }

    std::vector<StateInterface> Rebel::export_state_interfaces()
    {
        std::vector<StateInterface> state_interfaces;

        for (int i = 0; i < 6; ++i)
        {
            state_interfaces.emplace_back(StateInterface(
                JOINT_NAME[i], hardware_interface::HW_IF_POSITION, &pos[i]));
            state_interfaces.emplace_back(StateInterface(
                JOINT_NAME[i], hardware_interface::HW_IF_VELOCITY, &vel[i]));
            // Motor current in raw CRI units, NOT newton-metres. Exported as
            // effort so it lands in /joint_states next to the positions it has
            // to be read against -- a current reading only means something once
            // you know whether the joint was moving at the time.
            state_interfaces.emplace_back(StateInterface(
                JOINT_NAME[i], hardware_interface::HW_IF_EFFORT, &eff[i]));
        }

        return state_interfaces;
    }

    std::vector<CommandInterface> Rebel::export_command_interfaces()
    {
        std::vector<CommandInterface> command_interfaces;

        for (int i = 0; i < 6; ++i)
        {
            command_interfaces.emplace_back(CommandInterface(
                JOINT_NAME[i], hardware_interface::HW_IF_VELOCITY, &vel_cmd[i]));
        }

        return command_interfaces;
    }

    return_type Rebel::read(const rclcpp::Time &, const rclcpp::Duration &period)
    {
        read();

        vel[0] = (pos[0] - last_pos[0]) / period.seconds();
        vel[1] = (pos[1] - last_pos[1]) / period.seconds();
        vel[2] = (pos[2] - last_pos[2]) / period.seconds();
        vel[3] = (pos[3] - last_pos[3]) / period.seconds();
        vel[4] = (pos[4] - last_pos[4]) / period.seconds();
        vel[5] = (pos[5] - last_pos[5]) / period.seconds();

        last_pos[0] = pos[0];
        last_pos[1] = pos[1];
        last_pos[2] = pos[2];
        last_pos[3] = pos[3];
        last_pos[4] = pos[4];
        last_pos[5] = pos[5];
        return return_type::OK;
    }

    void Rebel::read()
    {
        pos[0] = currentStatus.posJointCurrent.at(0) * degToRad;
        pos[1] = currentStatus.posJointCurrent.at(1) * degToRad;
        pos[2] = currentStatus.posJointCurrent.at(2) * degToRad;
        pos[3] = currentStatus.posJointCurrent.at(3) * degToRad;
        pos[4] = currentStatus.posJointCurrent.at(4) * degToRad;
        pos[5] = currentStatus.posJointCurrent.at(5) * degToRad;

        for (int i = 0; i < 6; ++i)
        {
            eff[i] = static_cast<double>(currentStatus.currentjoints.at(i));
        }
    }

    return_type Rebel::write(const rclcpp::Time &, const rclcpp::Duration &)
    {
        // Curently no use for time or period, here.
        write();
        return return_type::OK;
    }

    void Rebel::write()
    {
        // Check and call DIO callback
        if (rclcpp::ok())
        {
            rclcpp::spin_some(node_);
        }

        // Never let a buffered controller command reach the robot in zero-torque mode.
        if (handGuiding.load())
        {
            std::lock_guard<std::mutex> lockGuard(aliveLock);
            j1 = j2 = j3 = j4 = j5 = j6 = 0.0f;
            return;
        }

        // Apply dead-band: zero out commands below VELOCITY_DEADBAND_PCT so the
        // robot's own servo holds final position instead of chasing micro-corrections.
        auto db = [](float v) -> float {
            return (std::fabs(v) < VELOCITY_DEADBAND_PCT) ? 0.0f : v;
        };

        std::lock_guard<std::mutex> lockGuard(aliveLock);
        j1 = db(JOINT_VELOCITY_SCALE * (float)vel_cmd[0] / degToRad);
        j2 = db(JOINT_VELOCITY_SCALE * (float)vel_cmd[1] / degToRad);
        j3 = db(JOINT_VELOCITY_SCALE * (float)vel_cmd[2] / degToRad);
        j4 = db(JOINT_VELOCITY_SCALE * (float)vel_cmd[3] / degToRad);
        j5 = db(JOINT_VELOCITY_SCALE * (float)vel_cmd[4] / degToRad);
        j6 = db(JOINT_VELOCITY_SCALE * (float)vel_cmd[5] / degToRad);
    }

    void Rebel::dio_callback(
        const std::shared_ptr<igus_rebel_msgs::srv::SetDigitalOutput::Request> request,
        std::shared_ptr<igus_rebel_msgs::srv::SetDigitalOutput::Response> response)
    {
        SetDigitalOut(request->output.output, request->output.is_on);
        response->success = true;
    }

    void Rebel::hand_guiding_callback(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response)
    {
        const bool enable = request->data;

        // Inhibit ROS commands before asking the robot to release torque.
        handGuiding.store(true);
        {
            std::lock_guard<std::mutex> lockGuard(aliveLock);
            j1 = j2 = j3 = j4 = j5 = j6 = 0.0f;
        }

        unsigned long previousResponseCount;
        {
            std::lock_guard<std::mutex> lock(zeroTorqueLock);
            previousResponseCount = zeroTorqueResponseCount;
        }

        Command(CriKeywords::COMMAND_ZEROTORQUE + (enable ? " True" : " False"));

        // The message thread receives the authoritative allowed/enabled state.
        std::unique_lock<std::mutex> lock(zeroTorqueLock);
        const bool received = zeroTorqueCondition.wait_for(
            lock, std::chrono::milliseconds(1000),
            [this, previousResponseCount] {
                return zeroTorqueResponseCount != previousResponseCount;
            });

        if (!received)
        {
            response->success = false;
            response->message =
                "Timed out waiting for the Rebel ZeroTorque confirmation; arm controller remains stopped";
            return;
        }

        if (enable)
        {
            if (!zeroTorqueAllowed)
            {
                handGuiding.store(false);
                response->success = false;
                response->message =
                    "ZeroTorque unavailable (allowed=false, enabled=false): enable it in the Rebel robot "
                    "configuration and verify that the joint firmware supports torque mode";
                return;
            }

            if (!zeroTorqueEnabled)
            {
                handGuiding.store(false);
                response->success = false;
                response->message =
                    "ZeroTorque allowed but did not activate (allowed=true, enabled=false): the robot must "
                    "be referenced and its motors enabled";
                return;
            }

            response->success = true;
            response->message = "Hand guiding active: ZeroTorque confirmed";
            return;
        }

        if (zeroTorqueEnabled)
        {
            response->success = false;
            response->message = "Rebel still reports ZeroTorque active; arm controller remains stopped";
            return;
        }

        lock.unlock();

        // CRI specifies that leaving ZeroTorque puts the motors in Disabled state.
        // Re-enable them while jog remains inhibited; the coordinator starts the ROS
        // trajectory controller only after this service succeeds.
        for (double &command : vel_cmd)
        {
            command = 0.0;
        }
        Command(CriKeywords::COMMAND_RESET);
        Command(CriKeywords::COMMAND_ENABLE);
        Command(CriKeywords::COMMAND_MOTIONTYPEJOINT);
        handGuiding.store(false);

        response->success = true;
        response->message = "Hand guiding stopped and Rebel motors re-enabled";
    }

    // Recover from a tripped joint module. CRI leaves the motors disabled after
    // an overcurrent or position-lag fault and silently ignores every jog from
    // then on, so without this the only way back is restarting the stack --
    // which on the rover means losing the whole arm session mid-task.
    //
    // Jog is zeroed BEFORE re-enabling. The controller may still be holding a
    // stale velocity command from the moment of the trip, and re-enabling into
    // that would drive the arm straight back into whatever it stalled against.
    void Rebel::reset_callback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        if (handGuiding.load())
        {
            response->success = false;
            response->message = "Refusing to reset while hand guiding is active";
            return;
        }

        for (double &command : vel_cmd)
        {
            command = 0.0;
        }
        {
            std::lock_guard<std::mutex> lockGuard(aliveLock);
            j1 = j2 = j3 = j4 = j5 = j6 = 0.0f;
        }

        Command(CriKeywords::COMMAND_RESET);
        Command(CriKeywords::COMMAND_ENABLE);
        Command(CriKeywords::COMMAND_MOTIONTYPEJOINT);

        RCLCPP_WARN(rclcpp::get_logger("igus_rebel"),
                    "Reset requested: jog zeroed, motors re-enabled");

        response->success = true;
        response->message = "Reset and Enable sent; motors re-enabled";
    }

    void Rebel::GetReferenceInfo()
    {
        Command(std::string("GetReferencingInfo"));
    }

    void Rebel::Start()
    {
        continueMessage = true;
        messageThread = std::thread(&Rebel::MessageThreadFunction, this);

        rebelSocket->Start();

        // std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Command(CriKeywords::COMMAND_CONNECT); // Gets a CMDERROR in CRI_V17
        Command(CriKeywords::COMMAND_SETACTIVE + " true");
        Command(CriKeywords::COMMAND_RESET);
        Command(CriKeywords::COMMAND_ENABLE);

        continueAlive = true;
        aliveThread = std::thread(&Rebel::AliveThreadFunction, this);

        GetConfig(CriKeywords::CONFIG_GETKINEMATICLIMITS);
        SetControlMode(ControlMode::VELOCITY);
    }

    void Rebel::Stop()
    {
        j1 = 0.0f;
        j2 = 0.0f;
        j3 = 0.0f;
        j4 = 0.0f;
        j5 = 0.0f;
        j6 = 0.0f;

        std::this_thread::sleep_for(std::chrono::milliseconds(aliveWaitMs + 10));

        continueAlive = false;

        if (aliveThread.joinable())
        {
            aliveThread.join();
        }

        Command(CriKeywords::COMMAND_DISABLE);
        // Command(CriKeywords::COMMAND_DISCONNECT);
        Command(CriKeywords::COMMAND_QUIT);

        rebelSocket->Stop();

        continueMessage = false;

        if (messageThread.joinable())
        {
            messageThread.join();
        }
    }
}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
    Igus::Rebel, SystemInterface);

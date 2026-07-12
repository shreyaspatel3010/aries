#include "rclcpp/rclcpp.hpp"

#include "igus_rebel/CriMessages.hpp"
#include "igus_rebel/CriKeywords.hpp"

#define FLOAT_PRINT_PRECISION 3

namespace Igus
{
    namespace CriMessages
    {
        MessageType CriMessage::GetMessageType(const std::string &msg)
        {
            std::string::size_type typeStart;
            // Sometimes there is an additional space in the message for reasons that are not clear to me.
            if (msg.at(0) == ' ')
            {
                typeStart = msg.find(" ", 1) + 1;
            }
            else
            {
                typeStart = msg.find(" ") + 1;
            }
            std::string::size_type typeEnd = msg.find(" ", typeStart);

            std::string typeString = msg.substr(typeStart, typeEnd - typeStart);

            if (typeString == CriKeywords::TYPE_STATUS)
            {
                return MessageType::STATUS;
            }

            if (typeString == CriKeywords::TYPE_OPINFO)
            {
                return MessageType::OPINFO;
            }

            if (typeString == CriKeywords::TYPE_GSIG)
            {
                return MessageType::GSIG;
            }

            if (typeString == CriKeywords::TYPE_GRIPPERSTATE)
            {
                return MessageType::GRIPPERSTATE;
            }

            if (typeString == CriKeywords::TYPE_RUNSTATE)
            {
                return MessageType::RUNSTATE;
            }

            if (typeString == CriKeywords::TYPE_MESSAGE)
            {
                return MessageType::MESSAGE;
            }

            if (typeString == CriKeywords::TYPE_CMD)
            {
                return MessageType::CMD;
            }

            if (typeString == CriKeywords::TYPE_CONFIG)
            {
                return MessageType::CONFIG;
            }

            if (typeString == CriKeywords::TYPE_INFO)
            {
                return MessageType::INFO;
            }

            if (typeString == CriKeywords::TYPE_LOGMSG)
            {
                return MessageType::LOGMSG;
            }

            if (typeString == CriKeywords::TYPE_VARIABLES)
            {
                return MessageType::VARIABLES;
            }

            if (typeString == CriKeywords::TYPE_CMDERROR)
            {
                return MessageType::CMDERROR;
            }

            if (typeString == CriKeywords::TYPE_CMDACK)
            {
                return MessageType::CMDACK;
            }

            if (typeString == CriKeywords::TYPE_CYCLESTAT)
            {
                return MessageType::CYCLESTAT;
            }

            RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Unknown message type: \"%s\"", typeString.c_str());
            return MessageType::UNKNOWN;
        }

        std::string CriMessage::ParseMessageString(const std::string &statusString, const std::string::size_type &begin,
                                                   const std::string::size_type &end, const std::string::size_type &keywordLength)
        {
            int start = begin + keywordLength + 1;
            int length = end - start - 1;

            return statusString.substr(start, length);
        }

        template <class T, std::size_t N>
        void CriMessage::FillArray(std::array<T, N> &array, const std::string &spaceSeparatedValues)
        {
            typename std::array<T, N>::size_type idx = 0;
            std::string::size_type begin = 0;
            std::string::size_type end = spaceSeparatedValues.find(" ", begin + 1);

            while (end != std::string::npos)
            {
                if (idx >= N)
                {
                    RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Parsing error");
                    return;
                }

                std::string value = spaceSeparatedValues.substr(begin, end - begin);

                if (std::is_same<float, T>::value)
                {
                    array.at(idx) = std::stof(value);
                }

                if (std::is_same<int, T>::value)
                {
                    array.at(idx) = std::stoi(value);
                }

                begin = spaceSeparatedValues.find(" ", end);
                end = spaceSeparatedValues.find(" ", begin + 1);
                idx++;
            }

            std::string value = spaceSeparatedValues.substr(begin);

            if (std::is_same<float, T>::value)
            {
                array.at(idx) = std::stof(value);
            }

            if (std::is_same<int, T>::value)
            {
                array.at(idx) = std::stoi(value);
            }
        }

        template <class T>
        void CriMessage::FillVector(std::vector<T> &vector, const std::string &spaceSeparatedValues)
        {
            std::string::size_type begin = 0;
            std::string::size_type end = spaceSeparatedValues.find(" ", begin + 1);

            while (end != std::string::npos)
            {
                std::string value = spaceSeparatedValues.substr(begin, end - begin);

                if (std::is_same<float, T>::value)
                {
                    vector.push_back(std::stof(value));
                }

                if (std::is_same<int, T>::value)
                {
                    vector.push_back(std::stoi(value));
                }

                begin = spaceSeparatedValues.find(" ", end);
                end = spaceSeparatedValues.find(" ", begin + 1);
            }

            std::string value = spaceSeparatedValues.substr(begin);

            if (std::is_same<float, T>::value)
            {
                vector.push_back(std::stof(value));
            }

            if (std::is_same<int, T>::value)
            {
                vector.push_back(std::stoi(value));
            }
        }

        template <class T, std::size_t N>
        std::string CriMessage::ArrayToString(std::array<T, N> &array)
        {
            std::ostringstream msg;
            msg << std::showpoint;
            msg << std::fixed;
            msg << std::setprecision(FLOAT_PRINT_PRECISION);

            for (T val : array)
            {
                msg << val << " ";
            }

            return msg.str();
        }

        template <class T>
        std::string CriMessage::VectorToString(std::vector<T> &vector)
        {
            std::ostringstream msg;
            msg << std::showpoint;
            msg << std::fixed;
            msg << std::setprecision(FLOAT_PRINT_PRECISION);

            for (T val : vector)
            {
                msg << val << " ";
            }

            return msg.str();
        }

        Status::Status(const std::string &messageString) : CriMessage(MessageType::STATUS)
        {
            std::string::size_type modeStart = messageString.find(CriKeywords::STATUS_MODE);
            std::string::size_type posJointSetPointStart = messageString.find(CriKeywords::STATUS_POSJOINTSETPOINT);
            std::string::size_type posJointCurrentStart = messageString.find(CriKeywords::STATUS_POSJOINTCURRENT);
            std::string::size_type posCartRobotStart = messageString.find(CriKeywords::STATUS_POSCARTROBOT);
            std::string::size_type posCartPlattformStart = messageString.find(CriKeywords::STATUS_POSCARTPLATTFORM);
            std::string::size_type overrideValueStart = messageString.find(CriKeywords::STATUS_OVERRIDE);
            std::string::size_type dinStart = messageString.find(CriKeywords::STATUS_DIN);
            std::string::size_type doutStart = messageString.find(CriKeywords::STATUS_DOUT);
            std::string::size_type eStopStart = messageString.find(CriKeywords::STATUS_ESTOP);
            std::string::size_type supplyStart = messageString.find(CriKeywords::STATUS_SUPPLY);
            std::string::size_type currentallStart = messageString.find(CriKeywords::STATUS_CURRENTALL);
            std::string::size_type currentjointsStart = messageString.find(CriKeywords::STATUS_CURRENTJOINTS);
            std::string::size_type errorStart = messageString.find(CriKeywords::STATUS_ERROR);
            std::string::size_type kinstateStart = messageString.find(CriKeywords::STATUS_KINSTATE);

            if (
                !(
                    (modeStart < posJointSetPointStart) &&
                    (posJointSetPointStart < posJointCurrentStart) &&
                    (posJointCurrentStart < posCartRobotStart) &&
                    (posCartRobotStart < posCartPlattformStart) &&
                    (posCartPlattformStart < overrideValueStart) &&
                    (overrideValueStart < dinStart) &&
                    (dinStart < doutStart) &&
                    (doutStart < eStopStart) &&
                    (eStopStart < supplyStart) &&
                    (supplyStart < currentallStart) &&
                    (currentallStart < currentjointsStart) &&
                    (currentjointsStart < errorStart) &&
                    (errorStart < kinstateStart)))
            {
                RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Bad parsing error for message \"%s\"", messageString.c_str());
                return;
            }

            std::string modeString = ParseMessageString(messageString, modeStart, posJointSetPointStart,
                                                        CriKeywords::STATUS_MODE.size());
            std::string posJointSetPointString = ParseMessageString(messageString, posJointSetPointStart,
                                                                    posJointCurrentStart, CriKeywords::STATUS_POSJOINTSETPOINT.size());
            std::string posJointCurrentString = ParseMessageString(messageString, posJointCurrentStart,
                                                                   posCartRobotStart, CriKeywords::STATUS_POSJOINTCURRENT.size());
            std::string posCartRobotString = ParseMessageString(messageString, posCartRobotStart,
                                                                posCartPlattformStart, CriKeywords::STATUS_POSCARTROBOT.size());
            std::string posCartPlattformString = ParseMessageString(messageString, posCartPlattformStart,
                                                                    overrideValueStart, CriKeywords::STATUS_POSCARTPLATTFORM.size());
            std::string overrideValueString = ParseMessageString(messageString, overrideValueStart,
                                                                 dinStart, CriKeywords::STATUS_OVERRIDE.size());
            std::string dinString = ParseMessageString(messageString, dinStart, doutStart,
                                                       CriKeywords::STATUS_DIN.size());
            std::string doutString = ParseMessageString(messageString, doutStart, eStopStart,
                                                        CriKeywords::STATUS_DOUT.size());
            std::string eStopString = ParseMessageString(messageString, eStopStart, supplyStart,
                                                         CriKeywords::STATUS_ESTOP.size());
            std::string supplyString = ParseMessageString(messageString, supplyStart, currentallStart,
                                                          CriKeywords::STATUS_SUPPLY.size());
            std::string currentallString = ParseMessageString(messageString, currentallStart, currentjointsStart,
                                                              CriKeywords::STATUS_CURRENTALL.size());
            std::string currentjointsString = ParseMessageString(messageString, currentjointsStart, errorStart,
                                                                 CriKeywords::STATUS_CURRENTJOINTS.size());
            std::string errorString = ParseMessageString(messageString, errorStart, kinstateStart,
                                                         CriKeywords::STATUS_ERROR.size());
            std::string kinstateString = ParseMessageString(messageString, kinstateStart, messageString.size() + 1,
                                                            CriKeywords::STATUS_KINSTATE.size());

            std::string::size_type errorSummaryEnd = errorString.find(" ");
            errorSummary = errorString.substr(0, errorSummaryEnd);
            std::string errorJointsString = errorString.substr(errorSummaryEnd + 1);

            mode = GetMode(modeString);
            FillArray(posJointSetPoint, posJointSetPointString);
            FillArray(posJointCurrent, posJointCurrentString);
            FillArray(posCartRobot, posCartRobotString);
            FillArray(posCartPlattform, posCartPlattformString);
            try {
                overrideValue = std::stof(overrideValueString);
                din = std::stoi(dinString);     // TODO: Process further to actual meaning
                dout = std::stoi(doutString);   // TODO: Process further to actual meaning
                eStop = std::stoi(eStopString); // TODO: Process further to actual meaning
                supply = std::stoi(supplyString);
                currentall = std::stoi(currentallString);
            } catch (const std::invalid_argument &e) {
                RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Error parsing status message: %s", e.what());
            }
            FillArray(currentjoints, currentjointsString);
            // errorSummary already set above.
            FillArray(errorJoints, errorJointsString); // TODO: Process further to actual meaning
            kinstate = GetKinstate(kinstateString);
        }

        Status::Status() : CriMessage(MessageType::STATUS)
        {
            mode = Mode::UNKNOWN;
            posJointSetPoint.fill(0.0f);
            posJointCurrent.fill(0.0f);
            posCartRobot.fill(0.0f);
            posCartPlattform.fill(0.0f);
            overrideValue = 0.0f;
            din = 0;
            dout = 0;
            eStop = 0;
            supply = 0;
            currentall = 0;
            currentjoints.fill(0);
            errorSummary = "NotInitialized";
            errorJoints.fill(0);
            kinstate = Kinstate::UNKNOWN;
        }

        std::string Status::ModeToString(const Mode &mode)
        {
            switch (mode)
            {
            case Mode::JOINT:
                return "Joint";
            case Mode::CARTBASE:
                return "Cartesian (base)";
            case Mode::CARTTOOL:
                return "Cartesian (tool)";
            case Mode::PLATFORM:
                return "Platform";
            case Mode::FSM:
                return "FSM";
            case Mode::UNKNOWN:
                return "Unknown";
            }

            return std::string();
        }

        void Status::Print()
        {
            std::cout << ToString();
        }

        void Status::Log()
        {
            rclcpp::Clock::SharedPtr clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
            // RCLCPP_INFO_THROTTLE(rclcpp::get_logger("igus_rebel"), *clock_, 60, "Rebel status:\n%s", ToString().c_str());
        }

        std::string Status::ToString()
        {
            std::ostringstream msg;
            msg << std::showpoint
                << std::fixed
                << std::setprecision(FLOAT_PRINT_PRECISION)
                << "            mode: " << Status::ModeToString(mode) << std::endl
                << "posJointSetPoint: " << ArrayToString(posJointSetPoint) << std::endl
                << " posJointCurrent: " << ArrayToString(posJointCurrent) << std::endl
                << "    posCartRobot: " << ArrayToString(posCartRobot) << std::endl
                << "posCartPlattform: " << ArrayToString(posCartPlattform) << std::endl
                << "   overrideValue: " << overrideValue << std::endl
                << "             din: " << din << std::endl
                << "            dout: " << dout << std::endl
                << "           eStop: " << eStop << std::endl
                << "          supply: " << supply << std::endl
                << "      currentall: " << currentall << std::endl
                << "   currentjoints: " << ArrayToString(currentjoints) << std::endl
                << "    errorSummary: " << errorSummary << std::endl
                << "     errorJoints: " << ArrayToString(errorJoints) << std::endl
                << "        kinstate: " << (int)kinstate << std::endl
                << std::endl;

            return msg.str();
        }

        Mode Status::GetMode(const std::string &modeString)
        {
            if (modeString == CriKeywords::MODE_JOINT)
            {
                return Mode::JOINT;
            }

            if (modeString == CriKeywords::MODE_CARTBASE)
            {
                return Mode::CARTBASE;
            }

            if (modeString == CriKeywords::MODE_CARTTOOL)
            {
                return Mode::CARTTOOL;
            }

            if (modeString == CriKeywords::MODE_PLATFORM)
            {
                return Mode::PLATFORM;
            }

            if (modeString == CriKeywords::MODE_FSM)
            {
                return Mode::FSM;
            }

            RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Unknown robot mode during parsing: %s", modeString.c_str());
            return Mode::UNKNOWN;
        }

        Kinstate Status::GetKinstate(const std::string &kinstateString)
        {
            int kinstateInt = std::stoi(kinstateString);

            switch (kinstateInt)
            {
            case (int)Kinstate::NO_ERROR:
                return Kinstate::NO_ERROR;

            case (int)Kinstate::JOINT_LIMIT_MIN:
                return Kinstate::JOINT_LIMIT_MIN;

            case (int)Kinstate::JOINT_LIMIT_MAX:
                return Kinstate::JOINT_LIMIT_MAX;

            case (int)Kinstate::CARTESIAN_SINGULARITY_CENTER:
                return Kinstate::CARTESIAN_SINGULARITY_CENTER;

            case (int)Kinstate::CARTESIAN_SINGULARITY_REACH:
                return Kinstate::CARTESIAN_SINGULARITY_REACH;

            case (int)Kinstate::CARTESIAN_SINGULARITY_WRIST:
                return Kinstate::CARTESIAN_SINGULARITY_WRIST;

            case (int)Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_1:
                return Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_1;

            case (int)Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_2:
                return Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_2;

            case (int)Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_3:
                return Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_3;

            case (int)Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_4:
                return Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_4;

            case (int)Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_5:
                return Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_5;

            case (int)Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_6:
                return Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_6;

            case (int)Kinstate::MOTION_NOT_ALLOWED:
                return Kinstate::MOTION_NOT_ALLOWED;

            default:
                return Kinstate::UNKNOWN;
            }
        }

        Message::Message(const std::string &messageString) : CriMessage(MessageType::MESSAGE)
        {
            std::string::size_type messageStart =
                messageString.find(CriKeywords::TYPE_MESSAGE) + CriKeywords::TYPE_MESSAGE.size() + 1;
            message = messageString.substr(messageStart);
        }

        Command::Command(const std::string &messageString) : CriMessage(MessageType::CMD)
        {
            std::string::size_type commandStart =
                messageString.find(CriKeywords::TYPE_CMD) + CriKeywords::TYPE_CMD.size() + 1;
            command = messageString.substr(commandStart);
        }

        Info::Info(const std::string &messageString) : CriMessage(MessageType::INFO)
        {
            std::string::size_type infoStart =
                messageString.find(CriKeywords::TYPE_INFO) + CriKeywords::TYPE_INFO.size() + 1;
            info = messageString.substr(infoStart);
        }

        LogMsg::LogMsg(const std::string &messageString) : CriMessage(MessageType::LOGMSG)
        {
            std::string::size_type logLvlStart = messageString.find(CriKeywords::TYPE_LOGMSG) + CriKeywords::TYPE_LOGMSG.size() + 1;
            std::string::size_type timestampStart = messageString.find(" ", logLvlStart) + 1;
            std::string::size_type msgStart = messageString.find(" ", timestampStart) + 1;

            logLevel = GetLogLevel(messageString.substr(logLvlStart, (timestampStart - logLvlStart - 1)));
            timestamp = std::stol(messageString.substr(timestampStart, (msgStart - timestampStart - 1)));
            logMsg = messageString.substr(msgStart);
        }

        LogLevel LogMsg::GetLogLevel(const std::string &logLvlString)
        {
            if (logLvlString == CriKeywords::LOG_LEVEL_DEBUG)
            {
                return LogLevel::DEBUG;
            }

            if (logLvlString == CriKeywords::LOG_LEVEL_APP_INFO)
            {
                return LogLevel::APP_INFO;
            }

            if (logLvlString == CriKeywords::LOG_LEVEL_APP_ERROR)
            {
                return LogLevel::APP_ERROR;
            }

            if (logLvlString == CriKeywords::LOG_LEVEL_INFO)
            {
                return LogLevel::INFO;
            }

            if (logLvlString == CriKeywords::LOG_LEVEL_WARN)
            {
                return LogLevel::WARN;
            }

            if (logLvlString == CriKeywords::LOG_LEVEL_ERROR)
            {
                return LogLevel::ERROR;
            }

            if (logLvlString == CriKeywords::LOG_LEVEL_FATAL)
            {
                return LogLevel::FATAL;
            }

            RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Unknown log level during parsing: %s", logLvlString.c_str());
            return LogLevel::UNKNOWN;
        }

        Variables::Variables(const std::string &messageString) : CriMessage(MessageType::VARIABLES),
                                                                 numberVariables(),
                                                                 positionVariables()
        {
            std::regex variables_regex("\\S+");
            auto variables_begin = std::sregex_iterator(messageString.begin(), messageString.end(), variables_regex);
            auto variables_end = std::sregex_iterator();

            std::sregex_iterator it = variables_begin;
            while (it != variables_end)
            {
                std::string matchStr = GetMatchString((*it));

                if (matchStr.compare(CriKeywords::VARIABLES_NR_VARIABLE) == 0)
                {
                    NumberVariable nv;

                    it++;
                    nv.name = GetMatchString(*it);

                    it++;
                    nv.value = std::stod(GetMatchString(*it));

                    numberVariables.push_back(nv);
                }

                if (matchStr.compare(CriKeywords::VARIABLES_POS_VARIABLE) == 0)
                {
                    PositionVariable pv;

                    it++;
                    pv.name = GetMatchString(*it);

                    for (auto i = 0; i < 15; i++)
                    {
                        it++;
                        pv.values[i] = std::stod(GetMatchString(*it));
                    }
                    positionVariables.push_back(pv);
                }

                it++;
            }
        }

        std::string Variables::GetMatchString(const std::smatch &smatch)
        {
            std::string matchStr = smatch.str();
            if (matchStr.length() > 0 && matchStr.back() == ',')
            {
                matchStr.pop_back();
            }
            return matchStr;
        }

        CmdError::CmdError(const std::string &messageString) : CriMessage(MessageType::CMDERROR)
        {
            std::string::size_type commandErrorStart = messageString.find(CriKeywords::TYPE_CMDERROR) + CriKeywords::TYPE_CMDERROR.size();
            std::string::size_type countStart = messageString.find(" ", commandErrorStart) + 1;
            std::string::size_type errorMsgStart = messageString.find(" ", countStart) + 1;

            recjectedCmd = std::stoi(messageString.substr(countStart, errorMsgStart - countStart - 1));
            error = messageString.substr(errorMsgStart);
        }

        CmdAck::CmdAck(const std::string &messageString) : CriMessage(MessageType::CMDACK)
        {
            std::string::size_type commandAckStart = messageString.find(CriKeywords::TYPE_CMDACK) + CriKeywords::TYPE_CMDACK.size();
            acceptedCmd = std::stoi(messageString.substr(commandAckStart));
        }

        Cyclestat::Cyclestat(const std::string &messageString) : CriMessage(MessageType::CYCLESTAT)
        {
            std::string::size_type cyclestatStart = messageString.find(CriKeywords::TYPE_CYCLESTAT) + CriKeywords::TYPE_CYCLESTAT.size();
            std::string::size_type cycletimeStart = messageString.find(" ", cyclestatStart) + 1;
            std::string::size_type workloadStart = messageString.find(" ", cycletimeStart) + 1;

            cycletime = std::stoi(messageString.substr(cycletimeStart, workloadStart - cycletimeStart));
            workload = std::stoi(messageString.substr(workloadStart));
        }

        ConfigType Config::GetConfigType(const std::string &msg)
        {
            std::string::size_type typeStart = msg.find(" ") + 1;
            std::string::size_type typeEnd = msg.find(" ", typeStart);

            std::string typeString = msg.substr(typeStart, typeEnd - typeStart);

            if (typeString != CriKeywords::TYPE_CONFIG)
            {
                RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Got message to parse for config data that was not a config message.");
                return ConfigType::UNKNOWN;
            }

            std::string::size_type configTypeStart = typeEnd + 1;
            std::string::size_type configTypeEnd = msg.find(" ", configTypeStart + 1);

            std::string configTypeString = msg.substr(configTypeStart, configTypeEnd - configTypeStart);

            if (configTypeString == CriKeywords::CONFIG_GETKINEMATICLIMITS_ANSWER)
            {
                return ConfigType::KINEMATICLIMITS;
            }

            RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Unknown config type: \"%s\"", configTypeString.c_str());
            return ConfigType::UNKNOWN;
        }

        KinematicLimits::KinematicLimits(const std::string &messageString) : Config(ConfigType::KINEMATICLIMITS)
        {
            std::string::size_type answerStart = messageString.find(CriKeywords::CONFIG_GETKINEMATICLIMITS_ANSWER);
            answerStart += CriKeywords::CONFIG_GETKINEMATICLIMITS_ANSWER.size() + 1;

            std::vector<float> minMax;
            FillVector(minMax, messageString.substr(answerStart));

            if (minMax.size() % 2 != 0)
            {
                RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Error parsing config message of type %d", (int)configType);
                return;
            }

            for (size_t i = 0; i < minMax.size(); i += 2)
            {
                minMaxPairs.push_back(std::pair<float, float>(minMax.at(i), minMax.at(i + 1)));
            }
        }

        std::string KinematicLimits::ToString()
        {
            std::ostringstream msg;

            for (size_t i = 0; i < minMaxPairs.size() - 1; i++)
            {
                msg << "(" << minMaxPairs.at(i).first << ", " << minMaxPairs.at(i).second << ") ";
            }

            msg << "(" << minMaxPairs.at(minMaxPairs.size() - 1).first << ", " << minMaxPairs.at(minMaxPairs.size() - 1).second << ")";

            return msg.str();
        }

        void KinematicLimits::Print()
        {
            std::cout << "Kinematic limits: " << ToString() << std::endl;
        }
    }
}

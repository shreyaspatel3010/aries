#ifndef CRI_KEYWORDS_HPP_
#define CRI_KEYWORDS_HPP_

#include <string>

namespace Igus
{
    namespace CriKeywords
    {
        const std::string START = "CRISTART";
        const std::string END = "CRIEND";

        const std::string TYPE_STATUS = "STATUS";
        const std::string TYPE_OPINFO = "OPINFO";
        const std::string TYPE_GSIG = "GSIG";
        const std::string TYPE_GRIPPERSTATE = "GRIPPERSTATE";
        const std::string TYPE_RUNSTATE = "RUNSTATE";
        const std::string TYPE_MESSAGE = "MESSAGE";
        const std::string TYPE_CMD = "CMD";
        const std::string TYPE_CONFIG = "CONFIG";
        const std::string TYPE_INFO = "INFO";
        const std::string TYPE_LOGMSG = "LOGMSG";
        const std::string TYPE_VARIABLES = "VARIABLES";
        const std::string TYPE_CMDERROR = "CMDERROR";
        const std::string TYPE_CMDACK = "CMDACK";
        const std::string TYPE_CYCLESTAT = "CYCLESTAT";

        const std::string STATUS_MODE = "MODE";
        const std::string STATUS_POSJOINTSETPOINT = "POSJOINTSETPOINT";
        const std::string STATUS_POSJOINTCURRENT = "POSJOINTCURRENT";
        const std::string STATUS_POSCARTROBOT = "POSCARTROBOT";
        const std::string STATUS_POSCARTPLATTFORM = "POSCARTPLATTFORM";
        const std::string STATUS_OVERRIDE = "OVERRIDE";
        const std::string STATUS_DIN = "DIN";
        const std::string STATUS_DOUT = "DOUT";
        const std::string STATUS_ESTOP = "ESTOP";
        const std::string STATUS_SUPPLY = "SUPPLY";
        const std::string STATUS_CURRENTALL = "CURRENTALL";
        const std::string STATUS_CURRENTJOINTS = "CURRENTJOINTS";
        const std::string STATUS_ERROR = "ERROR";
        const std::string STATUS_KINSTATE = "KINSTATE";
        const std::string STATUS_FRAMEROBOT = "FRAMEROBOT";

        const std::string COMMAND_CONNECT = "Connect";
        const std::string COMMAND_RESET = "Reset";
        const std::string COMMAND_ENABLE = "Enable";
        const std::string COMMAND_DISABLE = "Disable";
        const std::string COMMAND_DISCONNECT = "Disconnect";
        const std::string COMMAND_MOTIONTYPEJOINT = "MotionTypeJoint";
        const std::string COMMAND_MOTIONTYPECARTBASE = "MotionTypeCartBase";
        const std::string COMMAND_QUIT = "QUIT";
        const std::string COMMAND_SETACTIVE = "SetActive";
        const std::string COMMAND_DOUT = "DOUT";

        const std::string CONFIG_GETKINEMATICLIMITS = "GetKinematicLimits";
        const std::string CONFIG_GETKINEMATICLIMITS_ANSWER = "KinematicLimits";

        const std::string MODE_JOINT = "joint";
        const std::string MODE_CARTBASE = "cartbase";
        const std::string MODE_CARTTOOL = "carttool";
        const std::string MODE_PLATFORM = "platform";
        const std::string MODE_FSM = "fsm";

        const std::string LOG_LEVEL_DEBUG = "DEBUG";
        const std::string LOG_LEVEL_APP_INFO = "APP_INFO";
        const std::string LOG_LEVEL_APP_ERROR = "APP_ERROR";
        const std::string LOG_LEVEL_INFO = "INFO";
        const std::string LOG_LEVEL_WARN = "WARN";
        const std::string LOG_LEVEL_ERROR = "ERROR";
        const std::string LOG_LEVEL_FATAL = "FATAL";

        const std::string VARIABLES_NR_VARIABLE = "ValueNrVariable";
        const std::string VARIABLES_POS_VARIABLE = "ValuePosVariable";
    }
}

#endif

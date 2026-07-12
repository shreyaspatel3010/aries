#include "igus_rebel/RebelSocket.hpp"
#include "igus_rebel/CriKeywords.hpp"

#include "rclcpp/rclcpp.hpp"

#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

namespace Igus
{
    //
    // Constructor(s)/ Destructor(s)
    //
    RebelSocket::RebelSocket(const std::string &ip, const int &port, const int &timeout)
        : sock(0),
          ip(ip),
          port(port),
          timeout(timeout),
          unprocessedMessages(),
          fragmentBuffer{0}
    {
    }

    RebelSocket::~RebelSocket()
    {
        Stop();
    }

    //
    // private functions
    //
    void RebelSocket::MakeConnection()
    {
        // Make sure that we do not try to establish the same connection multiple times
        // at the same time.
        std::lock_guard<std::mutex> lockGuard(connectionLock);

        while (connectionNeeded)
        {
            sock = 0;
            struct sockaddr_in serv_addr;

            if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
            {
                RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Socket creation error.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(port);

            // Convert IPv4 and IPv6 addresses from text to binary form
            if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0)
            {
                RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Invalid robot IP address / Address not supported.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
            {
                RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Connection Failed.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            connectionNeeded = false;
            RCLCPP_INFO(rclcpp::get_logger("igus_rebel"), "Connected to ReBeL at %s:%d", ip.c_str(), port);
        }
    }

    void RebelSocket::SeparateMessages(const char *msg)
    {
        const char *start;
        const char *end = msg;

        while (true)
        {
            start = std::strstr(end, CriKeywords::START.c_str());

            if (start == nullptr)
            {
                break;
            }

            end = std::strstr(start, CriKeywords::END.c_str());

            if (end == nullptr)
            {
                // Found a start without end.
                const char *remainingStart = start + CriKeywords::START.size();
                const char *remainingEnd = std::strchr(remainingStart, '\0');

                if (remainingEnd != nullptr)
                {
                    fragmentLength = remainingEnd - remainingStart;

                    for (int i = 0; i < fragmentLength; i++)
                    {
                        fragmentBuffer[i] = *(remainingStart + i);
                    }
                }
                else
                {
                    RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Socket read was not null-terminated, somehow.");
                }

                break;
            }

            {
                std::lock_guard<std::mutex> lockGuard(messageLock);
                unprocessedMessages.push_front(
                    std::string(
                        start + CriKeywords::START.size() + 1,
                        end - (start + CriKeywords::START.size() + 1) - 1));
            }
        }
    }

    void RebelSocket::ReceiveThreadFunction()
    {
        RCLCPP_DEBUG(rclcpp::get_logger("igus_rebel"), "Starting to receive messages from robot.");

        char buffer[bufferSize * 2] = {0};

        while (continueReceive)
        {
            if (connectionNeeded)
            {
                MakeConnection();
            }

            int valread = read(sock, buffer + fragmentLength, bufferSize);

            if (!IsSocketOk())
            {
                connectionNeeded = true;
            }
            else
            {
                if (valread == 0)
                {
                    RCLCPP_WARN(rclcpp::get_logger("igus_rebel"), "Empty message received");
                    connectionNeeded = true;
                }
                else
                {
                    if (fragmentLength > 0)
                    {
                        for (int i = 0; i < fragmentLength; i++)
                        {
                            buffer[i] = fragmentBuffer[i];
                        }
                        fragmentLength = 0;
                    }

                    SeparateMessages(buffer);
                    for (int i = 0; i < bufferSize * 2; i++)
                    {
                        buffer[i] = 0;
                    }
                }
            }
        }

        RCLCPP_DEBUG(rclcpp::get_logger("igus_rebel"), "Stopped to receive messages from robot.");
    }

    // Make sure that we do not fill our entire memory with messages from the robot in case something
    // goes wrong with processing them.
    // Also, later we should just stop the robot here, because this could be unsafe.
    void RebelSocket::ListCheckThreadFunction()
    {
        RCLCPP_DEBUG(rclcpp::get_logger("igus_rebel"), "Starting to check if the message list is being processed.");

        while (continueReceive)
        {
            if (unprocessedMessages.size() > maxUnprocessedMessages)
            {
                RCLCPP_WARN(rclcpp::get_logger("igus_rebel"), "Robot messages are not processed fast enough. Discarding messages.");

                while (unprocessedMessages.size() > (maxUnprocessedMessages * 0.9))
                {
                    unprocessedMessages.pop_back();
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(listCheckWaitMs));
        }

        RCLCPP_DEBUG(rclcpp::get_logger("igus_rebel"), "Stopped to check if the message list is being processed.");
    }

    bool RebelSocket::IsSocketOk()
    {
        int error = 0;
        socklen_t len = sizeof(error);
        int retval = getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &len);

        if (retval != 0)
        {
            RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Error getting socket error code: %s", strerror(retval));
            return false;
        }

        if (error != 0)
        {
            RCLCPP_ERROR(rclcpp::get_logger("igus_rebel"), "Socket error: %s", strerror(error));
            return false;
        }

        return true;
    }

    //
    // public functions
    //
    void RebelSocket::Start()
    {
        connectionNeeded = true;
        continueReceive = true;

        listCheckThread = std::thread(&RebelSocket::ListCheckThreadFunction, this);
        receiveThread = std::thread(&RebelSocket::ReceiveThreadFunction, this);
    }

    void RebelSocket::Stop()
    {
        connectionNeeded = false;
        continueReceive = false;

        if (receiveThread.joinable())
        {
            receiveThread.join();
        }

        if (listCheckThread.joinable())
        {
            listCheckThread.join();
        }
    }

    bool RebelSocket::HasMessage()
    {
        return unprocessedMessages.size() > 0;
    }

    std::string RebelSocket::GetMessage()
    {
        std::lock_guard<std::mutex> lockGuard(messageLock);

        if (!HasMessage())
        {
            return "";
        }

        std::string msg = unprocessedMessages.back();
        unprocessedMessages.pop_back();

        return msg;
    }

    void RebelSocket::SendMessage(const std::string &msg)
    {
        std::lock_guard<std::mutex> lockGuard(socketWriteLock);

        if (connectionNeeded)
        {
            MakeConnection();
        }

        int sent = send(sock, msg.c_str(), msg.length(), 0);

        if (!IsSocketOk())
        {
            connectionNeeded = true;
        }

        if (sent < 0)
        {
            connectionNeeded = true;
        }
    }
}

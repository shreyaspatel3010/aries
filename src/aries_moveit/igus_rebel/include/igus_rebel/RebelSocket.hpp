#ifndef REBEL_SOCKET_HPP_
#define REBEL_SOCKET_HPP_
#include "rclcpp/rclcpp.hpp"
#include <stdio.h>
#include <sys/socket.h>
#include <string>
#include <list>

#include <thread>
#include <mutex>

namespace Igus
{
    class RebelSocket
    {
    private:
        int sock;
        std::string ip;
        int port;
        int timeout;
        std::list<std::string> unprocessedMessages;

        bool continueReceive = false;
        std::thread receiveThread;
        std::thread listCheckThread;
        std::mutex socketWriteLock;
        std::mutex connectionLock;
        std::mutex messageLock;
        unsigned long maxUnprocessedMessages = 25;
        int listCheckWaitMs = 500;

        bool connectionNeeded = false;
        static const int bufferSize = 4096;

        char fragmentBuffer[bufferSize];
        int fragmentLength = 0;

        void MakeConnection();
        void SeparateMessages(const char *);

        void ReceiveThreadFunction();
        void ListCheckThreadFunction();

        bool IsSocketOk();

    public:
        RebelSocket(const std::string &, const int &, const int &);
        ~RebelSocket();

        void Start();
        void Stop();
        bool HasMessage();
        std::string GetMessage();
        void SendMessage(const std::string &);
    };
}

#endif

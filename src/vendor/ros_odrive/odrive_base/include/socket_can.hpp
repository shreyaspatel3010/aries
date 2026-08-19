#ifndef SOCKET_CAN_HPP
#define SOCKET_CAN_HPP

#include "epoll_event_loop.hpp"
#include <linux/can.h>
#include <linux/can/raw.h>
#include <mutex>
#include <string>
#include <functional>

using FrameProcessor = std::function<void(const can_frame&)>;

class SocketCanIntf {
public:
    bool init(const std::string& interface, EpollEventLoop* event_loop, FrameProcessor frame_processor);

    // Re-open and re-bind the socket to the interface init() was given.
    //
    // A raw CAN socket is bound to an interface *index*, not to a name. When a
    // USB CAN adapter is unplugged the kernel unregisters that netdev, wakes
    // this socket with EPOLLERR, and every later send fails with ENXIO. Plugging
    // the adapter back in creates a new netdev with a new index, so the old
    // socket stays dead no matter what happens on the bus: the only recovery
    // without restarting the process is to bind a fresh socket.
    bool reinit();

    void deinit();
    bool send_can_frame(const can_frame& frame);

    // True when a socket is open and no fatal error has been seen on it.
    bool is_ready() const;
    // Interface index this socket is bound to, or -1 when it is closed.
    int interface_index() const;

    bool read_nonblocking();
private:
    mutable std::recursive_mutex mutex_;
    std::string interface_;
    int socket_id_ = -1;
    int if_index_ = -1;
    EpollEventLoop* event_loop_ = nullptr;
    EpollEventLoop::EvtId socket_evt_id_ = nullptr;
    FrameProcessor frame_processor_;
    bool broken_ = true;
    bool send_error_logged_ = false;

    bool open_locked();
    void close_locked();
    void on_socket_event(uint32_t mask);
    void process_can_frame(const can_frame& frame) {
        frame_processor_(frame);
    }
};

#endif  // SOCKET_CAN_HPP

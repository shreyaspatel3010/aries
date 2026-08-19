#include "socket_can.hpp"
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/types.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <cerrno>
#include <net/if.h>
#include <sys/ioctl.h>

namespace {

// Errors that mean this socket can never transmit again: the interface it is
// bound to is gone (ENXIO/ENODEV), the link is administratively down
// (ENETDOWN), or the descriptor was already closed by the error handler
// (EBADF). Everything else — a full transmit queue above all — is transient and
// must not tear the socket down.
bool is_fatal_send_error(int err) {
    return err == ENXIO || err == ENODEV || err == ENETDOWN || err == EBADF ||
           err == ENOTCONN || err == EPIPE;
}

}  // namespace

bool SocketCanIntf::init(const std::string& interface, EpollEventLoop* event_loop, FrameProcessor frame_processor) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    interface_ = interface;
    event_loop_ = event_loop;
    frame_processor_ = std::move(frame_processor);
    return open_locked();
}

bool SocketCanIntf::reinit() {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    if (event_loop_ == nullptr || !frame_processor_) {
        std::cerr << "Cannot reconnect CAN interface before init()" << std::endl;
        return false;
    }
    close_locked();
    return open_locked();
}

bool SocketCanIntf::open_locked() {
    socket_id_ = socket(PF_CAN, SOCK_RAW | SOCK_NONBLOCK, CAN_RAW);
    if (socket_id_ == -1) {
        std::cerr << "Failed to create socket" << std::endl;
        return false;
    }

    struct ifreq ifr;
    std::memset(&ifr, 0, sizeof(ifr));
    std::strncpy(ifr.ifr_name, interface_.c_str(), IFNAMSIZ - 1);
    if (ioctl(socket_id_, SIOCGIFINDEX, &ifr) == -1) {
        std::cerr << "Failed to get interface index for " << interface_ << ": " << std::strerror(errno) << std::endl;
        close(socket_id_);
        socket_id_ = -1;
        return false;
    }

    struct sockaddr_can addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(socket_id_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == -1) {
        std::cerr << "Failed to bind socket to " << interface_ << ": " << std::strerror(errno) << std::endl;
        close(socket_id_);
        socket_id_ = -1;
        return false;
    }

    struct msghdr message = {
        .msg_name = nullptr,
        .msg_namelen = 0,
        .msg_iov = nullptr,
        .msg_iovlen = 0,
        .msg_control = nullptr,
        .msg_controllen = 0,
        .msg_flags = 0
    };

    int retcode = recvmsg(socket_id_, &message, 0);
    if (retcode < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        close(socket_id_);
        socket_id_ = -1;
        return false;
    }

    if (!event_loop_->register_event(&socket_evt_id_, socket_id_, EPOLLIN, [this](uint32_t mask) { on_socket_event(mask); })) {
        std::cerr << "Failed to register socket with event loop" << std::endl;
        close(socket_id_);
        socket_id_ = -1;
        return false;
    }

    if_index_ = ifr.ifr_ifindex;
    broken_ = false;
    send_error_logged_ = false;
    return true;
}

void SocketCanIntf::close_locked() {
    if (socket_evt_id_ != nullptr) {
        event_loop_->deregister_event(socket_evt_id_);
        socket_evt_id_ = nullptr;
    }
    if (socket_id_ >= 0) {
        close(socket_id_);
    }
    // Never leave the old number behind: a stale descriptor gets recycled by
    // the next open() in this process, and a send would then write CAN frames
    // into an unrelated file.
    socket_id_ = -1;
    if_index_ = -1;
    broken_ = true;
}

void SocketCanIntf::deinit() {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    close_locked();
}

bool SocketCanIntf::is_ready() const {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return socket_id_ >= 0 && !broken_;
}

int SocketCanIntf::interface_index() const {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return if_index_;
}

bool SocketCanIntf::send_can_frame(const can_frame& frame) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    if (socket_id_ < 0 || broken_) {
        if (!send_error_logged_) {
            send_error_logged_ = true;
            std::cerr << "Dropping CAN frame 0x" << std::hex << frame.can_id << std::dec
                      << ": " << interface_ << " is disconnected — call the reconnect service"
                      << std::endl;
        }
        return false;
    }

    ssize_t nbytes = write(socket_id_, &frame, sizeof(frame));
    if (nbytes == -1) {
        const int send_errno = errno;
        if (!send_error_logged_) {
            send_error_logged_ = true;
            std::cerr << "Failed to send CAN frame 0x" << std::hex
                      << frame.can_id << std::dec << ": "
                      << std::strerror(send_errno)
                      << " (errno " << send_errno << ")" << std::endl;
        }
        if (is_fatal_send_error(send_errno)) {
            close_locked();
        }
        return false;
    }

    send_error_logged_ = false;
    return true;
}

void SocketCanIntf::on_socket_event(uint32_t mask) {
    if (mask & EPOLLIN) {
        while (read_nonblocking() && is_ready());
    }
    if (mask & EPOLLERR) {
        std::cerr << "interface disappeared" << std::endl;
        deinit();
        return;
    }
    if (mask & ~(EPOLLIN | EPOLLERR)) {
        std::cerr << "unexpected event " << mask << std::endl;
        deinit();
        return;
    }
    return;
}

bool SocketCanIntf::read_nonblocking() {
    struct can_frame frame;
    struct cmsghdr ctrlmsg;

    struct iovec vec = {.iov_base = &frame, .iov_len = sizeof(frame)};
    struct msghdr message = {
        .msg_name = nullptr,
        .msg_namelen = 0,
        .msg_iov = &vec, 
        .msg_iovlen = 1,
        .msg_control = &ctrlmsg,
        .msg_controllen = sizeof(ctrlmsg),
        .msg_flags = 0
        };

    ssize_t n_received;
    {
        std::lock_guard<std::recursive_mutex> guard(mutex_);
        if (socket_id_ < 0) return false;
        n_received = recvmsg(socket_id_, &message, MSG_DONTWAIT);
    }

    if (n_received < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // std::cerr << "no message received" << std::endl;
            return false;
        } else {
            std::cerr << "Socket read failed: " << std::strerror(errno) << std::endl;
            return false;
        }
    }

    if (n_received < static_cast<ssize_t>(sizeof(struct can_frame))) {
        std::cerr << "invalid message length " << n_received << std::endl;
        return true;
    }

    // Dispatched with the socket lock released: a frame handler must never be
    // able to hold off a reconnect coming from the ROS executor thread.
    process_can_frame(frame);
    return true;
}

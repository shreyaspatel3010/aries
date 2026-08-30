#include "st3215_gripper_hardware/sts_bus.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <chrono>
#include <vector>

namespace st3215_gripper_hardware
{

namespace
{
constexpr uint8_t INST_PING = 0x01;
constexpr uint8_t INST_READ = 0x02;
constexpr uint8_t INST_WRITE = 0x03;

speed_t baud_constant(int baud)
{
  switch (baud) {
    case 1000000: return B1000000;
    case 500000: return B500000;
    case 460800: return B460800;
    case 250000: return B230400;   // no B250000; caller should not use it
    case 115200: return B115200;
    case 57600: return B57600;
    case 38400: return B38400;
    case 19200: return B19200;
    case 9600: return B9600;
    default: return 0;
  }
}
}  // namespace

StsBus::~StsBus() { close(); }

bool StsBus::open(const std::string & port, int baud, int timeout_ms)
{
  close();
  timeout_ms_ = timeout_ms;

  fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    last_error_ = "open(" + port + "): " + std::strerror(errno);
    return false;
  }

  const speed_t speed = baud_constant(baud);
  if (speed == 0) {
    last_error_ = "unsupported baud " + std::to_string(baud);
    close();
    return false;
  }

  termios tio{};
  if (tcgetattr(fd_, &tio) != 0) {
    last_error_ = std::string("tcgetattr: ") + std::strerror(errno);
    close();
    return false;
  }
  cfmakeraw(&tio);
  cfsetispeed(&tio, speed);
  cfsetospeed(&tio, speed);
  tio.c_cflag |= (CLOCAL | CREAD);
  tio.c_cflag &= ~static_cast<tcflag_t>(CSTOPB);   // 8N1
  tio.c_cflag &= ~static_cast<tcflag_t>(PARENB);
  tio.c_cflag &= ~static_cast<tcflag_t>(CRTSCTS);
  // Reads are deadlined with poll() rather than VTIME, so the port itself is
  // fully non-blocking.  VTIME's granularity is 100 ms, which is twenty times
  // the whole control period.
  tio.c_cc[VMIN] = 0;
  tio.c_cc[VTIME] = 0;
  if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
    last_error_ = std::string("tcsetattr: ") + std::strerror(errno);
    close();
    return false;
  }
  tcflush(fd_, TCIOFLUSH);
  last_error_.clear();
  return true;
}

void StsBus::close()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool StsBus::read_exact(uint8_t * dst, size_t n)
{
  using clock = std::chrono::steady_clock;
  const auto deadline = clock::now() + std::chrono::milliseconds(timeout_ms_);
  size_t got = 0;
  while (got < n) {
    const auto now = clock::now();
    if (now >= deadline) {
      ++timeouts_;
      return false;
    }
    const auto left =
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    pollfd p{fd_, POLLIN, 0};
    const int rc = ::poll(&p, 1, static_cast<int>(left) + 1);
    if (rc <= 0) {
      if (rc < 0 && errno == EINTR) { continue; }
      ++timeouts_;
      return false;
    }
    const ssize_t r = ::read(fd_, dst + got, n - got);
    if (r > 0) {
      got += static_cast<size_t>(r);
    } else if (r < 0 && errno != EAGAIN && errno != EINTR) {
      last_error_ = std::string("read: ") + std::strerror(errno);
      return false;
    }
  }
  return true;
}

bool StsBus::transact(uint8_t id, uint8_t inst, const uint8_t * params, uint8_t n_params,
                      uint8_t * reply, uint8_t n_reply)
{
  if (fd_ < 0) { return false; }

  std::vector<uint8_t> tx;
  tx.reserve(static_cast<size_t>(n_params) + 6);
  tx.push_back(0xFF);
  tx.push_back(0xFF);
  tx.push_back(id);
  tx.push_back(static_cast<uint8_t>(n_params + 2));
  tx.push_back(inst);
  for (uint8_t i = 0; i < n_params; ++i) { tx.push_back(params[i]); }
  uint32_t sum = 0;
  for (size_t i = 2; i < tx.size(); ++i) { sum += tx[i]; }
  tx.push_back(static_cast<uint8_t>(~sum & 0xFF));

  // The bus is half duplex and the adapter echoes nothing, but a previous
  // reply that arrived late is still sitting in the buffer and would be parsed
  // as this one's.  Drop it before asking.
  tcflush(fd_, TCIFLUSH);

  size_t written = 0;
  while (written < tx.size()) {
    const ssize_t w = ::write(fd_, tx.data() + written, tx.size() - written);
    if (w > 0) {
      written += static_cast<size_t>(w);
    } else if (w < 0 && errno != EAGAIN && errno != EINTR) {
      last_error_ = std::string("write: ") + std::strerror(errno);
      return false;
    }
  }

  uint8_t head[5];
  if (!read_exact(head, sizeof(head))) { return false; }
  if (head[0] != 0xFF || head[1] != 0xFF) {
    ++bad_checksums_;
    return false;
  }
  const int n_body = static_cast<int>(head[3]) - 1;   // params + checksum
  if (n_body < 1 || n_body - 1 != static_cast<int>(n_reply)) {
    ++bad_checksums_;
    return false;
  }
  std::vector<uint8_t> body(static_cast<size_t>(n_body));
  if (!read_exact(body.data(), body.size())) { return false; }

  uint32_t chk = head[2] + head[3] + head[4];
  for (size_t i = 0; i + 1 < body.size(); ++i) { chk += body[i]; }
  if (static_cast<uint8_t>(~chk & 0xFF) != body.back()) {
    ++bad_checksums_;
    return false;
  }
  if (head[4] != 0) { last_servo_error_ = head[4]; }
  for (uint8_t i = 0; i < n_reply; ++i) { reply[i] = body[i]; }
  return true;
}

bool StsBus::ping(uint8_t id)
{
  return transact(id, INST_PING, nullptr, 0, nullptr, 0);
}

bool StsBus::read8(uint8_t id, uint8_t addr, uint8_t & out)
{
  const uint8_t p[2] = {addr, 1};
  return transact(id, INST_READ, p, 2, &out, 1);
}

bool StsBus::read16(uint8_t id, uint8_t addr, uint16_t & out)
{
  uint8_t d[2];
  const uint8_t p[2] = {addr, 2};
  if (!transact(id, INST_READ, p, 2, d, 2)) { return false; }
  out = static_cast<uint16_t>(d[0] | (d[1] << 8));
  return true;
}

bool StsBus::read_block(uint8_t id, uint8_t addr, uint8_t count, uint8_t * out)
{
  const uint8_t p[2] = {addr, count};
  return transact(id, INST_READ, p, 2, out, count);
}

bool StsBus::write8(uint8_t id, uint8_t addr, uint8_t value)
{
  const uint8_t p[2] = {addr, value};
  return transact(id, INST_WRITE, p, 2, nullptr, 0);
}

bool StsBus::write16(uint8_t id, uint8_t addr, uint16_t value)
{
  const uint8_t p[3] = {addr, static_cast<uint8_t>(value & 0xFF),
                        static_cast<uint8_t>((value >> 8) & 0xFF)};
  return transact(id, INST_WRITE, p, 3, nullptr, 0);
}

std::string status_flags(uint8_t status)
{
  if (status == 0) { return ""; }
  static const struct { uint8_t bit; const char * name; } kBits[] = {
    {STATUS_VOLTAGE, "voltage"},
    {STATUS_SENSOR, "sensor"},
    {STATUS_TEMPERATURE, "overheat"},
    {STATUS_CURRENT, "overcurrent"},
    {STATUS_ANGLE, "angle"},
    {STATUS_OVERLOAD, "overload"},
  };
  std::string out;
  for (const auto & b : kBits) {
    if (status & b.bit) {
      if (!out.empty()) { out += "+"; }
      out += b.name;
    }
  }
  // Bits 6 and 7 are not in the table. Say so rather than dropping them: an
  // unexplained bit is exactly the thing worth seeing.
  if (status & 0xC0u) {
    if (!out.empty()) { out += "+"; }
    out += "unknown";
  }
  return out;
}

}  // namespace st3215_gripper_hardware

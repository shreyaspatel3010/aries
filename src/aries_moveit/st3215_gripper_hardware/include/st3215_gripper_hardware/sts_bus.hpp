#pragma once

// Feetech SMS/STS half-duplex serial protocol, enough of it to drive one
// ST3215 in position mode.
//
// This is the same wire format scripts/st3215_test.py speaks, and the register
// numbers and sign conventions below were verified against the physical servo
// with that script.  Keep the two in step: if a register moves here it has
// moved there too.
//
// PACKET
//     request   FF FF  ID  LEN  INST  PARAM...  CHK
//     response  FF FF  ID  LEN  ERR   PARAM...  CHK
//     LEN = number of params + 2
//     CHK = ~(ID + LEN + INST/ERR + sum(PARAM)) & 0xFF
//
// 16-bit registers are LITTLE-endian.  Signed registers are NOT two's
// complement: the direction sits in one flag bit and the rest is magnitude.
// Which bit depends on the register - speed and current use bit 15, present
// load uses bit 10 because it is a 0-1000 PWM duty and bit 10 is free.  Decode
// with sign_magnitude() and never with a cast.

#include <cstdint>
#include <string>

namespace st3215_gripper_hardware
{

// SMS/STS control table.  EPROM below 40, SRAM at and above it.
enum : uint8_t
{
  // EPROM protection limits.  The servo enforces these itself: exceed one and
  // it raises the matching bit in REG_STATUS and unloads torque according to
  // its unloading condition (reg 19).  We only ever READ them - they are the
  // servo's own idea of "too hot" / "too low" and are what a live display
  // must compare against, rather than numbers invented here.
  REG_MAX_TEMPERATURE = 13,   // deg C, 1 per LSB
  REG_MAX_VOLTAGE = 14,       // 0.1 V per LSB
  REG_MIN_VOLTAGE = 15,       // 0.1 V per LSB
  REG_PROTECTION_CURRENT = 28,  // 16-bit, 6.5 mA per LSB
  REG_MODE = 33,          // EPROM. 0 = position, 1 = wheel.
  REG_TORQUE_ENABLE = 40,
  REG_ACCELERATION = 41,
  REG_GOAL_POSITION = 42,
  REG_GOAL_TIME = 44,
  REG_GOAL_SPEED = 46,
  REG_TORQUE_LIMIT = 48,
  REG_LOCK = 55,
  REG_PRESENT_POSITION = 56,
  REG_PRESENT_SPEED = 58,
  REG_PRESENT_LOAD = 60,
  REG_PRESENT_VOLTAGE = 62,
  REG_PRESENT_TEMPERATURE = 63,
  REG_STATUS = 65,
  REG_MOVING = 66,
  REG_PRESENT_CURRENT = 69,
};

// REG_PRESENT_POSITION .. REG_PRESENT_CURRENT+1 inclusive, so ONE block read
// gets position, speed, load, voltage, temperature, status, moving and
// current.  It is 15 bytes instead of 6 - about 150 us more wire time at
// 1 Mbaud - and it is still one USB round trip, which is the part that costs.
// Reading the telemetry registers separately would triple the poll's latency
// budget for numbers that only feed a display.
constexpr uint8_t TELEMETRY_BLOCK_START = REG_PRESENT_POSITION;
constexpr uint8_t TELEMETRY_BLOCK_LEN = REG_PRESENT_CURRENT + 2 - REG_PRESENT_POSITION;  // 15

/// Offset of \p reg within a block read starting at TELEMETRY_BLOCK_START.
constexpr int tel(uint8_t reg) { return reg - TELEMETRY_BLOCK_START; }

// Register units.
constexpr double CURRENT_LSB_MA = 6.5;   // present current and protection current
constexpr double VOLTAGE_LSB_V = 0.1;    // present voltage and both voltage limits
constexpr double LOAD_FULL_SCALE = 1000.0;  // present load is a 0-1000 PWM duty

// REG_STATUS bits.  Same layout as the unloading condition (reg 19), the LED
// alarm condition (reg 20), and the ERR byte every reply carries.
//
// TAKEN FROM THE WAVESHARE ST3215 MEMORY TABLE, not provoked on this servo -
// deliberately fault-testing a gripper servo to confirm a bit number is not
// worth the servo.  So the display names the bits it decodes but always shows
// the raw byte beside them: if one is ever seen, the number is checkable
// against the table even if the label here is wrong.
enum : uint8_t
{
  STATUS_VOLTAGE = 1u << 0,
  STATUS_SENSOR = 1u << 1,
  STATUS_TEMPERATURE = 1u << 2,
  STATUS_CURRENT = 1u << 3,
  STATUS_ANGLE = 1u << 4,
  STATUS_OVERLOAD = 1u << 5,
};

/// Human-readable list of the set bits in a REG_STATUS byte, "" when clear.
std::string status_flags(uint8_t status);

constexpr int STEPS_PER_REV = 4096;

// Stay clear of the hard 0 / 4095 stops, the same margin st3215_test.py keeps.
constexpr int POS_MIN = 50;
constexpr int POS_MAX = 4045;

/// Decode the servo's flag-bit-plus-magnitude encoding at \p bit.
inline int sign_magnitude(uint16_t raw, int bit)
{
  const uint16_t mask = static_cast<uint16_t>(1u << bit);
  return (raw & mask) ? -static_cast<int>(raw & ~mask) : static_cast<int>(raw);
}

/// One servo bus on one serial port.  Not thread safe: the hardware component
/// touches it from a single I/O thread.
class StsBus
{
public:
  StsBus() = default;
  ~StsBus();

  StsBus(const StsBus &) = delete;
  StsBus & operator=(const StsBus &) = delete;

  /// Open \p port at \p baud in raw 8N1 with a \p timeout_ms read deadline.
  /// Returns false and fills last_error() on failure.
  bool open(const std::string & port, int baud, int timeout_ms);
  void close();
  bool is_open() const { return fd_ >= 0; }

  bool ping(uint8_t id);
  bool read8(uint8_t id, uint8_t addr, uint8_t & out);
  bool read16(uint8_t id, uint8_t addr, uint16_t & out);
  /// Read \p count consecutive bytes.  One round trip instead of several,
  /// which is what makes a position+speed+load poll affordable in the loop.
  bool read_block(uint8_t id, uint8_t addr, uint8_t count, uint8_t * out);
  bool write8(uint8_t id, uint8_t addr, uint8_t value);
  bool write16(uint8_t id, uint8_t addr, uint16_t value);

  const std::string & last_error() const { return last_error_; }
  /// Packets dropped because the checksum did not verify.  A desynchronised
  /// reply otherwise reads as a perfectly plausible number, which is exactly
  /// how a fast position poll ends up producing impossible velocities.
  uint64_t bad_checksums() const { return bad_checksums_; }
  uint64_t timeouts() const { return timeouts_; }
  /// Error byte from the most recent reply that carried one.
  uint8_t last_servo_error() const { return last_servo_error_; }

private:
  bool transact(uint8_t id, uint8_t inst, const uint8_t * params, uint8_t n_params,
                uint8_t * reply, uint8_t n_reply);
  bool read_exact(uint8_t * dst, size_t n);

  int fd_{-1};
  int timeout_ms_{5};
  std::string last_error_;
  uint64_t bad_checksums_{0};
  uint64_t timeouts_{0};
  uint8_t last_servo_error_{0};
};

}  // namespace st3215_gripper_hardware

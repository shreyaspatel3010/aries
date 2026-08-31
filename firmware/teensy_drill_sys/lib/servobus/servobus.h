// ST3215 bus-servo bridge -- TEMPORARY, until the USB bus-servo adapter is
// replaced.
//
// WHAT THIS IS FOR
// The secondary gripper is a single ST3215 on a one-wire half-duplex TTL bus at
// 1 Mbaud. It is normally reached through a URT-1 or Waveshare USB adapter, and
// the host end -- st3215_gripper_hardware's StsBus, and scripts/st3215_test.py
// -- just opens that adapter's serial port and speaks STS packets down it.
//
// On 2026-08-31 that adapter stopped enumerating (USB -71 on every port, board
// running hot). This class lets THIS BOARD stand in for it: bytes arriving on a
// second USB serial device are put on the servo bus, and the servo's answer is
// handed back. It is a wire, not a protocol -- nothing here parses an STS
// packet, so every host-side tool keeps working with no changes at all beyond
// pointing servo_bus.port at the second port this board presents.
//
// DELETE THIS WHOLE DIRECTORY when the replacement adapter arrives, along with
// the three lines in main.cpp and -D USB_DUAL_SERIAL in platformio.ini. It is a
// stopgap on the gripper's control path and does not belong there permanently:
// the bus wire has to run the length of the rover, past three H-bridges, to
// reach a servo that a £10 adapter could reach with 100 mm of cable.
//
// WHY IT NEEDS USB_DUAL_SERIAL
// Serial is micro-ROS's transport and nothing else may touch it (see the note
// at the top of main.cpp). USB_DUAL_SERIAL makes the board present a SECOND CDC
// device on the same cable, SerialUSB1, which is what this bridge uses. On the
// host it appears as another /dev/ttyACM*.
//
// WHY Serial1, HALF DUPLEX
// Pin 1 is Serial1 TX, and in single-wire mode the LPUART drives and reads that
// one pin -- see the ST3215_BUS block in pins.h for why the TX pin is the one
// that matters and what full duplex would cost.
#ifndef SERVOBUS_H
#define SERVOBUS_H

#include <Arduino.h>

#if !defined(USB_DUAL_SERIAL) && !defined(USB_TRIPLE_SERIAL)
#error "servobus needs -D USB_DUAL_SERIAL: SerialUSB1 does not exist without it, and Serial belongs to micro-ROS."
#endif

class ServoBus
{
public:
  // Opens the bus half-duplex and works out whether the UART echoes what it
  // transmits. Safe to call with nothing attached and with the bus unpowered.
  void init();

  // Moves bytes both ways. Non-blocking apart from the transmit itself, which
  // at 1 Mbaud is ~10 us per byte. Call it every loop().
  void poll();

  // True when the UART hands back its own transmission -- measured at init(),
  // not assumed. Reported on the diagnostic topic so a silent bus can be told
  // apart from a bridge that is eating the replies.
  bool echoes() const { return echoes_; }

  // Bytes relayed since boot, host->bus and bus->host. Zero in both directions
  // means nothing has ever spoken to this bridge.
  uint32_t to_bus() const { return to_bus_; }
  uint32_t to_host() const { return to_host_; }

private:
  void discard_echo(int count);

  static const uint32_t kBaud = 1000000;
  // Longest STS packet in use here is well under 32 bytes; 64 is a whole USB
  // packet's worth and costs nothing on a board with 512 KB of RAM2.
  static const int kChunk = 64;
  // The echo is generated WHILE transmitting, so by the time flush() returns
  // every echoed byte is already buffered. This bound only exists so a wrong
  // guess cannot wedge the drill's control loop.
  static const uint32_t kEchoTimeoutUs = 500;

  bool echoes_ = false;
  uint32_t to_bus_ = 0;
  uint32_t to_host_ = 0;
};

#endif  // SERVOBUS_H

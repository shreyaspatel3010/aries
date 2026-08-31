#include "servobus.h"

#include "pins.h"

// The bridge drives Serial1, whose TX pad IS pin 1 on a Teensy 4.1. If
// ST3215_BUS is ever moved, the UART has to move with it -- fail the build
// rather than quietly bridging a pin nobody wired.
static_assert(!PIN_IS_ASSIGNED(ST3215_BUS) || ST3215_BUS == 1,
              "ServoBus uses Serial1 (TX = pin 1); ST3215_BUS says otherwise.");

void ServoBus::init()
{
  // Not this build's gripper: pins.h left the bus PIN_UNASSIGNED because the
  // four-bar owns pin 1 instead. Touching Serial1 here would take the pad back
  // off the servo library, so stand down entirely -- poll() does too.
  if (!PIN_IS_ASSIGNED(ST3215_BUS))
    return;

  Serial1.begin(kBaud, SERIAL_8N1_HALF_DUPLEX);

  // DOES THIS UART HAND BACK ITS OWN TRANSMISSION? Measure it; do not assume.
  //
  // In single-wire mode the core sets LPUART_CTRL_LOOPS | RSRC, which points
  // the receiver at the TX pad, and then only toggles TXDIR around a write
  // (cores/teensy4/HardwareSerial.cpp:576-578 and 697-702). It never clears RE,
  // so the receiver stays enabled and connected to a pin the transmitter is
  // driving -- which says the transmission comes straight back. That is a
  // reading of the core, not a datasheet guarantee, and getting it wrong is
  // expensive in both directions:
  //
  //   assume echo, get none   -> the first bytes of every REPLY are discarded
  //   assume none, get echo   -> the host sees its own request back as a reply
  //
  // Both look like a broken servo. So ask the hardware once, at boot, and let
  // poll() act on the answer.
  //
  // 0x00 IS SAFE TO SEND. Every STS packet begins 0xFF 0xFF, so a lone zero
  // byte cannot be parsed as one by anything on the bus, powered or not. And
  // the echo is a property of this UART, so this works with no servo attached.
  while (Serial1.available()) Serial1.read();
  Serial1.write((uint8_t)0x00);
  Serial1.flush();

  const uint32_t deadline = micros() + kEchoTimeoutUs;
  while (!Serial1.available() && (int32_t)(micros() - deadline) < 0) {}
  echoes_ = Serial1.available() > 0;
  while (Serial1.available()) Serial1.read();
}

void ServoBus::poll()
{
  if (!PIN_IS_ASSIGNED(ST3215_BUS))
    return;

  // NO HOST ON THE SECOND PORT: drain the bus and drop it.
  //
  // usb_serial write blocks when its buffer fills and nothing is draining it,
  // and this runs inside the loop that also holds the drill's watchdogs. A
  // bridge nobody is listening to must not be able to stall a motor timeout, so
  // the only thing it does in that state is keep the UART's buffer from filling
  // up with whatever the bus says.
  if (!SerialUSB1)
  {
    while (Serial1.available()) Serial1.read();
    return;
  }

  // HOST -> BUS. The host writes a whole STS packet in one go (StsBus::write
  // does a single write() of the framed packet), so a burst on SerialUSB1 is a
  // packet. Send it, then wait for the transmission to complete before looking
  // for an answer -- the servo cannot have replied before the last byte of the
  // request left the pin, which is what makes the echo unambiguous below.
  int n = SerialUSB1.available();
  if (n > 0)
  {
    uint8_t buf[kChunk];
    if (n > kChunk) n = kChunk;
    for (int i = 0; i < n; ++i) buf[i] = (uint8_t)SerialUSB1.read();

    Serial1.write(buf, n);
    Serial1.flush();          // returns on TC, which is where the core drops TXDIR
    to_bus_ += (uint32_t)n;

    if (echoes_) discard_echo(n);
  }

  // BUS -> HOST. Whatever is left is the servo talking.
  while (Serial1.available())
  {
    SerialUSB1.write((uint8_t)Serial1.read());
    ++to_host_;
  }
}

void ServoBus::discard_echo(int count)
{
  // EXACTLY as many bytes as were sent, never "everything waiting". A servo
  // with a short return delay can begin answering within a character time of
  // the request finishing, so draining the buffer wholesale would eat the front
  // of the reply. The count is the only thing that separates the two.
  const uint32_t deadline = micros() + kEchoTimeoutUs;
  while (count > 0 && (int32_t)(micros() - deadline) < 0)
  {
    if (Serial1.available())
    {
      Serial1.read();
      --count;
    }
  }
}

// Developed as a part of HSM Aries team,
// by Shivansh Mehta (https://github.com/Shivansh-Mehta),
// for the European Rover Challenge

#ifndef SCIENCE_H
#define SCIENCE_H

#include <Arduino.h>
#include <math.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include "Zanshin_BME680.h"

#include "SparkFun_SCD4x_Arduino_Library.h"

// THE ADC CONTRACT, AND BOTH HALVES HAVE TO AGREE WITH THE HARDWARE.
//
// 3.3 V is the Teensy 4.1's analog reference and is not adjustable on this
// part. 1023 is the 10-bit default of analogRead() -- the Teensy 4 can do 12
// bits, but only if analogReadResolution(12) is called, and nothing here calls
// it. If that is ever added, THIS CONSTANT MUST CHANGE WITH IT: leaving 1023
// against a 12-bit read scales every analog sensor by 4 and nothing anywhere
// reports it, because the readings stay in a believable range.
constexpr float V_REF = 3.3f;
constexpr float ADC_RES = 1023.0f;

// WHAT "NO READING" LOOKS LIKE ON THE WIRE.
//
// NaN, and never 0.0. Zero is a legitimate value for nearly everything this
// board measures -- 0 degrees C, 0 % moisture, 0 ppm TDS -- so a sensor that
// was never initialised, or whose read failed, would otherwise be
// indistinguishable from one reporting a real low reading. NaN shows up as
// `nan` in `ros2 topic echo`, makes every `if (x > threshold)` false, and
// cannot be averaged into a plausible wrong number.
//
// The delivered firmware zero-filled the telemetry array at boot and left it
// that way until each sensor was explicitly commanded, so an untouched pH index
// read 0.0 -- a strong acid -- rather than "nobody has asked yet".
constexpr float SCIENCE_NO_READING = NAN;

class AnalogSensor
{
protected:
  uint8_t m_pin;
  bool m_ready = false;
  float read_voltage() const;

public:
  AnalogSensor(uint8_t pin);
  virtual ~AnalogSensor() = default;

  // Returns false when the pin is PIN_UNASSIGNED, matching how the drill
  // board's Driver/LimitSwitch treat an un-numbered pin: inert rather than
  // configuring pin 255.
  bool init();
  bool is_ready() const { return m_ready; }

  virtual float get_value() = 0;
};

class DigitalSensor
{
protected:
  // Tracks whether init() actually succeeded. Reads made before or without a
  // successful init() return SCIENCE_NO_READING instead of silently touching
  // an uninitialised sensor object.
  bool m_ready = false;

public:
  virtual ~DigitalSensor() = default;
  virtual bool init() = 0;
  virtual void request_read() {}
  bool is_ready() const { return m_ready; }
};

class pHSensor : public AnalogSensor
{
private:
  float m_calibration_offset;

public:
  pHSensor(uint8_t pin, float offset = 0.0f);
  float get_value() override;
  void set_calibration_offset(float offset);
};

class CapacitiveMoistureSensor : public AnalogSensor
{
public:
  CapacitiveMoistureSensor(uint8_t pin);
  float get_value() override;
};

class TDSSensor : public AnalogSensor
{
public:
  TDSSensor(uint8_t pin);
  float get_value() override;
};

class ORPSensor : public AnalogSensor
{
private:
  float m_calibration_offset;

public:
  ORPSensor(uint8_t pin, float offset = 0.0f);
  float get_value() override;
  void set_calibration_offset(float offset);
};

class DS18B20Sensor : public DigitalSensor
{
private:
  uint8_t m_pin;
  OneWire m_oneWire;
  DallasTemperature m_sensor;

public:
  DS18B20Sensor(uint8_t pin);

  bool init() override;
  void request_read() override;
  float get_value();
};

class BME688Sensor : public DigitalSensor
{
private:
  BME680_Class m_sensor;

public:
  BME688Sensor();

  bool init() override;
  void request_read() override; // triggers a forced-mode conversion, non-blocking
  void get_data(float &temp, float &hum, float &press, float &gas);
};

// Talks over Wire1 (SDA 17, SCL 16 on a Teensy 4.1) -- a separate bus from the
// BME688 above, which is stuck on the default Wire because the Zanshin library
// has no way to target an alternate TwoWire.
class SCD41Sensor : public DigitalSensor
{
private:
  SCD4x m_sensor;

public:
  SCD41Sensor();

  bool init() override;

  // No request_read() override: SCD4x::readMeasurement() is already a
  // non-blocking check-and-fetch -- it polls the data-ready flag and returns
  // immediately when nothing is waiting -- so get_value() calls it directly
  // rather than needing a trigger/wait split.
  //
  // IT WILL OFTEN RETURN NO READING, AND THAT IS NORMAL. The part produces a
  // new sample roughly every 5 seconds in periodic mode, so a read command
  // landing between samples genuinely has nothing to hand back.
  float get_value();
};

#endif

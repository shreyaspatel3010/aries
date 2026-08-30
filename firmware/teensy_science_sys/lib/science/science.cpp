// Developed as a part of HSM Aries team,
// by Shivansh Mehta (https://github.com/Shivansh-Mehta),
// for the European Rover Challenge

#include "science.h"

#include "pins.h"

AnalogSensor::AnalogSensor(uint8_t pin) : m_pin(pin) {}

bool AnalogSensor::init()
{
  if (!PIN_IS_ASSIGNED(m_pin))
  {
    m_ready = false;
    return false;
  }

  pinMode(m_pin, INPUT);
  m_ready = true;
  return true;
}

float AnalogSensor::read_voltage() const
{
  return analogRead(m_pin) * (V_REF / ADC_RES);
}

// pH ==========================================================================
pHSensor::pHSensor(uint8_t pin, float offset)
    : AnalogSensor(pin), m_calibration_offset(offset) {}

float pHSensor::get_value()
{
  if (!m_ready)
    return SCIENCE_NO_READING;

  const float voltage = read_voltage();

  // Two-point line through the probe's two calibration buffers: pH 7.0 at
  // 1.5 V and pH 4.0 at 2.03 V. The slope is NEGATIVE (more voltage, more
  // acid), which is why the arithmetic below looks inverted.
  //
  // THESE TWO VOLTAGES ARE THE EMBEDDED TEAM'S and were not re-measured here.
  // They are a property of the individual probe and its board, and they drift
  // as the probe ages -- see protocols.md, which is the procedure for
  // re-deriving them.
  const float slope = ((7.0f - 4.0f) / (1.5f - 2.03f));
  const float raw_ph = 7.0f + (slope * (voltage - 1.5f));

  // The offset is APPLIED here. In an earlier version of the delivered code it
  // was stored, exposed through a setter, and then never added to the result --
  // so calibrating the probe changed nothing at all.
  return raw_ph + m_calibration_offset;
}

void pHSensor::set_calibration_offset(float offset)
{
  m_calibration_offset = offset;
}

// Capacitive soil moisture ====================================================
CapacitiveMoistureSensor::CapacitiveMoistureSensor(uint8_t pin)
    : AnalogSensor(pin) {}

float CapacitiveMoistureSensor::get_value()
{
  if (!m_ready)
    return SCIENCE_NO_READING;

  const float raw = static_cast<float>(analogRead(m_pin));

  // Raw counts, not volts, and the two endpoints are this specific probe's:
  // 759 counts in air (dry) and 403 in water (saturated). Counts fall as
  // moisture rises, hence the subtraction.
  //
  // CLAMPED TO 0..100 ON PURPOSE, and this is the one place on the board where
  // clamping is right rather than lazy: the output is a PERCENTAGE OF A RANGE
  // that was defined by those two endpoints, so a reading outside them is not
  // more information, it is a probe drier than air or wetter than water.
  const float moisture_pct = ((759.0f - raw) / (759.0f - 403.0f)) * 100.0f;

  if (moisture_pct < 0.0f)
    return 0.0f;
  if (moisture_pct > 100.0f)
    return 100.0f;
  return moisture_pct;
}

// TDS / EC ====================================================================
TDSSensor::TDSSensor(uint8_t pin) : AnalogSensor(pin) {}

float TDSSensor::get_value()
{
  if (!m_ready)
    return SCIENCE_NO_READING;

  const float voltage = read_voltage();

  // ppm per volt. STILL A PLACEHOLDER in the sense that matters: 194.107/0.713
  // is one measurement the embedded team took (194.107 ppm read at 0.713 V),
  // not a fit through several, and protocols.md describes it as the number to
  // replace. Treat the ppm figure as an order of magnitude until the probe has
  // been in a known calibration fluid.
  const float calibration_factor = (194.107f / 0.713f);

  const float tds_value = voltage * calibration_factor;

  // Clamp only the negative side, where amplifier offset can pull a genuinely
  // clean sample below zero. There is no upper endpoint to clamp against.
  return (tds_value < 0.0f) ? 0.0f : tds_value;
}

// ORP =========================================================================
ORPSensor::ORPSensor(uint8_t pin, float offset)
    : AnalogSensor(pin), m_calibration_offset(offset) {}

float ORPSensor::get_value()
{
  if (!m_ready)
    return SCIENCE_NO_READING;

  const float voltage = read_voltage();

  // SEN0165: the probe's millivolts are recovered from the divided, amplified
  // board output. 75x is the board's op-amp gain and 30 is its divider
  // constant; both are properties of the DFRobot board, not of this rover.
  const float orp_value = ((30.0f * V_REF * 1000.0f) - (75.0f * voltage * 1000.0f)) / 75.0f;

  // The offset here is large (531.6 mV in main.cpp) because it absorbs the
  // board's own zero, which is set with a physical potentiometer -- see
  // protocols.md. It is not a small trim.
  return orp_value + m_calibration_offset;
}

void ORPSensor::set_calibration_offset(float offset)
{
  m_calibration_offset = offset;
}

// DS18B20 soil temperature ====================================================
DS18B20Sensor::DS18B20Sensor(uint8_t pin)
    : m_pin(pin), m_oneWire(pin), m_sensor(&m_oneWire) {}

bool DS18B20Sensor::init()
{
  if (!PIN_IS_ASSIGNED(m_pin))
  {
    m_ready = false;
    return false;
  }

  pinMode(m_pin, INPUT_PULLUP);
  m_sensor.begin();

  // THE LINE THAT KEEPS THIS BOARD RESPONSIVE. Without it DallasTemperature
  // blocks inside requestTemperatures() for the full 750 ms conversion, which
  // would be 750 ms of a micro-ROS executor that is not spinning and an agent
  // ping that is not being answered.
  m_sensor.setWaitForConversion(false);

  // 12-bit explicitly, because the 750 ms the caller waits out is the 12-bit
  // conversion time. A probe left at a lower resolution converts faster and
  // reads coarser, and the wait would then be silently generous rather than
  // matched.
  m_sensor.setResolution(12);

  // Is anything actually on the bus? This is the only sensor here that can
  // answer that question cheaply.
  m_ready = (m_sensor.getDeviceCount() != 0);
  return m_ready;
}

void DS18B20Sensor::request_read()
{
  if (!m_ready)
    return; // do not trigger a conversion on a probe that was never found

  m_sensor.requestTemperatures();
}

float DS18B20Sensor::get_value()
{
  if (!m_ready)
    return SCIENCE_NO_READING;

  const float temp = m_sensor.getTempCByIndex(0);

  // DEVICE_DISCONNECTED_C. The library returns -127.0 for a probe it cannot
  // read, which is a sentinel, not a temperature -- passing it through would
  // put a plausible-looking (if absurd) number on the telemetry array. The
  // caller retries once before this is reached; see main.cpp.
  if (temp <= -127.0f)
    return SCIENCE_NO_READING;

  return temp;
}

// BME688 ======================================================================
BME688Sensor::BME688Sensor() {}

bool BME688Sensor::init()
{
  m_ready = m_sensor.begin(I2C_STANDARD_MODE);
  if (!m_ready)
    return false;

  m_sensor.setOversampling(TemperatureSensor, Oversample16);
  m_sensor.setOversampling(HumiditySensor, Oversample16);
  m_sensor.setOversampling(PressureSensor, Oversample16);
  m_sensor.setIIRFilter(IIR4);
  m_sensor.setGas(320, 150); // heater plate 320 C for 150 ms

  return true;
}

void BME688Sensor::request_read()
{
  if (!m_ready)
    return;

  // Non-blocking: starts a forced-mode conversion and returns. The result is
  // collected in get_data() once the caller's tick counter has waited out the
  // conversion and the gas heater's soak.
  m_sensor.triggerMeasurement();
}

void BME688Sensor::get_data(float &temp, float &hum, float &press, float &gas)
{
  if (!m_ready)
  {
    temp = hum = press = gas = SCIENCE_NO_READING;
    return;
  }

  int32_t raw_temp, raw_hum, raw_press, raw_gas;

  // waitSwitch = false: the conversion time has already been waited out by the
  // caller's tick counter, so this grabs the completed reading instead of
  // blocking the executor here.
  m_sensor.getSensorData(raw_temp, raw_hum, raw_press, raw_gas, false);

  temp = raw_temp / 100.0f;   // deg C
  hum = raw_hum / 1000.0f;    // % RH
  press = raw_press / 100.0f; // hPa
  gas = raw_gas / 100.0f;     // Ohms
}

// SCD41 CO2 ===================================================================
SCD41Sensor::SCD41Sensor() {}

bool SCD41Sensor::init()
{
  // begin() calls stopPeriodicMeasurement() internally -- 500 ms of blocking,
  // acceptable here because init() is a one-time bring-up on an explicit
  // command and never on the read path -- before verifying the part is there
  // via getSerialNumber(). That stop is what makes this survive a board left
  // running periodic measurement by a previous session, which otherwise refuses
  // every configuration command.
  m_ready = m_sensor.begin(Wire1);
  if (!m_ready)
    return false;

  m_ready = m_sensor.startPeriodicMeasurement();
  return m_ready;
}

float SCD41Sensor::get_value()
{
  if (!m_ready)
    return SCIENCE_NO_READING;

  // Non-blocking check-and-fetch: polls the data-ready flag and only pulls a
  // sample if one is waiting. In periodic mode the part produces one roughly
  // every 5 seconds, so a read command landing between samples legitimately has
  // nothing to return -- which is a no-reading, not a zero.
  if (!m_sensor.readMeasurement())
    return SCIENCE_NO_READING;

  return static_cast<float>(m_sensor.getCO2());
}

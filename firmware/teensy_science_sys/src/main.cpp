// Developed as a part of HSM Aries team,
// by Shivansh Mehta (https://github.com/Shivansh-Mehta),
// for the European Rover Challenge
//
// Teensy 4.1 SCIENCE board -- micro-ROS over USB Serial.
//
// THE SECOND BOARD. firmware/teensy_drill_sys is a different Teensy on a
// different USB port with its own micro-ROS agent; this one carries only
// sensors and drives nothing. The two cannot be merged without renumbering,
// because four of this board's pins are something else entirely over there --
// see the note at the top of pins.h.
//
// TOPICS
//
//   TOPIC                   TYPE                MEANING
//   /science/telemetry      Float32MultiArray   10 values, 1 Hz, RELIABLE
//   /science/sensor_cmd     UInt8               sensor*10 + action, RELIABLE
//
// THE TELEMETRY ARRAY IS INDEX-ORDERED AND THE ORDER IS THE WIRE FORMAT. It
// carries no names, so nothing anywhere can catch two entries being swapped:
// exchange pH and moisture and both numbers stay entirely plausible. The order
// is fixed by TelemetryIndex below and by `fields` in
// aries_science/config/science.yaml, and the two have to agree.
//
// PULL, NOT STREAM. Nothing is sampled unless it is asked for. The array is
// published every second regardless, so a value that was read once keeps being
// republished until it is read again -- the topic is a LATEST-KNOWN board, not
// a live feed. That is the embedded team's design (see protocols.md) and it is
// deliberate: several of these sensors are slow, and two of them are consumed
// by being read.
//
// So an index means one of three things, and they are distinguishable:
//   a number  the last value this sensor was commanded to produce
//   NaN       never read, or the read failed / the sensor is not there
//   (age)     not carried on the wire -- if it matters, read it again
//
// Commands are `sensor_id * 10 + action`, action 1 = init, 2 = read:
//
//   01/02 pH        11/12 moisture   21/22 TDS-EC   31/32 ORP
//   41/42 DS18B20   51/52 BME688 (fills 5,6,7,8)    91/92 SCD41 CO2
//
// A SENSOR MUST BE INIT'd BEFORE IT WILL READ. `02` before `01` returns NaN
// rather than a number, on purpose -- an uninitialised analog pin reads
// something, and that something is not a measurement.
//
// The agent is started by aries_hardware.launch.py alongside the drill's. By
// hand:
//   ros2 run micro_ros_agent micro_ros_agent serial --dev <port> -b 115200
//
// NOTHING MAY WRITE TO Serial. It is the micro-ROS transport. The LED on pin 13
// is the only status channel this board has -- see led_update().

#include <Arduino.h>

#include <micro_ros_platformio.h>
#include <rcl/error_handling.h>
#include <rcl/rcl.h>
#include <rclc/executor.h>
#include <rclc/rclc.h>

#include <std_msgs/msg/float32_multi_array.h>
#include <std_msgs/msg/u_int8.h>

#include "pins.h"
#include "science.h"

// --- Tunables ---------------------------------------------------------------

// THE ROS DOMAIN. MUST MATCH network.domain_id IN
// src/aries_common/config/devices.yaml.
//
// A micro-ROS client defaults to domain 0. Nothing in the transport or the
// handshake carries the domain, so unless it is set here the board joins domain
// 0 while the entire rover runs on 30 -- and the failure is completely silent:
//
//   * the agent connects, and its log shows every entity created correctly
//   * the board's own node never appears in `ros2 node list`
//   * /science/telemetry has `Publisher count: 0` and never delivers a sample
//
// The delivered firmware called rclc_support_init(&support, 0, NULL, &allocator),
// which takes the default domain. That is the same bug the drill board shipped
// with, and it cost most of 2026-08-26 to find there -- the board was findable
// the whole time, on `ROS_DOMAIN_ID=0 ros2 node list`.
//
// It has to be compiled in: the board has no config file and no way to learn
// the domain at runtime.
static const size_t ROS_DOMAIN = 30;

// Telemetry period. 1 Hz, matching the delivered firmware and protocols.md.
//
// Faster would not make the data fresher. Nothing is sampled between commands,
// so a higher rate only republishes the same values under a new timestamp,
// which reads as a live sensor and is not one.
static const uint32_t TELEMETRY_PERIOD_MS = 1000;

// The state-machine tick. Everything slow on this board is timed in whole
// ticks of it, so it divides TELEMETRY_PERIOD_MS exactly.
static const uint32_t TICK_PERIOD_MS = 100;

// rcl's publish and teardown functions are declared warn_unused_result, and a
// (void) cast does NOT silence that in GCC -- only consuming the value does.
#define IGNORE_RC(expr)               \
  do                                  \
  {                                   \
    rcl_ret_t ignored_rc_ = (expr);   \
    (void)ignored_rc_;                \
  } while (0)

// --- Sensors ----------------------------------------------------------------

pHSensor ph_sensor(PIN_PH);
CapacitiveMoistureSensor moisture_sensor(PIN_MOISTURE);
TDSSensor tds_sensor(PIN_TDS);

// 531.612976 mV of offset, which is not a trim -- it absorbs the SEN0165
// board's own zero, set with a physical potentiometer. The embedded team's
// number; see protocols.md for how it was arrived at and how to redo it.
ORPSensor orp_sensor(PIN_ORP, 531.612976f);

DS18B20Sensor soil_temp_sensor(PIN_TEMP_SOIL);
BME688Sensor bme_sensor;

// No address argument: the SparkFun library owns it, and the SCD4x is fixed at
// 0x62. It talks over Wire1 -- see science.h.
SCD41Sensor scd_sensor;

// --- Telemetry array --------------------------------------------------------
//
// THE INDEX ORDER IS THE WIRE FORMAT. See the header comment.
enum TelemetryIndex
{
  IDX_PH = 0,            // 0..14, 7 is neutral
  IDX_SOIL_MOISTURE = 1, // %
  IDX_TDS = 2,           // ppm
  IDX_ORP = 3,           // mV
  IDX_SOIL_TEMP = 4,     // deg C, DS18B20 probe
  IDX_BME_TEMP = 5,      // deg C, ambient air
  IDX_BME_HUM = 6,       // % RH
  IDX_BME_PRESS = 7,     // hPa
  IDX_BME_GAS = 8,       // Ohms, VOC resistance; higher is cleaner
  IDX_SCD_CO2 = 9,       // ppm, fresh air is ~400
  TELEMETRY_SIZE = 10,
};

// THE MESSAGE BODY, AND IT HAS TO BE HANDED TO THE MESSAGE BY HAND.
//
// Float32MultiArray has a DYNAMIC ARRAY in it and micro-ROS does not allocate
// one. The message is a zero-filled global, so data.data is NULL and
// data.capacity is 0 until something assigns them -- and rcl_publish on that
// does not fail loudly, it serialises a zero-length array. The host then
// reports an empty telemetry message forever while both ends look healthy.
static float telemetry_buffer[TELEMETRY_SIZE];

// --- micro-ROS entities -----------------------------------------------------

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

rcl_publisher_t telemetry_pub;
std_msgs__msg__Float32MultiArray telemetry_msg;

rcl_subscription_t sensor_cmd_sub;
std_msgs__msg__UInt8 sensor_cmd_msg;

enum MicroROSState
{
  WAITING_AGENT,
  AGENT_AVAILABLE,
  AGENT_CONNECTED,
  AGENT_DISCONNECTED,
};
static MicroROSState uros_state = WAITING_AGENT;

// How far create_entities() got last time.
//
// destroy_entities() runs on the failure path too, and finalising a handle that
// was never initialised is not a no-op: support/node/sub are zero-filled
// globals, so rcl_context_get_rmw_context(&support.context) dereferences a NULL
// impl pointer and hard-faults the MCU. A faulted board stops pinging and stops
// publishing while USB stays enumerated, so the agent sits there with the port
// open and the board never rejoins -- indistinguishable from an unflashed
// Teensy, and only a physical reset clears it.
static uint8_t entities_stage = 0;

// --- Deferred reads ---------------------------------------------------------
//
// Two sensors cannot answer immediately, and neither may be waited for inline:
// this loop is also the agent's ping responder, and a board that stops
// answering pings is a board the agent tears the session down on.
//
// So a read command ARMS a countdown and returns. The tick handler collects the
// value when it expires. Both counters are in ticks of TICK_PERIOD_MS.
static uint8_t temp_wait_ticks = 0;
static uint8_t temp_retry_count = 0;
static uint8_t bme_wait_ticks = 0;

// 9 ticks = 900 ms, comfortably past the DS18B20's 750 ms 12-bit conversion.
static const uint8_t TEMP_WAIT_TICKS = 9;

// 4 ticks = 400 ms: 16x oversampling on three channels plus the gas heater's
// 150 ms soak.
static const uint8_t BME_WAIT_TICKS = 4;

// One automatic retry on a -127. A lone disconnect sentinel is very often a
// transient OneWire glitch -- a missed conversion or a bad scratchpad CRC --
// rather than a probe that has actually come off. Deliberately NOT retried on
// 0.0, which is a temperature the probe can legitimately report.
static const uint8_t TEMP_MAX_RETRIES = 1;

static bool create_entities();
static void destroy_entities();

// --- Status LED -------------------------------------------------------------
//
// The only channel this board has: Serial belongs to micro-ROS.
//
//   slow blink (500 ms)   waiting for the agent
//   solid on              connected
static void led_update()
{
  static uint32_t last_ms = 0;
  const uint32_t now = millis();

  if (uros_state == AGENT_CONNECTED)
  {
    digitalWrite(LED_BUILTIN, HIGH);
    return;
  }

  if (now - last_ms >= 500)
  {
    last_ms = now;
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  }
}

// --- Commands ---------------------------------------------------------------

static void handle_init(uint8_t sensor_id)
{
  switch (sensor_id)
  {
  case IDX_PH:
    ph_sensor.init();
    break;
  case IDX_SOIL_MOISTURE:
    moisture_sensor.init();
    break;
  case IDX_TDS:
    tds_sensor.init();
    break;
  case IDX_ORP:
    orp_sensor.init();
    break;
  case IDX_SOIL_TEMP:
    soil_temp_sensor.init();
    break;
  case IDX_BME_TEMP:
    bme_sensor.init();
    break;
  case IDX_SCD_CO2:
    scd_sensor.init();
    break;
  default:
    // An unrecognised id does nothing. There is no sensible guess to make, and
    // a mistyped command that silently re-initialised the wrong sensor would
    // reset a calibration nobody was looking at.
    break;
  }
}

static void handle_read(uint8_t sensor_id)
{
  switch (sensor_id)
  {
  case IDX_PH:
    telemetry_buffer[IDX_PH] = ph_sensor.get_value();
    break;
  case IDX_SOIL_MOISTURE:
    telemetry_buffer[IDX_SOIL_MOISTURE] = moisture_sensor.get_value();
    break;
  case IDX_TDS:
    telemetry_buffer[IDX_TDS] = tds_sensor.get_value();
    break;
  case IDX_ORP:
    telemetry_buffer[IDX_ORP] = orp_sensor.get_value();
    break;

  case IDX_SOIL_TEMP:
    // Deferred: arm the countdown and return. See the note on temp_wait_ticks.
    soil_temp_sensor.request_read();
    temp_wait_ticks = TEMP_WAIT_TICKS;
    temp_retry_count = 0; // fresh retry budget on every operator-requested read
    break;

  case IDX_BME_TEMP:
    // Deferred, and fills FOUR indices when it lands: 5, 6, 7 and 8.
    bme_sensor.request_read();
    bme_wait_ticks = BME_WAIT_TICKS;
    break;

  case IDX_SCD_CO2:
    // Not deferred, but it may legitimately have nothing to give: the part
    // produces a sample about every 5 seconds and get_value() returns
    // SCIENCE_NO_READING between them.
    telemetry_buffer[IDX_SCD_CO2] = scd_sensor.get_value();
    break;

  default:
    break;
  }
}

static void sensor_cmd_callback(const void *msgin)
{
  const std_msgs__msg__UInt8 *msg = (const std_msgs__msg__UInt8 *)msgin;

  const uint8_t sensor_id = msg->data / 10;
  const uint8_t action = msg->data % 10;

  if (action == 1)
    handle_init(sensor_id);
  else if (action == 2)
    handle_read(sensor_id);
  // Anything else is ignored rather than guessed at. protocols.md defines two
  // actions and there is no third reading of this byte.
}

// --- Tick -------------------------------------------------------------------

static void timer_callback(rcl_timer_t *timer_obj, int64_t last_call_time)
{
  (void)last_call_time;
  if (timer_obj == NULL)
    return;

  // Collect the DS18B20 once its conversion has had time to finish.
  if (temp_wait_ticks > 0)
  {
    temp_wait_ticks--;
    if (temp_wait_ticks == 0)
    {
      const float temp = soil_temp_sensor.get_value();

      // get_value() maps the library's -127 disconnect sentinel to NaN, so a
      // failed read is a NaN here rather than a plausible-looking number.
      if (isnan(temp) && temp_retry_count < TEMP_MAX_RETRIES)
      {
        temp_retry_count++;
        soil_temp_sensor.request_read();
        temp_wait_ticks = TEMP_WAIT_TICKS;
      }
      else
      {
        telemetry_buffer[IDX_SOIL_TEMP] = temp;
      }
    }
  }

  // Collect the BME688's forced-mode conversion. One command, four indices.
  if (bme_wait_ticks > 0)
  {
    bme_wait_ticks--;
    if (bme_wait_ticks == 0)
    {
      bme_sensor.get_data(
          telemetry_buffer[IDX_BME_TEMP],
          telemetry_buffer[IDX_BME_HUM],
          telemetry_buffer[IDX_BME_PRESS],
          telemetry_buffer[IDX_BME_GAS]);
    }
  }

  // Publish on the whole-second boundary. The counter is in ticks so that the
  // two countdowns above and this share one clock.
  static uint8_t publish_tick = 0;
  const uint8_t ticks_per_publish = (uint8_t)(TELEMETRY_PERIOD_MS / TICK_PERIOD_MS);

  if (++publish_tick >= ticks_per_publish)
  {
    publish_tick = 0;
    IGNORE_RC(rcl_publish(&telemetry_pub, &telemetry_msg, NULL));
  }
}

// --- micro-ROS lifecycle ----------------------------------------------------

static bool create_entities()
{
  allocator = rcl_get_default_allocator();
  entities_stage = 0;

  // Domain must be set through init options -- rclc_support_init() takes the
  // default, which is 0. See ROS_DOMAIN above for what that costs.
  //
  // EVERY PATH OUT OF HERE MUST fini THE OPTIONS. rclc_support_init_with_options
  // copies them into the context, so this copy is ours to release. micro-ROS
  // allocates from a fixed static pool and create_entities() is retried on
  // every reconnect, so leaking one set per attempt is not a slow leak -- it is
  // a board that works on the first connection after a flash and then never
  // again.
  rcl_init_options_t init_options = rcl_get_zero_initialized_init_options();
  if (RCL_RET_OK != rcl_init_options_init(&init_options, allocator))
    return false;

  if (RCL_RET_OK != rcl_init_options_set_domain_id(&init_options, ROS_DOMAIN))
  {
    IGNORE_RC(rcl_init_options_fini(&init_options));
    return false;
  }

  const rcl_ret_t support_rc =
      rclc_support_init_with_options(&support, 0, NULL, &init_options, &allocator);
  IGNORE_RC(rcl_init_options_fini(&init_options));
  if (RCL_RET_OK != support_rc)
    return false;
  entities_stage = 1;

  if (RCL_RET_OK != rclc_node_init_default(&node, "science_module_node", "", &support))
    return false;
  entities_stage = 2;

  // RELIABLE, AND THAT IS NOT A PREFERENCE. 1 Hz argues for best effort, but
  // the host subscribes with rclpy's default QoS, which is RELIABLE, and a
  // BEST_EFFORT publisher against a RELIABLE subscriber is an incompatible pair
  // that DDS makes NO match for at all -- both sides list the topic,
  // `ros2 topic info` shows a publisher and a subscriber, and not one message
  // is ever delivered. Change this only together with the QoS in
  // aries_science's subscription.
  if (RCL_RET_OK != rclc_publisher_init_default(
                        &telemetry_pub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
                        "/science/telemetry"))
    return false;
  entities_stage = 3;

  // RELIABLE, and this is the strongest case for it on either board: every
  // command is a one-shot that nothing re-sends, and a dropped read leaves the
  // operator looking at the previous value believing it is the new one.
  if (RCL_RET_OK != rclc_subscription_init_default(
                        &sensor_cmd_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8),
                        "/science/sensor_cmd"))
    return false;
  entities_stage = 4;

  if (RCL_RET_OK != rclc_timer_init_default(
                        &timer, &support, RCL_MS_TO_NS(TICK_PERIOD_MS), timer_callback))
    return false;
  entities_stage = 5;

  // TWO HANDLES: one timer, one subscription. rclc_executor_add_* past the end
  // of this array returns an error nothing here checks, and the extra handle is
  // then simply never spun.
  if (RCL_RET_OK != rclc_executor_init(&executor, &support.context, 2, &allocator))
    return false;
  entities_stage = 6;

  rclc_executor_add_timer(&executor, &timer);
  rclc_executor_add_subscription(&executor, &sensor_cmd_sub, &sensor_cmd_msg,
                                 &sensor_cmd_callback, ON_NEW_DATA);

  return true;
}

static void destroy_entities()
{
  if (entities_stage == 0)
    return;

  rmw_context_t *rmw_ctx = rcl_context_get_rmw_context(&support.context);
  (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_ctx, 0);

  // Reverse creation order.
  if (entities_stage >= 6)
    rclc_executor_fini(&executor);
  if (entities_stage >= 5)
    IGNORE_RC(rcl_timer_fini(&timer));
  if (entities_stage >= 4)
    IGNORE_RC(rcl_subscription_fini(&sensor_cmd_sub, &node));
  if (entities_stage >= 3)
    IGNORE_RC(rcl_publisher_fini(&telemetry_pub, &node));
  if (entities_stage >= 2)
    IGNORE_RC(rcl_node_fini(&node));
  rclc_support_fini(&support);

  entities_stage = 0;
}

// --- Arduino entry points ---------------------------------------------------

void setup()
{
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // BME688 is on the default Wire (18/19) because the Zanshin library is
  // hardcoded to it; the SCD41 is on Wire1 (17/16). Wire2 is deliberately NOT
  // started -- the delivered firmware called Wire2.begin() with the comment
  // "reserved / unused for now", which configures a peripheral and claims pins
  // 25 and 24 for nothing.
  Wire.begin();
  Wire1.begin();

  // NaN, NOT ZERO, and this is the whole reason the array starts here rather
  // than at its default. Zero is a legitimate reading for nearly everything on
  // this board -- 0 degrees C, 0 % moisture, 0 ppm TDS -- so a zero-filled boot
  // state is indistinguishable from a set of real low readings. The delivered
  // firmware filled it with 0.0f and left it there until each sensor was first
  // commanded, so an untouched pH index read 0.0: a strong acid.
  for (int i = 0; i < TELEMETRY_SIZE; ++i)
    telemetry_buffer[i] = SCIENCE_NO_READING;

  // Point the message at its buffer ONCE, here, rather than inside
  // create_entities(). Both are globals that outlive every agent session, and
  // create_entities() runs again on every reconnect -- doing it here means a
  // reconnect cannot leave the message pointing at nothing. See the note on
  // telemetry_buffer for what an unassigned sequence actually does.
  telemetry_msg.data.data = telemetry_buffer;
  telemetry_msg.data.size = TELEMETRY_SIZE;
  telemetry_msg.data.capacity = TELEMETRY_SIZE;

  // No dimensions. The layout block is optional in Float32MultiArray and the
  // host reads msg.data alone, so an empty dim sequence keeps three rosidl
  // strings off a board that would have to allocate them. It still has to be
  // explicitly empty rather than merely zeroed by accident, because
  // serialisation walks it.
  telemetry_msg.layout.dim.data = NULL;
  telemetry_msg.layout.dim.size = 0;
  telemetry_msg.layout.dim.capacity = 0;
  telemetry_msg.layout.data_offset = 0;

  // SENSORS ARE NOT INITIALISED HERE. They wait for an action-1 command over
  // ROS, which is the embedded team's design and is kept: several of these
  // parts are slow to bring up (the SCD41 blocks 500 ms inside begin()), and
  // doing all seven at boot would sit in setup() with the agent waiting.
  //
  // The consequence is worth stating plainly: STRAIGHT AFTER A RESET EVERY
  // INDEX IS NaN and stays that way until the operator sends the init and read
  // commands. That is not a fault.

  set_microros_serial_transports(Serial);
}

void loop()
{
  const uint32_t now = millis();
  static uint32_t last_ping_ms = 0;

  led_update();

  switch (uros_state)
  {
  case WAITING_AGENT:
    if (now - last_ping_ms > 500)
    {
      last_ping_ms = now;
      uros_state = (RMW_RET_OK == rmw_uros_ping_agent(200, 1)) ? AGENT_AVAILABLE : WAITING_AGENT;
    }
    break;

  case AGENT_AVAILABLE:
    uros_state = create_entities() ? AGENT_CONNECTED : WAITING_AGENT;
    if (uros_state == WAITING_AGENT)
      destroy_entities();
    break;

  case AGENT_CONNECTED:
    // Ping every 500 ms, allowing 3 retries x 200 ms before declaring a
    // disconnect, which tolerates brief serial latency without tearing the
    // session down.
    if (now - last_ping_ms > 500)
    {
      last_ping_ms = now;
      uros_state =
          (RMW_RET_OK == rmw_uros_ping_agent(200, 3)) ? AGENT_CONNECTED : AGENT_DISCONNECTED;
    }
    if (uros_state != AGENT_CONNECTED)
      break;

    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));
    break;

  case AGENT_DISCONNECTED:
    // NOTHING TO MAKE SAFE. This board drives no actuator -- unlike the drill
    // board, which stops every motor here -- so the teardown is the whole job.
    // Any deferred read in flight is abandoned; its countdown simply expires
    // against a timer that no longer exists.
    temp_wait_ticks = 0;
    bme_wait_ticks = 0;
    destroy_entities();
    // Re-initialise the transport so the agent can reopen the port immediately
    // instead of waiting for USB re-enumeration.
    set_microros_serial_transports(Serial);
    uros_state = WAITING_AGENT;
    break;
  }
}

// Developed as a part of HSM Aries team,
// by Shivansh Mehta (https://github.com/Shivansh-Mehta),
// for the European Rover Challenge
//
// Teensy 4.1 drill / science board -- micro-ROS over USB Serial.
//
// ONE BOARD. This firmware replaces firmware/legacy/teensy_gripper/*.ino
// entirely: it drives the drill AND the gripper servo AND the stack light off
// the same Teensy, because that is how the harness is wired. pin-def-ref.txt
// hands the auger the three pins the old sketch used for the stack light and moves
// the light to 37/36/35, which only makes sense as a rewire of the one board.
//
// TOPICS -- the three shared with the rest of the workspace use the workspace's
// names and its message semantics, NOT the names this firmware was written
// with. See the note on each below; the stack light one is a safety issue, not
// a naming preference.
//
//   TOPIC                      TYPE     MEANING                        C++ NAME
//   /gripper/cmd               Float32  jaws, 0..1        BEST EFFORT  gservo
//   /gripper/state             Float32  jaws echoed, 100 Hz, BEST EFF  gservo
//   stacklight_subscription    UInt8    1=red 2=yellow 3=green 4=off   stalig
//   motor1/cmd_speed           Int32    -255..255  AUGER, spins        auger
//   motor2/cmd_speed           Int32    -255..255  FEED, drill up/down feed_motor
//   linact/state               UInt8    1=extend 2=retract 3/4=home    bin_actuator
//   linact/cext                Float32  signed mm, sample bin fore/aft bin_actuator
//   sand_box/lid/cmd           Float32  0=closed 1=open                lid_servo
//
// THE TOPIC NAMES AND THE C++ NAMES DELIBERATELY DIFFER for the drill's three
// axes. The topics are the wire contract with aries_bringup's drill_driver
// (config/drill_driver.yaml names them, test_drill_driver.py pins them), so
// they are frozen. The C++ names say what each one MOVES, because motor1 /
// motor2 / linact do not, and getting them the wrong way round means driving
// the mast instead of the spindle:
//
//   auger         spins the cutting head. Does not travel.
//   feed_motor    moves the WHOLE DRILL up and down -- the vertical axis.
//   bin_actuator  slides the sample bin fore/aft. Horizontal.
//
// The URDF's names are worse still: drill_motor_joint is the VERTICAL FEED and
// drill_bit_joint is the auger's rotation. Those are load-bearing across the
// URDF, the gz bridge and the joystick, so they stay as they are.
//
// NOTHING ON THE HOST DRIVES THE LID YET -- deliberately. There is no joystick
// binding and no node; it is reachable with `ros2 topic pub` and from a mission
// script, and the operator surface comes later.
//
// The agent is started by aries_hardware.launch.py. By hand:
//   ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200
//
// Do not raise that baud. Linux speed_t values are encodings, not bit rates,
// and the largest valid one is B4000000. Anything above it is rejected by
// cfsetospeed with EINVAL, which the agent does not check, leaving the port
// speed unset. This link is USB CDC, so the device ignores baud anyway.
//
// NOTHING MAY WRITE TO Serial. It is the micro-ROS transport. The LED is the
// only status channel this board has -- see led_update().

#include <Arduino.h>
#include <micro_ros_platformio.h>

#include <rcl/error_handling.h>
#include <rcl/rcl.h>
#include <rclc/executor.h>
#include <rclc/rclc.h>

#include <std_msgs/msg/float32.h>
#include <std_msgs/msg/int32.h>
#include <std_msgs/msg/u_int8.h>

#include "drill.h"
#include "emg.h"
#include "pins.h"

// --- Tunables ---------------------------------------------------------------

// Stop the drill motors if no command arrives for this long.
//
// drill_joystick.py publishes at 30 Hz only while LT is held, then a burst of
// zeros for stop_hold_sec, then goes silent -- so this watchdog fires on every
// idle period, and correctly does nothing, because the last thing said was
// zero. What it is actually for is the case zeros never arrive: the host node
// dies, or the link drops, while the auger is turning. Without it a spinning
// auger has no way to learn that nobody is driving it any more.
//
// It cannot be defeated by a host that re-sends its last command on a timer,
// because drill_joystick.py stops publishing entirely rather than repeating.
static const uint32_t MOTOR_COMMAND_TIMEOUT_MS = 500;

// /gripper/state publish period. 100 Hz matches the ros2_control loop the
// gripper hardware interface runs at.
static const uint32_t GRIPPER_STATE_PERIOD_MS = 10;

// THE ROS DOMAIN. MUST MATCH network.domain_id IN
// src/aries_common/config/devices.yaml.
//
// A micro-ROS client defaults to domain 0. Nothing in the transport or the
// handshake carries the domain, so unless it is set here the board joins domain
// 0 while the entire rover runs on 30 -- and the failure is completely silent:
//
//   * the agent connects, and its log shows every entity created correctly
//   * `ros2 topic list` on the rover shows /gripper/state, because the HOST
//     side advertises it too (the hardware interface subscribes, drill_driver
//     publishes the motor topics) -- so the names all look present
//   * but `Publisher count: 0`, and the board's own node never appears
//   * TeensyGripperSystem then reports "Never received /gripper/state" forever
//     and silently swallows every gripper command
//
// That is exactly what happened on 2026-08-26. The board was findable the whole
// time -- on `ROS_DOMAIN_ID=0 ros2 node list`.
//
// It has to be compiled in: the board has no config file and no way to learn
// the domain at runtime. test_firmware_domain.py pins this against devices.yaml
// so the two cannot drift apart unnoticed.
static const size_t ROS_DOMAIN = 30;

// rcl's teardown and publish functions are declared warn_unused_result, and a
// (void) cast does NOT silence that in GCC -- only actually consuming the value
// does. Every use below is a path where the return is genuinely not actionable:
// the session is already being torn down, or the sample is one of a hundred a
// second on a best-effort topic. Naming that here beats nine identical warnings
// scrolling past on every build and hiding a real one.
#define IGNORE_RC(expr)               \
  do                                  \
  {                                   \
    rcl_ret_t ignored_rc_ = (expr);   \
    (void)ignored_rc_;                \
  } while (0)

// --- micro-ROS entities -----------------------------------------------------

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;

rcl_subscription_t auger_cmd_sub;
std_msgs__msg__Int32 auger_cmd_msg;
AugerMotor auger(AUGER_PWM, AUGER_INA, AUGER_INB);

rcl_subscription_t feed_cmd_sub;
std_msgs__msg__Int32 feed_cmd_msg;
LeadScrewMotor feed_motor(FEED_PWM, FEED_INA, FEED_INB);

rcl_subscription_t bin_state_cmd_sub;
std_msgs__msg__UInt8 bin_state_cmd_msg;
rcl_subscription_t bin_cext_cmd_sub;
std_msgs__msg__Float32 bin_cext_cmd_msg;
LinearActuator bin_actuator(BIN_PWM, BIN_INA, BIN_INB);

rcl_subscription_t stalig_cmd_sub;
std_msgs__msg__UInt8 stalig_state_cmd_msg;
StackLight stalig(STALIG_G, STALIG_Y, STALIG_R);

rcl_subscription_t gservo_cmd_sub;
std_msgs__msg__Float32 gservo_cmd_msg;
rcl_publisher_t gservo_state_pub;
std_msgs__msg__Float32 gservo_state_msg;
SlewServo gservo(GRIPPER_SERVO);

// Lid of the front-left deck container (the sand box). Same class, same 0..1
// range and same slew rate as the gripper -- 0.0 is CLOSED, and it is also
// where SlewServo starts, so the lid is commanded shut within microseconds of
// reset rather than sitting wherever it was left.
//
// That boot write is a JUMP, not a slew: init() writes the position directly,
// before the slew clock starts. A lid left open therefore snaps shut on power
// up. Correct for a sample container, but it is a real movement at power-up
// and worth knowing about with fingers near the hinge.
rcl_subscription_t lid_cmd_sub;
std_msgs__msg__Float32 lid_cmd_msg;
SlewServo lid_servo(LID_SERVO_SAND_BOX);

// Both switches are on the FEED CARRIAGE, one at each end of its travel. The
// auger is a spindle and has no end of travel; the delivered firmware stopped
// it on switch 1 anyway. See pins.h.
LimitSwitch switch_feed_bottom(LIMIT_SWITCH1, 50);
LimitSwitch switch_feed_top(LIMIT_SWITCH2, 50);

// --- Command state ----------------------------------------------------------

// Last commanded feed speed, signed. POSITIVE IS UP, matching the host: the
// carriage joint runs -0.375 (bottom) .. 0.185 (top) in drill.xacro, and
// joystick.yaml documents D-pad UP as raising it. Flip it in joystick.yaml's
// invert_motor, not here.
static int feed_cmd_signed = 0;
static int feed_applied = 0;

static int auger_cmd_signed = 0;
static int auger_applied = 0;

static uint32_t last_motor_cmd_ms = 0;

// --- micro-ROS lifecycle ----------------------------------------------------

enum MicroROSState
{
  WAITING_AGENT,
  AGENT_AVAILABLE,
  AGENT_CONNECTED,
  AGENT_DISCONNECTED
};
static MicroROSState uros_state = WAITING_AGENT;

// How far create_entities() got last time.
//
// destroy_entities() runs on the failure path too, and finalising a handle that
// was never initialised is not a no-op: support/node/subs are zero-filled
// globals, so rcl_context_get_rmw_context(&support.context) dereferences a NULL
// impl pointer and hard-faults the MCU. A faulted board stops pinging and stops
// publishing while USB stays enumerated, so the agent sits there with the port
// open and the board never rejoins -- indistinguishable from an unflashed
// Teensy, and only a physical reset clears it. Tear down exactly what exists.
static uint8_t entities_stage = 0;

static void stop_all_motors();
static void apply_motor_commands(bool force);

void auger_cmd_callback(const void *msin);
void feed_cmd_callback(const void *msin);
void bin_state_cmd_callback(const void *msin);
void bin_cext_cmd_callback(const void *msin);
void stalig_state_cmd_callback(const void *msin);
void gservo_cmd_callback(const void *msin);
void lid_cmd_callback(const void *msin);

static bool create_entities()
{
  allocator = rcl_get_default_allocator();
  entities_stage = 0;

  // Domain must be set through init options -- rclc_support_init() takes the
  // default, which is 0. See ROS_DOMAIN above for what that costs.
  //
  // EVERY PATH OUT OF HERE MUST fini THE OPTIONS. rclc_support_init_with_options
  // copies them into the context, so this copy is ours to release. micro-ROS
  // allocates from a fixed static pool, and create_entities() is retried on
  // every reconnect -- so leaking one set per attempt is not a slow leak, it is
  // a board that works on the first connection after a flash and then never
  // again. Observed exactly that: 7 readers and 1 writer on the first agent,
  // then zero sessions on the next.
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

  if (RCL_RET_OK != rclc_node_init_default(&node, "teensy_drill_node", "", &support))
    return false;
  entities_stage = 2;

  // BEST EFFORT, both directions, and it has to stay that way on both ends.
  //
  // /gripper/state publishes at 100 Hz over an XRCE serial stream. On a
  // reliable stream every sample has to be acknowledged by the agent, and when
  // the stream window fills before the ACKs come back the publisher stalls and
  // retransmits the same frame instead of sending new ones -- observed as the
  // identical frame repeating while the topic went silent for seconds, which
  // the host reports as "No /gripper/state for 2.0 s".
  //
  // The host subscription in teensy_gripper_system.cpp is best-effort to match.
  // A BEST_EFFORT publisher and a RELIABLE subscriber are an incompatible pair
  // and DDS makes NO match at all -- the topic lists fine and never delivers.
  if (RCL_RET_OK != rclc_subscription_init_best_effort(
                        &gservo_cmd_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "/gripper/cmd"))
    return false;
  entities_stage = 3;

  if (RCL_RET_OK != rclc_publisher_init_best_effort(
                        &gservo_state_pub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "/gripper/state"))
    return false;
  entities_stage = 4;

  // RELIABLE, and no leading slash under an empty namespace -- which resolves
  // to /stacklight_subscription, the name stacklight.py has always published.
  if (RCL_RET_OK != rclc_subscription_init_default(
                        &stalig_cmd_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8), "stacklight_subscription"))
    return false;
  entities_stage = 5;

  // BEST EFFORT: these two are a continuous rate stream at 30 Hz where the
  // newest value supersedes the last, so a dropped sample costs one cycle and a
  // stalled stream costs the link. Release is covered by the host's half-second
  // burst of zeros and by MOTOR_COMMAND_TIMEOUT_MS here.
  if (RCL_RET_OK != rclc_subscription_init_best_effort(
                        &auger_cmd_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "motor1/cmd_speed"))
    return false;
  entities_stage = 6;

  if (RCL_RET_OK != rclc_subscription_init_best_effort(
                        &feed_cmd_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "motor2/cmd_speed"))
    return false;
  entities_stage = 7;

  // RELIABLE, unlike the two above: these are one-shot events, not a stream.
  // Nothing re-sends them, and a dropped "retract" leaves the bin where it was
  // with the operator believing otherwise.
  if (RCL_RET_OK != rclc_subscription_init_default(
                        &bin_state_cmd_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8), "linact/state"))
    return false;
  entities_stage = 8;

  if (RCL_RET_OK != rclc_subscription_init_default(
                        &bin_cext_cmd_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "linact/cext"))
    return false;
  entities_stage = 9;

  // RELIABLE: a lid is commanded once and then left, so there is no next
  // message to correct a dropped one. Nothing republishes it either -- there is
  // no host node for the lid at all yet.
  if (RCL_RET_OK != rclc_subscription_init_default(
                        &lid_cmd_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "sand_box/lid/cmd"))
    return false;
  entities_stage = 10;

  // SEVEN subscriptions. RMW_UXRCE_MAX_SUBSCRIPTIONS defaults to 5 in
  // micro_ros_platformio, so this needs the raised limit in colcon.meta at the
  // project root (currently 8) -- without it the sixth init above fails and the
  // board is dead. That file is part of the build, not a convenience, and it is
  // the file to check before adding an eighth.
  if (RCL_RET_OK != rclc_executor_init(&executor, &support.context, 7, &allocator))
    return false;
  entities_stage = 11;

  rclc_executor_add_subscription(&executor, &gservo_cmd_sub, &gservo_cmd_msg,
                                 &gservo_cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &stalig_cmd_sub, &stalig_state_cmd_msg,
                                 &stalig_state_cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &auger_cmd_sub, &auger_cmd_msg,
                                 &auger_cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &feed_cmd_sub, &feed_cmd_msg,
                                 &feed_cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &bin_state_cmd_sub, &bin_state_cmd_msg,
                                 &bin_state_cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &bin_cext_cmd_sub, &bin_cext_cmd_msg,
                                 &bin_cext_cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &lid_cmd_sub, &lid_cmd_msg,
                                 &lid_cmd_callback, ON_NEW_DATA);
  return true;
}

static void destroy_entities()
{
  if (entities_stage == 0)
    return;

  rmw_context_t *rmw_ctx = rcl_context_get_rmw_context(&support.context);
  (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_ctx, 0);

  // Reverse creation order.
  if (entities_stage >= 11)
    rclc_executor_fini(&executor);
  if (entities_stage >= 10)
    IGNORE_RC(rcl_subscription_fini(&lid_cmd_sub, &node));
  if (entities_stage >= 9)
    IGNORE_RC(rcl_subscription_fini(&bin_cext_cmd_sub, &node));
  if (entities_stage >= 8)
    IGNORE_RC(rcl_subscription_fini(&bin_state_cmd_sub, &node));
  if (entities_stage >= 7)
    IGNORE_RC(rcl_subscription_fini(&feed_cmd_sub, &node));
  if (entities_stage >= 6)
    IGNORE_RC(rcl_subscription_fini(&auger_cmd_sub, &node));
  if (entities_stage >= 5)
    IGNORE_RC(rcl_subscription_fini(&stalig_cmd_sub, &node));
  if (entities_stage >= 4)
    IGNORE_RC(rcl_publisher_fini(&gservo_state_pub, &node));
  if (entities_stage >= 3)
    IGNORE_RC(rcl_subscription_fini(&gservo_cmd_sub, &node));
  if (entities_stage >= 2)
    IGNORE_RC(rcl_node_fini(&node));
  rclc_support_fini(&support);

  entities_stage = 0;
}

// --- Motion -----------------------------------------------------------------

static void stop_all_motors()
{
  auger_cmd_signed = 0;
  feed_cmd_signed = 0;
  auger.stop_motor();
  feed_motor.stop_motor();
  bin_actuator.stop_motor();
  auger_applied = 0;
  feed_applied = 0;
}

// Push the current commands at the hardware, clamped by the limit switches.
//
// The switches are read as a LEVEL here, every cycle, not as the one-shot edge
// the delivered firmware used. An edge stops the motor exactly once: the flag
// is cleared by the read, so the next command drove straight back into the stop
// with nothing left to stop it again. Gating on the level means holding the pad
// against a closed switch simply does nothing, while the opposite direction
// stays free -- a carriage sitting on the bottom switch must still come back up.
static void apply_motor_commands(bool force)
{
  int feed_pwm = feed_cmd_signed;
  if (feed_pwm > 0 && switch_feed_top.is_at_stop())
    feed_pwm = 0;
  if (feed_pwm < 0 && switch_feed_bottom.is_at_stop())
    feed_pwm = 0;

  if (force || feed_pwm != feed_applied)
  {
    if (feed_pwm == 0)
      feed_motor.stop_motor();
    else
      feed_motor.drive_motor(abs(feed_pwm), feed_pwm > 0);
    feed_applied = feed_pwm;
  }

  // `auger_pwm`, not `auger`: the motor object is called `auger` now, and a
  // local of that name silently shadows it.
  const int auger_pwm = auger_cmd_signed;
  if (force || auger_pwm != auger_applied)
  {
    if (auger_pwm == 0)
      auger.stop_motor();
    else
      auger.drive_motor(abs(auger_pwm), auger_pwm > 0);
    auger_applied = auger_pwm;
  }
}

// --- Status LED -------------------------------------------------------------
//
// The only channel this board has: Serial belongs to micro-ROS.
//
//   fast blink (100 ms)   a pin this firmware needs is still PIN_UNASSIGNED
//   slow blink (500 ms)   waiting for the agent
//   solid on              connected and driving
static void led_update()
{
  static uint32_t last_ms = 0;
  const uint32_t now = millis();

  const bool pins_incomplete =
      !auger.usable() || !feed_motor.usable() || !bin_actuator.usable() ||
      !gservo.usable() || !lid_servo.usable() ||
      !switch_feed_bottom.usable() || !switch_feed_top.usable();

  uint32_t period = 0;
  if (pins_incomplete)
    period = 100;
  else if (uros_state != AGENT_CONNECTED)
    period = 500;

  if (period == 0)
  {
    digitalWrite(LED_BUILTIN, HIGH);
    return;
  }
  if (now - last_ms >= period)
  {
    last_ms = now;
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  }
}

// --- Arduino entry points ---------------------------------------------------

void setup()
{
  // GPIO FIRST, before anything that can block.
  //
  // The delivered firmware waited for the agent at the top of setup() and only
  // then called init_motor() on the three drivers. Until that wait returned,
  // every H-bridge direction pin was an un-driven input: the bridge inputs
  // floated for as long as the rover sat there with no agent running, which is
  // the whole time between powering the rover and starting the launch file. A
  // floating direction pin on a powered bridge is an uncommanded motor. Whatever
  // else this board is doing, it should be holding its outputs at a known safe
  // level within microseconds of reset.
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  switch_feed_bottom.init();
  switch_feed_top.init();
  auger.init_motor();
  feed_motor.init_motor();
  bin_actuator.init_motor();
  stalig.init_light();
  gservo.init();
  lid_servo.init();

  set_microros_serial_transports(Serial);

  // No blocking handshake here either. The loop's state machine does the
  // waiting, so the servo slew, the limit switches and the LED all keep running
  // while the agent is absent, and -- the reason the retired sketch grew this
  // in the first place -- the board rejoins by itself whenever the agent comes
  // back, instead of needing a physical reset after every launch-file restart.
  last_motor_cmd_ms = millis();
}

void loop()
{
  const uint32_t now = millis();
  static uint32_t last_ping_ms = 0;

  // Run regardless of agent state, so both servos always track their last
  // commanded target smoothly. The lid is included for the boot case: with no
  // agent ever connected it still slews to, and holds, CLOSED.
  gservo.update();
  lid_servo.update();

  // Watchdog. Also covers the disconnected case: no commands can arrive, so
  // this is what actually stops a motor that was running when the link died.
  if (now - last_motor_cmd_ms > MOTOR_COMMAND_TIMEOUT_MS)
  {
    if (auger_cmd_signed != 0 || feed_cmd_signed != 0)
    {
      auger_cmd_signed = 0;
      feed_cmd_signed = 0;
    }
  }

  // Re-evaluated every cycle so a switch closing stops the motor even when no
  // new command has arrived.
  apply_motor_commands(false);

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
  {
    // Ping every 500 ms; allow 3 retries x 200 ms before declaring a
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

    // 2 ms, not 10: the 100 Hz /gripper/cmd stream is at the edge of a 10 ms
    // budget once loop overhead is counted, and the messages it drops are
    // gripper commands.
    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(2));

    static uint32_t last_state_pub_ms = 0;
    if (millis() - last_state_pub_ms >= GRIPPER_STATE_PERIOD_MS)
    {
      last_state_pub_ms = millis();
      gservo_state_msg.data = gservo.current();
      IGNORE_RC(rcl_publish(&gservo_state_pub, &gservo_state_msg, NULL));
    }
    break;
  }

  case AGENT_DISCONNECTED:
    // Everything off before the teardown. The link is gone, so no stop command
    // can arrive; leaving the auger turning until the watchdog notices would be
    // half a second of a cutting tool nobody is talking to.
    stop_all_motors();
    destroy_entities();
    // Re-initialise the transport so the agent can reopen the port immediately
    // instead of waiting for USB re-enumeration.
    set_microros_serial_transports(Serial);
    uros_state = WAITING_AGENT;
    break;
  }
}

// --- Callbacks --------------------------------------------------------------

void auger_cmd_callback(const void *msin)
{
  const std_msgs__msg__Int32 *msg = (const std_msgs__msg__Int32 *)msin;
  auger_cmd_signed = constrain(msg->data, -255, 255);
  last_motor_cmd_ms = millis();
  apply_motor_commands(false);
}

void feed_cmd_callback(const void *msin)
{
  const std_msgs__msg__Int32 *msg = (const std_msgs__msg__Int32 *)msin;
  feed_cmd_signed = constrain(msg->data, -255, 255);
  last_motor_cmd_ms = millis();
  apply_motor_commands(false);
}

void bin_state_cmd_callback(const void *msin)
{
  const std_msgs__msg__UInt8 *msg = (const std_msgs__msg__UInt8 *)msin;

  uint8_t state = msg->data;
  if (state == 1)
  {
    bin_actuator.extend();
  }
  else if (state == 2)
  {
    bin_actuator.retract();
  }
  else if (state == 3)
  {
    bin_actuator.home(true);
  }
  else if (state == 4)
  {
    bin_actuator.home(false);
  }
  else
  {
    // 0, or anything unrecognised, is a stop. The delivered code fell through
    // and did nothing, so there was no value on this topic that meant "stop"
    // once a move had started.
    bin_actuator.stop_motor();
  }
}

void bin_cext_cmd_callback(const void *msin)
{
  const std_msgs__msg__Float32 *msg = (const std_msgs__msg__Float32 *)msin;

  const float cext = msg->data;
  if (cext > 0.0f)
  {
    bin_actuator.extend(255, cext);
  }
  else if (cext < 0.0f)
  {
    bin_actuator.retract(255, -cext);
  }
  else
  {
    // Exactly 0.0 used to reach retract(255, 0.0), which armed an
    // IntervalTimer with a 0 us period. The timer never fired, so the motor
    // that had just been switched on was never switched off: a zero-length move
    // ran the actuator into its end stop. Stop is the only sensible reading of
    // a zero extension.
    bin_actuator.stop_motor();
  }
}

void stalig_state_cmd_callback(const void *msin)
{
  const std_msgs__msg__UInt8 *msg = (const std_msgs__msg__UInt8 *)msin;
  stalig.state(msg->data);
}

void gservo_cmd_callback(const void *msin)
{
  const std_msgs__msg__Float32 *msg = (const std_msgs__msg__Float32 *)msin;
  gservo.set_target(msg->data);
}

void lid_cmd_callback(const void *msin)
{
  // 0.0 closed, 1.0 open, anything between held there. set_target clamps, so
  // an out-of-range value is the nearer end rather than a servo driven past its
  // stop. No watchdog: a lid is meant to stay where it was put, and closing it
  // on a lost link would tip out a sample the operator was mid-way through
  // collecting.
  const std_msgs__msg__Float32 *msg = (const std_msgs__msg__Float32 *)msin;
  lid_servo.set_target(msg->data);
}

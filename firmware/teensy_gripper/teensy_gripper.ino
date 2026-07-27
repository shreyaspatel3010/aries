// Teensy Gripper Firmware — micro-ROS over USB Serial
//
// Subscribes to /gripper/cmd             (std_msgs/Float32, normalized 0.0–1.0)
// Publishes  to /gripper/state           (std_msgs/Float32, normalized 0.0–1.0)
// Subscribes to stacklight_subscription  (std_msgs/UInt8, 1=red 2=yellow 3=green 4=disable)
//
// Automatically reconnects to the micro-ROS agent whenever it appears —
// no physical disconnect/reconnect needed after restarting the launch file.
//
// Requirements:
//   - micro_ros_arduino, JAZZY build, to match this workspace's ROS 2 distro:
//       https://github.com/micro-ROS/micro_ros_arduino/releases  (v2.0.8-jazzy)
//     The distro must match, and so must the toolchain: the library ships
//     libmicroros.a PRECOMPILED, so a build made against an older newlib fails
//     to link against a current Teensy core with
//       undefined reference to `__locale_ctype_ptr'
//     out of librmw-validate_node_name / librcl-validate_topic_name. The
//     3.0.0-iron build did exactly that on Teensy core 1.60.0 (gcc 11.3.1) and
//     made this sketch unbuildable. If that error returns, the installed
//     library is the wrong build — replace it rather than patching the symbol.
//   - Arduino IDE: Tools > Board > Teensy 4.x
//                  Tools > USB Type > Serial
//
// The agent is started automatically by aries_hardware.launch.py.
// To run it manually:
//   ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200
//
// Do not raise that baud: Linux speed_t values are encodings, not bit rates,
// and the largest valid one is B4000000 (== 4111). Anything above it — 6000000
// was used here previously — is rejected by cfsetospeed with EINVAL, which the
// agent does not check, leaving the port speed unset. This link is USB CDC
// (Tools > USB Type > Serial), so the device ignores baud anyway.

#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float32.h>
#include <std_msgs/msg/u_int8.h>
#include <Servo.h>

// --- STACKLIGHT PINS ---
#define RED_PIN    PIN_A4   // pin 18
#define YELLOW_PIN PIN_A5   // pin 19
#define GREEN_PIN  PIN_A8   // pin 22

enum stacklight_color { red = 1, yellow, green, disable };

// --- FORWARD DECLARATIONS ---
void stacklight_callback(const void *msg_in);
void cmd_callback(const void *msg_in);

// --- CONFIGURE THESE ---
const int   SERVO_PIN    = 9;
const int   SERVO_MIN_US = 800;
const int   SERVO_MAX_US = 2300;
const bool  USE_SERVO_FEEDBACK = false;
const int   SERVO_FEEDBACK_PIN = A0;
const int   FEEDBACK_MIN_ADC   = 200;
const int   FEEDBACK_MAX_ADC   = 850;
const bool  DETACH_WHEN_IDLE   = false;
// Must be > 2000 ms (C++ side stale-state limit) so the servo never detaches
// during a brief micro-ROS disconnect before the keepalive can resume.
const unsigned long IDLE_DETACH_MS = 3000;
// Slew-rate limit — keeps servo motion smooth despite USB-serial timing jitter.
// DSC55MG-270 spec: 0.10 s/60° = 600°/s. Use slightly under physical max.
const float SERVO_SLEW_DEG_PER_SEC = 550.0f;
// -----------------------

// micro-ROS state machine
enum MicroROSState { WAITING_AGENT, AGENT_AVAILABLE, AGENT_CONNECTED, AGENT_DISCONNECTED };
MicroROSState uros_state = WAITING_AGENT;

// micro-ROS objects
rcl_node_t             node;
rcl_allocator_t        allocator;
rclc_support_t         support;
rclc_executor_t        executor;
rcl_subscription_t     cmd_sub;
rcl_publisher_t        state_pub;
std_msgs__msg__Float32 cmd_msg;
std_msgs__msg__Float32 state_msg;
rcl_subscription_t     stacklight_sub;
std_msgs__msg__UInt8   stacklight_msg;

Servo gripper;
float currentNormalized = 0.0f;  // actual (slewed) servo position
float targetNormalized  = 0.0f;  // desired position from latest ROS2 command
bool  servoAttached     = false;
unsigned long lastCommandMs = 0;
int normalizedToUs(float t) {
  if (t < 0.0f) t = 0.0f;
  if (t > 1.0f) t = 1.0f;
  return (int)(SERVO_MIN_US + t * (SERVO_MAX_US - SERVO_MIN_US));
}

float readFeedbackNormalized() {
  if (!USE_SERVO_FEEDBACK) return currentNormalized;
  int raw = analogRead(SERVO_FEEDBACK_PIN);
  if (raw < FEEDBACK_MIN_ADC) raw = FEEDBACK_MIN_ADC;
  if (raw > FEEDBACK_MAX_ADC) raw = FEEDBACK_MAX_ADC;
  float t = (float)(raw - FEEDBACK_MIN_ADC) / (float)(FEEDBACK_MAX_ADC - FEEDBACK_MIN_ADC);
  if (t < 0.0f) t = 0.0f;
  if (t > 1.0f) t = 1.0f;
  return t;
}

// Slew-rate limiter — called every loop() iteration.
// Moves currentNormalized toward targetNormalized at SERVO_SLEW_DEG_PER_SEC,
// then writes the PWM. This decouples servo motion from command arrival timing,
// eliminating jerkiness caused by USB-serial jitter.
void updateServo() {
  static unsigned long lastMs = 0;
  unsigned long nowMs = millis();
  unsigned long dtMs  = nowMs - lastMs;
  if (dtMs == 0) return;
  lastMs = nowMs;

  // Servo range spans 270°; normalize maxStep accordingly.
  const float maxStep = SERVO_SLEW_DEG_PER_SEC / 270.0f / 1000.0f * (float)dtMs;
  float diff = targetNormalized - currentNormalized;
  if      (diff >  maxStep) currentNormalized += maxStep;
  else if (diff < -maxStep) currentNormalized -= maxStep;
  else                      currentNormalized  = targetNormalized;

  if (servoAttached) {
    gripper.writeMicroseconds(normalizedToUs(currentNormalized));
  }
}

// Callback: received /gripper/cmd — just update the target; servo tracking
// happens in updateServo() at ~500 Hz, independent of command arrival timing.
void cmd_callback(const void * msg_in) {
  const std_msgs__msg__Float32 * msg = (const std_msgs__msg__Float32 *)msg_in;
  float normalized = msg->data;
  if (normalized < 0.0f) normalized = 0.0f;
  if (normalized > 1.0f) normalized = 1.0f;
  targetNormalized = normalized;
  lastCommandMs = millis();
}

// ---------- micro-ROS lifecycle ----------

// How far create_entities() got last time.  destroy_entities() is called on the
// failure path too (see AGENT_AVAILABLE in loop()), and finalising a handle that
// was never initialised is not a no-op: support/node/cmd_sub are zero-filled
// globals, so rcl_context_get_rmw_context(&support.context) dereferences a NULL
// impl pointer and hard-faults the MCU.  A faulted sketch stops pinging and
// stops publishing while USB stays enumerated, so the agent sits there with the
// port open and the board never rejoins — indistinguishable from an unflashed
// Teensy, and only a physical reset clears it.  Tear down exactly what exists.
uint8_t entities_stage = 0;   // 1 support, 2 node, 3 cmd_sub, 4 state_pub,
                              // 5 stacklight_sub, 6 executor

bool create_entities() {
  allocator = rcl_get_default_allocator();
  entities_stage = 0;
  if (RCL_RET_OK != rclc_support_init(&support, 0, NULL, &allocator))         return false;
  entities_stage = 1;
  if (RCL_RET_OK != rclc_node_init_default(&node, "teensy_gripper", "", &support)) return false;
  entities_stage = 2;
  if (RCL_RET_OK != rclc_subscription_init_best_effort(
        &cmd_sub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "/gripper/cmd"))  return false;
  entities_stage = 3;
  // BEST EFFORT, not _init_default (which is RELIABLE). This publishes at
  // 100 Hz over a serial XRCE stream: on a reliable stream every sample has to
  // be acknowledged by the agent, and when the stream window fills before the
  // ACKs come back the publisher stalls and retransmits the same frame instead
  // of sending new ones. Captured on the wire as the identical frame repeating
  // while /gripper/state went silent for seconds at a time, which the host side
  // reports as "No /gripper/state for 2.0 s — Teensy session is down".
  // State is a periodic sample: losing one is harmless, blocking the stream is
  // not. cmd_sub above is best-effort for the same reason.
  // NOTE: the host subscription in teensy_gripper_system.cpp must stay
  // best-effort too — a BEST_EFFORT publisher and a RELIABLE subscriber are
  // incompatible and will not match at all.
  if (RCL_RET_OK != rclc_publisher_init_best_effort(
        &state_pub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "/gripper/state")) return false;
  entities_stage = 4;
  if (RCL_RET_OK != rclc_subscription_init_default(
        &stacklight_sub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8), "stacklight_subscription")) return false;
  entities_stage = 5;
  if (RCL_RET_OK != rclc_executor_init(&executor, &support.context, 2, &allocator)) return false;
  entities_stage = 6;
  rclc_executor_add_subscription(&executor, &cmd_sub, &cmd_msg, &cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &stacklight_sub, &stacklight_msg, &stacklight_callback, ON_NEW_DATA);
  return true;
}

void destroy_entities() {
  if (entities_stage == 0) return;   // nothing was ever created

  rmw_context_t *rmw_ctx = rcl_context_get_rmw_context(&support.context);
  (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_ctx, 0);

  // Reverse creation order.
  if (entities_stage >= 6) rclc_executor_fini(&executor);
  if (entities_stage >= 5) rcl_subscription_fini(&stacklight_sub, &node);
  if (entities_stage >= 4) rcl_publisher_fini(&state_pub, &node);
  if (entities_stage >= 3) rcl_subscription_fini(&cmd_sub, &node);
  if (entities_stage >= 2) rcl_node_fini(&node);
  rclc_support_fini(&support);

  entities_stage = 0;
}

// ---------- Arduino entry points ----------

void setup() {
  // GPIO — initialised once, independent of agent state
  pinMode(RED_PIN,    OUTPUT); digitalWrite(RED_PIN,    LOW);
  pinMode(YELLOW_PIN, OUTPUT); digitalWrite(YELLOW_PIN, LOW);
  pinMode(GREEN_PIN,  OUTPUT); digitalWrite(GREEN_PIN,  LOW);

  if (!DETACH_WHEN_IDLE) {
    gripper.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    gripper.writeMicroseconds(normalizedToUs(currentNormalized));
    servoAttached = true;
  }
  if (USE_SERVO_FEEDBACK) {
    pinMode(SERVO_FEEDBACK_PIN, INPUT);
  }

  set_microros_transports();
}

void loop() {
  static unsigned long last_ping_ms = 0;
  unsigned long now = millis();

  // Run slew-rate update every loop iteration regardless of agent state so the
  // servo always tracks the last commanded target smoothly.
  updateServo();

  switch (uros_state) {

    case WAITING_AGENT:
      if (now - last_ping_ms > 500) {
        last_ping_ms = now;
        uros_state = (RMW_RET_OK == rmw_uros_ping_agent(200, 1))
                     ? AGENT_AVAILABLE : WAITING_AGENT;
      }
      break;

    case AGENT_AVAILABLE:
      uros_state = create_entities() ? AGENT_CONNECTED : WAITING_AGENT;
      if (uros_state == WAITING_AGENT) destroy_entities();
      break;

    case AGENT_CONNECTED:
      // Ping every 500 ms; allow 3 retries × 200 ms before declaring disconnect.
      // This tolerates brief serial latency without triggering a full session teardown.
      if (now - last_ping_ms > 500) {
        last_ping_ms = now;
        uros_state = (RMW_RET_OK == rmw_uros_ping_agent(200, 3))
                     ? AGENT_CONNECTED : AGENT_DISCONNECTED;
      }
      if (uros_state == AGENT_CONNECTED) {
        // 2 ms timeout: loop runs at ~500 Hz, ensuring no 100 Hz /gripper/cmd
        // messages are ever missed due to timing jitter on either side.
        // The previous 10 ms value was at the edge of the cmd publish period
        // and could cause occasional missed messages when loop overhead made
        // the cycle slightly longer than 10 ms.
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(2));
        
        // --- IDLE LOGIC AND EEPROM SAVE ---
        if (millis() - lastCommandMs > IDLE_DETACH_MS) {
          
          if (DETACH_WHEN_IDLE && servoAttached) {
            gripper.detach();
            servoAttached = false;
          }
        }

        // Publish state at 100 Hz (10 ms) to match the ROS2 control loop rate.
        static unsigned long last_state_pub_ms = 0;
        if (millis() - last_state_pub_ms >= 10) {
          last_state_pub_ms = millis();
          state_msg.data = readFeedbackNormalized();
          rcl_publish(&state_pub, &state_msg, NULL);
        }
      }
      break;

    case AGENT_DISCONNECTED:
      destroy_entities();
      // Re-initialise the serial transport so the agent can reopen the port
      // immediately instead of waiting for USB re-enumeration.
      set_microros_transports();
      uros_state = WAITING_AGENT;
      break;
  }
}

void stacklight_callback(const void *msg_in) {
  const std_msgs__msg__UInt8 *msg = (const std_msgs__msg__UInt8 *)msg_in;
  if (!msg) return;

  digitalWrite(RED_PIN,    LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN,  LOW);

  switch (msg->data) {
    case stacklight_color::red:     digitalWrite(RED_PIN,    HIGH); break;
    case stacklight_color::yellow:  digitalWrite(YELLOW_PIN, HIGH); break;
    case stacklight_color::green:   digitalWrite(GREEN_PIN,  HIGH); break;
    case stacklight_color::disable: /* all already LOW */           break;
    default:                        digitalWrite(RED_PIN,    HIGH); break;
  }
}
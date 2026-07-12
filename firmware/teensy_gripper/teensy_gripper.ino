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
//   - Install the micro_ros_arduino library (https://github.com/micro-ROS/micro_ros_arduino)
//   - Arduino IDE: Tools > Board > Teensy 4.x
//                  Tools > USB Type > Serial
//
// The agent is started automatically by aries_hardware.launch.py.
// To run it manually:
//   ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 6000000

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

bool create_entities() {
  allocator = rcl_get_default_allocator();
  if (RCL_RET_OK != rclc_support_init(&support, 0, NULL, &allocator))         return false;
  if (RCL_RET_OK != rclc_node_init_default(&node, "teensy_gripper", "", &support)) return false;
  if (RCL_RET_OK != rclc_subscription_init_best_effort(
        &cmd_sub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "/gripper/cmd"))  return false;
  if (RCL_RET_OK != rclc_publisher_init_default(
        &state_pub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "/gripper/state")) return false;
  if (RCL_RET_OK != rclc_subscription_init_default(
        &stacklight_sub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8), "stacklight_subscription")) return false;
  if (RCL_RET_OK != rclc_executor_init(&executor, &support.context, 2, &allocator)) return false;
  rclc_executor_add_subscription(&executor, &cmd_sub, &cmd_msg, &cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &stacklight_sub, &stacklight_msg, &stacklight_callback, ON_NEW_DATA);
  return true;
}

void destroy_entities() {
  rmw_context_t *rmw_ctx = rcl_context_get_rmw_context(&support.context);
  (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_ctx, 0);
  rcl_publisher_fini(&state_pub, &node);
  rcl_subscription_fini(&cmd_sub, &node);
  rcl_subscription_fini(&stacklight_sub, &node);
  rclc_executor_fini(&executor);
  rcl_node_fini(&node);
  rclc_support_fini(&support);
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
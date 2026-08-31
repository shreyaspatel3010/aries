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
//   drill/limits               UInt8    bit0 bottom, bit1 top,          RELIABLE
//                                       bit2/3 = sign believed to        switch_feed_*
//                                       drive INTO top/bottom
//   stacklight_subscription    UInt8    1=red 2=yellow 3=green 4=off   stalig
//   motor1/cmd_speed           Int32    -255..255  AUGER, spins        auger
//   motor2/cmd_speed           Int32    -255..255  FEED, drill up/down feed_motor
//   linact/state               UInt8    1=extend 2=retract 3/4=home    bin_actuator
//   linact/cext                Float32  signed mm, sample bin fore/aft bin_actuator
//   sand_box/lid/cmd           Float32  lid SPEED, -1..1, 0=stop       lid_servo
//   pump/state                 UInt8    1=release 2=draw 3/4=home      pump
//                                       5=home-then-draw, 0=stop
//   pump/purge                 Float32  SECONDS of reverse run to      pump
//                                       empty the tube; <=0 and NaN
//                                       stop, clamped to 120 s
//   load_cells/raw             Int32MultiArray  three RAW converter    load_cell_*
//                                       counts, 10 Hz, in the order
//                                       sand box / stone box / bin
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
// THE LID IS ON THE PAD as of 2026-08-29: LT + right stick up/down, published
// by aries_teleop's drill_joystick at 30 Hz while the stick is held, with the
// usual half-second burst of zeros on release. That burst and the watchdog here
// are two independent ways for the lid to stop; it needs both, because it
// cannot stop itself.
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
#include <std_msgs/msg/int32_multi_array.h>
#include <std_msgs/msg/u_int32.h>
#include <std_msgs/msg/u_int64.h>
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

// load_cells/raw publish period. 10 Hz, matching publish_rate_hz in
// aries_load_cells/config/load_cells.yaml and the HX711's own 10 SPS: faster
// here would republish the same conversion under a new timestamp, which reads
// as a live sensor and is not one.
static const uint32_t LOAD_CELL_PERIOD_MS = 100;


// How long the lid may keep turning after the last command. It is a
// continuous-rotation servo with no end of travel and no stop of its own, so
// this is the only thing that stops it if drill_joystick dies or the link drops
// mid-move. 500 ms at the joystick's 30 Hz is fifteen missed messages -- far
// past ordinary jitter, and a fraction of the lid's travel.
static const uint32_t LID_COMMAND_TIMEOUT_MS = 500;

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

// The peristaltic pump, on the H-bridge the sample bin vacated when it moved
// to 22 / 19 / 18. Commanded in millilitres and run open-loop on a timer --
// there is no flow sensor -- so every dose is an estimate that drifts with head
// height, tube wear and battery state, exactly as the drill's three axes do.
rcl_subscription_t pump_cmd_sub;
std_msgs__msg__UInt8 pump_cmd_msg;

// SECOND TOPIC ON ONE MECHANISM, for the same reason the sample bin has two:
// pump/state is a UInt8 menu and a purge needs a NUMBER -- how many seconds to
// run -- which does not fit in a menu entry. The bin split state/cext the same
// way. Everything either topic starts, pump/state 0 still stops.
rcl_subscription_t pump_purge_sub;
std_msgs__msg__Float32 pump_purge_msg;

Pump pump(PUMP_PWM, PUMP_INA, PUMP_INB);

rcl_subscription_t stalig_cmd_sub;
std_msgs__msg__UInt8 stalig_state_cmd_msg;
StackLight stalig(STALIG_G, STALIG_Y, STALIG_R);

rcl_subscription_t gservo_cmd_sub;
std_msgs__msg__Float32 gservo_cmd_msg;
rcl_publisher_t gservo_state_pub;
std_msgs__msg__Float32 gservo_state_msg;
SlewServo gservo(GRIPPER_SERVO);

// Lid of the front-left deck container (the sand box). A CONTINUOUS-ROTATION
// servo, so the command is a speed in -1..1 and NOT a position -- see LidServo
// in drill.h for why that distinction is the whole class.
//
// It does not snap shut at power-up the way the old positional version did.
// init() writes the neutral (stop) pulse and nothing else, so a lid left open
// stays open until somebody drives it. That is the safer boot for a servo that
// cannot tell where it is: the alternative would be turning at full speed
// toward a closed position it has no way to detect reaching.
rcl_subscription_t lid_cmd_sub;
std_msgs__msg__Float32 lid_cmd_msg;
LidServo lid_servo(LID_SERVO_SAND_BOX, LID_SERVO_NEUTRAL_US, LID_SERVO_MAX_DEVIATION_US);

// Both switches are on the FEED CARRIAGE, one at each end of its travel. The
// auger is a spindle and has no end of travel; the delivered firmware stopped
// it on switch 1 anyway. See pins.h.
LimitSwitch switch_feed_bottom(LIMIT_SWITCH1, 50);
LimitSwitch switch_feed_top(LIMIT_SWITCH2, 50);

// THE ONLY SENSOR THE DRILL HAS. There is no encoder on any of the three axes,
// so these two switches are the entire feedback path -- and until 2026-08-27
// the firmware kept them to itself. Nothing on the host could tell a switch
// that was working from one on the wrong pin, unwired, or never closing,
// because all three look identical: the carriage simply does not stop.
// drill_joystick.py had filled that hole by dead-reckoning a position from
// the commanded rate and gating on the URDF limits, which is a guess, not a
// measurement, and drifts from the first slipped count onward.
//
// bit0 = bottom switch closed, bit1 = top switch closed. RELIABLE and
// published on change (plus a slow heartbeat) rather than streamed: it is an
// event, it is rare, and a dropped edge is exactly the sample that matters.
rcl_publisher_t drill_limits_pub;
std_msgs__msg__UInt8 drill_limits_msg;

// --- Load cells -------------------------------------------------------------
//
// Three HX711 amplifiers, each with its own private DT/SCK pair -- NOT the
// usual shared-clock chain. Six pins. See pins.h and PINOUT.md.
//
// THE CONSTRUCTION ORDER BELOW IS THE WIRE FORMAT. load_cells/raw carries one
// element per cell in the order of `cells` in
// aries_load_cells/config/load_cells.yaml:
//
//     ["sand_box", "stone_box", "drill_container"]
//
// The firmware sends no names, so there is nothing anywhere to catch a swap:
// exchange two entries here and the sand box reports the stone, both numbers
// stay entirely plausible, and no log line anywhere says a word.
//
// RAW COUNTS, NOT KILOGRAMS. Scale, offset and tare live in that package's
// YAML so a recalibration is an edit and a relaunch rather than a reflash with
// the rover open, and this board stays a thing that reads ADCs.
#define LOAD_CELL_COUNT 3
LoadCell load_cells[LOAD_CELL_COUNT] = {
    LoadCell(HX711_SAND_BOX_DT, HX711_SAND_BOX_SCK),
    LoadCell(HX711_STONE_BOX_DT, HX711_STONE_BOX_SCK),
    LoadCell(HX711_DRILL_CONTAINER_DT, HX711_DRILL_CONTAINER_SCK),
};

rcl_publisher_t load_cells_raw_pub;
std_msgs__msg__Int32MultiArray load_cells_raw_msg;

// THE MESSAGE BODY, AND IT HAS TO BE HANDED TO THE MESSAGE BY HAND.
//
// Int32MultiArray is the first message on this board with a DYNAMIC ARRAY in
// it, and micro-ROS does not allocate one for you. The message is a zero-filled
// global, so `data.data` is NULL and `data.capacity` is 0 until something
// assigns them -- and rcl_publish on that does not fail loudly, it serialises a
// zero-length array. The host then reports "load_cells/raw carried 0 counts for
// 3 cells; ignoring the message" forever while every entity looks perfectly
// healthy on both ends.
//
// Static, not malloc'd: the size is fixed at three and the board has no
// business taking a heap allocation on a path this hot.
static int32_t load_cells_raw_buf[LOAD_CELL_COUNT];

// Whether any amplifier has ever answered. Until one has, the array is NOT
// published at all.
//
// The alternative -- publish three rails from the first second -- is worse in
// the case that is true today: no cells fitted, and the host would show three
// standing faults that mean nothing more than "there is no hardware here". A
// silent topic is exactly what aries_load_cells' own "no counts yet - is the
// Teensy's firmware publishing?" message is for. Once ONE cell is alive the
// array goes out complete every cycle, rails included, so a single unplugged
// amplifier among three working ones is loud rather than reported as an empty
// box.
static bool load_cells_present = false;

// --- Pin scan: WHERE IS THE SWITCH ACTUALLY WIRED? --------------------------
//
// A switch that never closes and a switch on the wrong pin are the same event
// from the host -- an INPUT_PULLUP pin reads HIGH both when the switch is open
// and when nothing is connected to it at all. drill/limits reporting 0
// therefore cannot distinguish "correctly wired, carriage mid-travel" from
// "these two pins are not connected to anything".
//
// So the board checks every pin it is NOT using. Hold a limit switch closed and
// exactly one bit goes high, and that bit IS the pin the switch is on -- whether
// or not it is the pin pins.h believes. If NO bit goes high, the switch is not
// reaching the Teensy at all (open circuit, or it is not switching to GND) and
// no amount of renumbering pins.h will help.
//
// INPUT ONLY. Every pin here is configured INPUT_PULLUP and never driven, so
// this cannot fight anything already on the harness.
//
// 38 IS DELIBERATELY NOT IN THIS LIST since the lid servo moved onto it. A pin
// that is scanned AND driven is worse than either alone: setup() would make it
// INPUT_PULLUP, Servo::attach() would take it back as an OUTPUT, and the scan
// would then report the servo's own 50 Hz pulse train as a switch closing and
// opening forever. Keep this list and pins.h disjoint -- nothing checks it.
//
// 8, 9 AND 25 LEFT THIS LIST ON 2026-08-29, and 28, 29 and 30 joined it, when
// the auger moved onto 25/8/9 and the sample bin onto the auger's old 22/19/18.
// This is the same hazard as pin 38 above and it had already landed: the scan
// loop in setup() runs BEFORE auger.init_motor(), so the auger still won the
// pin mode and drove correctly, but pin_scan_state() went on reading its PWM
// and direction lines and reporting them as three bits chattering on
// drill/pin_scan whenever the auger turned -- blinding check_drill_limits.py,
// which is the one tool that can find a mis-wired switch.
//
// ADD THE PIN YOU FREE, AND REMOVE THE PIN YOU TAKE. Nothing checks this; the
// static_assert in pins.h covers kMap only, not this list.
static const uint8_t kScanPins[] = {
    0, 1, 2, 3, 4, 5, 10, 11, 12, 14, 20, 21, 24, 26, 27, 39,
    // The two the switches are SUPPOSED to be on, reported alongside for
    // comparison. Already INPUT_PULLUP via LimitSwitch::init().
    LIMIT_SWITCH1, LIMIT_SWITCH2,
};
static const uint8_t kScanCount = sizeof(kScanPins) / sizeof(kScanPins[0]);

rcl_publisher_t pin_scan_pub;
// 64 bits, not 32: pins 38 and 39 are free on a Teensy 4.1 and a 32-bit mask
// silently cannot address them, so a switch wired there was invisible to the
// first version of this scan.
std_msgs__msg__UInt64 pin_scan_msg;

// Bit N is set when digital pin N reads LOW -- i.e. something is pulling it
// down. Bit index IS the pin number, so the value decodes by eye.
static uint64_t pin_scan_state()
{
  uint64_t bits = 0;
  for (uint8_t i = 0; i < kScanCount; ++i)
  {
    const uint8_t pin = kScanPins[i];
    if (pin < 64 && digitalRead(pin) == LOW)
      bits |= (uint64_t)1 << pin;
  }
  return bits;
}

// --- Which PWM sign drives INTO each switch ---------------------------------
//
// This used to be assumed: positive raises the carriage, so positive runs into
// the TOP switch. A WRONG assumption here is invisible and total -- each switch
// gets consulted for the direction that moves AWAY from it, so the carriage
// drives through both stops and every switch looks dead while being read
// perfectly. It has been wrong twice on this machine already (a reversed
// H-bridge, then a bottom/top swap), and nothing reported it either time.
//
// So it is no longer assumed. Seeded from the convention, then corrected from
// what the mechanism actually does:
//
// A switch learns ONCE PER CLOSURE, on the edge, from the sign that was
// driving when it closed. That sign is then blocked for as long as the switch
// stays closed, and the opposite sign is ALWAYS free -- hit the top switch and
// down still works; hit the bottom switch and up still works. Escape is never
// gated on anything the firmware had to infer.
//
// LEARNING ONCE PER CLOSURE IS THE WHOLE POINT, and an earlier version of this
// got it wrong in a way worth recording. It also re-learned from a switch that
// merely STAYED closed while a sign was driven, meaning to catch a carriage
// sitting on a stop at power-up. But leaving a switch is not instant: drive
// down off the top switch and it is still closed for the first few
// millimetres. That rule would then conclude "down drives INTO the top switch",
// block the escape, and re-open the path back into the stop -- trapping the
// carriage against the very switch that was protecting it, and doing it
// worse than no gate at all.
//
// The closure EPISODE is what makes once-per-closure safe against contact
// bounce. is_at_stop() is a raw digitalRead with no debounce (the 50 ms
// debounce in LimitSwitch belongs to the interrupt, which nothing reads), so a
// bouncing contact on release would otherwise re-learn from the escape sign and
// cause the identical trap. An episode ends only after the switch has been
// continuously open for kReleaseMs, by which time the carriage has cleared it.
static const uint32_t kReleaseMs = 100;

// Seeded from the documented convention -- positive PWM raises the carriage,
// so positive runs into the TOP switch -- and corrected by the rule above the
// first time each switch is actually seen closing.
static int8_t sign_into_top = +1;
static int8_t sign_into_bottom = -1;

static inline int8_t sign_of(int v) { return v > 0 ? +1 : (v < 0 ? -1 : 0); }

// Rebuilt from the switches every cycle; see publish_drill_limits().
static uint8_t drill_limits_state()
{
  uint8_t bits = 0;
  if (switch_feed_bottom.is_at_stop())
    bits |= 0x01;
  if (switch_feed_top.is_at_stop())
    bits |= 0x02;
  // bit2/bit3: which PWM sign the gate currently believes drives INTO the top
  // and bottom switch (set = positive). Published so a wrong belief is visible
  // instead of silently disabling both stops.
  if (sign_into_top > 0)
    bits |= 0x04;
  if (sign_into_bottom > 0)
    bits |= 0x08;
  return bits;
}

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
void pump_cmd_callback(const void *msin)
{
  const std_msgs__msg__UInt8 *msg = (const std_msgs__msg__UInt8 *)msin;

  switch (msg->data)
  {
  case 1:
    pump.release();
    break;
  case 2:
    pump.draw();
    break;
  case 3:
    pump.home(true);
    break;
  case 4:
    pump.home(false);
    break;
  case 5:
    // Home then draw, as ONE run. Upstream wrote this as two back-to-back
    // calls, where the second re-armed the same IntervalTimer and cancelled the
    // first -- see Pump::home_then_draw().
    pump.home_then_draw();
    break;
  default:
    // 0, or anything unrecognised, stops. The bin's callback grew this for the
    // same reason: without it there is no value on this topic that means "stop"
    // once a dose has started, and a pump mid-dose is the one axis on this
    // board that is moving something it cannot take back.
    pump.stop_motor();
    break;
  }
}

// REVERSE RUN, to clear the tube. One Float32 of SECONDS; see Pump::purge()
// for why this is a duration and not a volume, and why a 0 stops rather than
// starting a default-length run.
//
// This is the same direction pump/state 1 and 3 drive. 3 is the fixed 30 s
// version of it -- this topic exists because emptying a line is watched, and
// the operator wants to say how long and then cut it short.
void pump_purge_callback(const void *msin)
{
  const std_msgs__msg__Float32 *msg = (const std_msgs__msg__Float32 *)msin;
  pump.purge(msg->data);
}

void stalig_state_cmd_callback(const void *msin);
void gservo_cmd_callback(const void *msin);
void lid_cmd_callback(const void *msin);
void pump_cmd_callback(const void *msin);
void pump_purge_callback(const void *msin);

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

  // RELIABLE, and for the strongest version of the reason the bin's two topics
  // are: this is a one-shot that MOVES LIQUID. Nothing re-sends it, and a
  // dropped dose is a dose the operator believes happened.
  if (RCL_RET_OK != rclc_subscription_init_default(
                        &pump_cmd_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8), "pump/state"))
    return false;
  entities_stage = 11;

  // RELIABLE, for the same reason as pump/state above and one more: this
  // topic's 0 is how a purge in progress is CUT SHORT. A dropped stop leaves
  // the pump running for the rest of a duration the operator has already
  // decided is too long.
  if (RCL_RET_OK != rclc_subscription_init_default(
                        &pump_purge_sub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "pump/purge"))
    return false;
  entities_stage = 12;

  // NINE subscriptions. RMW_UXRCE_MAX_SUBSCRIPTIONS defaults to 5 in
  // micro_ros_platformio, so this needs the raised limit in colcon.meta at the
  // project root (currently 10) -- without it the sixth init above fails and
  // the board is dead. That file is part of the build, not a convenience, and
  // THERE IS EXACTLY ONE SLOT LEFT: a tenth subscription is the last one that
  // fits, and an eleventh needs that number raised first.
  // RELIABLE, unlike /gripper/state: this is an edge, not a stream. Losing the
  // sample where a switch closes is losing the whole message.
  if (RCL_RET_OK != rclc_publisher_init_default(
                        &drill_limits_pub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8), "drill/limits"))
    return false;
  entities_stage = 13;

  if (RCL_RET_OK != rclc_publisher_init_default(
                        &pin_scan_pub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt64), "drill/pin_scan"))
    return false;
  entities_stage = 14;

  // RELIABLE, AND THAT IS NOT A PREFERENCE. The rate argues for best effort --
  // 10 Hz, newest supersedes -- but aries_load_cells subscribes with
  // `create_subscription(Int32MultiArray, ..., 10)`, which is rclpy's DEFAULT
  // QoS, which is RELIABLE. A BEST_EFFORT publisher and a RELIABLE subscriber
  // are an incompatible pair and DDS makes no match at all: both sides list the
  // topic, `ros2 topic info` shows a publisher and a subscriber, and not one
  // message is ever delivered. That is the same trap /gripper/state documents
  // above, from the other direction. Change this only together with the QoS on
  // the host subscription.
  //
  // FOURTH PUBLISHER. RMW_UXRCE_MAX_PUBLISHERS is 4 by default in
  // micro_ros_platformio and colcon.meta at the project root raises it to 5 --
  // without that, this init fails, create_entities() bails, and the board sits
  // in WAITING_AGENT with USB enumerated, looking exactly like an unflashed
  // Teensy. Check that file before adding a fifth.
  //
  // No leading slash under an empty namespace, as stacklight_subscription is,
  // which resolves to /load_cells/raw -- the name load_cells.yaml names.
  if (RCL_RET_OK != rclc_publisher_init_default(
                        &load_cells_raw_pub, &node,
                        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32MultiArray),
                        "load_cells/raw"))
    return false;
  entities_stage = 15;

  // NINE HANDLES, one per subscription -- pump/purge made it nine.
  // rclc_executor_add_subscription past the end of this array returns an error
  // nothing here checks, and the extra subscription is then simply never spun:
  // its topic exists, matches, and silently delivers nothing.
  if (RCL_RET_OK != rclc_executor_init(&executor, &support.context, 9, &allocator))
    return false;
  entities_stage = 16;

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
  rclc_executor_add_subscription(&executor, &pump_cmd_sub, &pump_cmd_msg,
                                 &pump_cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &pump_purge_sub, &pump_purge_msg,
                                 &pump_purge_callback, ON_NEW_DATA);
  return true;
}

static void destroy_entities()
{
  if (entities_stage == 0)
    return;

  rmw_context_t *rmw_ctx = rcl_context_get_rmw_context(&support.context);
  (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_ctx, 0);

  // Reverse creation order.
  if (entities_stage >= 16)
    rclc_executor_fini(&executor);
  if (entities_stage >= 15)
    IGNORE_RC(rcl_publisher_fini(&load_cells_raw_pub, &node));
  if (entities_stage >= 14)
    IGNORE_RC(rcl_publisher_fini(&pin_scan_pub, &node));
  if (entities_stage >= 13)
    IGNORE_RC(rcl_publisher_fini(&drill_limits_pub, &node));
  if (entities_stage >= 12)
    IGNORE_RC(rcl_subscription_fini(&pump_purge_sub, &node));
  if (entities_stage >= 11)
    IGNORE_RC(rcl_subscription_fini(&pump_cmd_sub, &node));
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
  // The pump is a timed one-shot like the bin, so it has no command stream to
  // fall silent -- if it is mid-dose when the link drops, this is what ends it.
  pump.stop_motor();
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
  const bool top_closed = switch_feed_top.is_at_stop();
  const bool bottom_closed = switch_feed_bottom.is_at_stop();

  // feed_applied is still the PREVIOUS pass's value here, which is exactly
  // what is wanted: the sign that was actually driving when the switch closed.
  const int8_t driving = sign_of(feed_applied);
  const uint32_t now_ms = millis();

  static bool top_episode = false;
  static bool bottom_episode = false;
  static uint32_t top_open_since = 0;
  static uint32_t bottom_open_since = 0;

  if (top_closed)
  {
    if (!top_episode)
    {
      top_episode = true;
      if (driving != 0)
        sign_into_top = driving;
    }
  }
  else
  {
    if (top_episode)
    {
      if (top_open_since == 0)
        top_open_since = now_ms;
      else if (now_ms - top_open_since >= kReleaseMs)
        top_episode = false;
    }
    if (!top_episode)
      top_open_since = 0;
  }

  if (bottom_closed)
  {
    if (!bottom_episode)
    {
      bottom_episode = true;
      if (driving != 0)
        sign_into_bottom = driving;
    }
  }
  else
  {
    if (bottom_episode)
    {
      if (bottom_open_since == 0)
        bottom_open_since = now_ms;
      else if (now_ms - bottom_open_since >= kReleaseMs)
        bottom_episode = false;
    }
    if (!bottom_episode)
      bottom_open_since = 0;
  }

  int feed_pwm = feed_cmd_signed;
  const int8_t want = sign_of(feed_pwm);
  if (want != 0 && top_closed && want == sign_into_top)
    feed_pwm = 0;
  if (want != 0 && bottom_closed && want == sign_into_bottom)
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
      !pump.usable() ||
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

  // Diagnostic pin scan -- see kScanPins. INPUT_PULLUP only, never driven.
  for (uint8_t i = 0; i < kScanCount; ++i)
    pinMode(kScanPins[i], INPUT_PULLUP);
  auger.init_motor();
  feed_motor.init_motor();
  bin_actuator.init_motor();
  pump.init_motor();
  stalig.init_light();
  gservo.init();
  lid_servo.init();

  for (uint8_t i = 0; i < LOAD_CELL_COUNT; ++i)
    load_cells[i].init();

  // Point the message at its buffer ONCE, here, rather than inside
  // create_entities(). The message and the buffer are both globals that
  // outlive every agent session, and create_entities() runs again on every
  // reconnect -- so doing it here means a reconnect cannot leave the message
  // pointing at nothing. See the note on load_cells_raw_buf for what an
  // unassigned sequence actually does (it publishes, emptily, in silence).
  load_cells_raw_msg.data.data = load_cells_raw_buf;
  load_cells_raw_msg.data.size = LOAD_CELL_COUNT;
  load_cells_raw_msg.data.capacity = LOAD_CELL_COUNT;
  // No dimensions. The layout block is optional in Int32MultiArray and
  // aries_load_cells reads msg.data alone, so an empty dim sequence keeps three
  // rosidl strings off a board that would have to allocate them. It still has
  // to be explicitly empty rather than merely zeroed by accident, because
  // serialisation walks it.
  load_cells_raw_msg.layout.dim.data = NULL;
  load_cells_raw_msg.layout.dim.size = 0;
  load_cells_raw_msg.layout.dim.capacity = 0;
  load_cells_raw_msg.layout.data_offset = 0;

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

  // Both regardless of agent state. gservo.update() is the slew step, and it
  // has to keep running or the jaws stop partway through a move the instant the
  // link hiccups.
  //
  // lid_servo.update() is a WATCHDOG, not a slew, and it matters most in
  // exactly the states where the agent is gone: it is what stops a lid that was
  // turning when the link dropped. Gate it on AGENT_CONNECTED and the one case
  // it exists for is the one case it would not cover.
  gservo.update();
  lid_servo.update(LID_COMMAND_TIMEOUT_MS);

  // Also regardless of agent state, for two reasons. The counts are wanted the
  // instant the link comes up rather than 100 ms later, and -- the one that
  // matters -- polling is how a cell ever gets to say it exists at all. Gate
  // this on the agent and a board that has been sitting unconnected reports
  // three rails for its first cycle after every reconnect.
  //
  // Each call is one digitalRead per cell unless a conversion is actually
  // waiting, which is 10 times a second per cell.
  for (uint8_t i = 0; i < LOAD_CELL_COUNT; ++i)
  {
    if (load_cells[i].update())
      load_cells_present = true;
  }

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

    // On change, so an operator watching the topic sees the exact moment a
    // switch closes; plus a 2 Hz heartbeat so a late subscriber learns the
    // current state without having to wait for the carriage to move, and so
    // "the switches say nothing" is distinguishable from "the board is gone".
    {
      static uint8_t last_limits = 0xFF;
      static uint32_t last_limits_pub_ms = 0;
      const uint8_t bits = drill_limits_state();
      if (bits != last_limits || millis() - last_limits_pub_ms >= 500)
      {
        last_limits = bits;
        last_limits_pub_ms = millis();
        drill_limits_msg.data = bits;
        IGNORE_RC(rcl_publish(&drill_limits_pub, &drill_limits_msg, NULL));
      }
    }

    // Every cycle, unconditionally -- no publish-on-change and no dead band.
    // This is a continuous measurement of a mass being poured, and the host is
    // averaging it (filter_samples in load_cells.yaml) and timing it
    // (timeout_s), both of which need the samples to keep arriving. `valid` on
    // the bin's number is the host's job and it never withholds the number.
    if (load_cells_present)
    {
      static uint32_t last_cells_pub_ms = 0;
      const uint32_t cells_now = millis();
      if (cells_now - last_cells_pub_ms >= LOAD_CELL_PERIOD_MS)
      {
        last_cells_pub_ms = cells_now;
        for (uint8_t i = 0; i < LOAD_CELL_COUNT; ++i)
          load_cells_raw_buf[i] = load_cells[i].reported(cells_now);
        IGNORE_RC(rcl_publish(&load_cells_raw_pub, &load_cells_raw_msg, NULL));
      }
    }

    {
      static uint64_t last_scan = 0xFFFFFFFFFFFFFFFFULL;
      static uint32_t last_scan_pub_ms = 0;
      const uint64_t scan = pin_scan_state();
      if (scan != last_scan || millis() - last_scan_pub_ms >= 500)
      {
        last_scan = scan;
        last_scan_pub_ms = millis();
        pin_scan_msg.data = scan;
        IGNORE_RC(rcl_publish(&pin_scan_pub, &pin_scan_msg, NULL));
      }
    }
    break;
  }

  case AGENT_DISCONNECTED:
    // Everything off before the teardown. The link is gone, so no stop command
    // can arrive; leaving the auger turning until the watchdog notices would be
    // half a second of a cutting tool nobody is talking to. The lid is stopped
    // here for the same reason and ahead of its own 500 ms timeout -- a known
    // disconnect is not something to wait out.
    stop_all_motors();
    lid_servo.stop();
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
  // A SPEED in -1..1, not a position. 0.0 stops; the sign is the direction, and
  // which sign opens the lid is chosen on the host (invert_lid in
  // joystick.yaml) because it is a fact about how the servo is mounted.
  // set_speed clamps, so an out-of-range value is full speed rather than a
  // pulse width outside what the servo will accept.
  //
  // Every message here also re-arms the watchdog in LidServo::update(). A host
  // that stops publishing therefore stops the lid within
  // LID_COMMAND_TIMEOUT_MS, whether it meant to or not.
  const std_msgs__msg__Float32 *msg = (const std_msgs__msg__Float32 *)msin;
  lid_servo.set_speed(msg->data);
}

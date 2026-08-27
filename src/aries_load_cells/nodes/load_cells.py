#!/usr/bin/env python3
"""Three load cells on the drill/science Teensy, published as weights in kilograms.

    sand_box         front-left deck box, the sand sample
    stone_box        the box behind it, also on the left, the stone sample
    drill_container  the drill's sample bin -- see THE BIN IS ONLY ON THE
                     CELLS WHEN IT IS PARKED, below

WHERE THE NUMBERS COME FROM. The cells hang off the same Teensy that already
runs the gripper servo and the stack light, so they arrive over the micro-ROS
link that aries_hardware.launch.py already brings up: no second board, no
second agent, no new entry in devices.yaml.

The firmware publishes RAW COUNTS and this node converts them. That split is
deliberate. A load cell's scale and zero are properties of the cell, its
amplifier and whatever it is bolted to, they are found by hanging known masses
off the rover, and they change whenever a box is unbolted or a cell is swapped.
Kept in load_cells.yaml they are edited and reloaded on the next launch, with
no rebuild (the workspace is --symlink-install) and no reflash. Kept in the
sketch, every recalibration is a trip to the Arduino IDE with the rover open.

THE FIRMWARE CONTRACT -- one publisher:

    load_cells/raw   std_msgs/Int32MultiArray, one element per cell, in the
                     order given by `cells` below. Declared with no leading
                     slash under an empty namespace, as the stack light's
                     subscription is, so it resolves to /load_cells/raw.

ONE publisher and not three. The firmware polls all three amplifiers on one
pass of its loop and they go out as one set; splitting them across three topics
would make this node re-pair samples that arrived together, and get it wrong at
exactly the moment the link is slow. The entity budget agrees -- that board is
at four publishers against the five its colcon.meta allows. If the firmware
ever ends up with three separate std_msgs/Int32 publishers anyway, set
`raw_topics` and this node reads those instead.

RELIABLE, BOTH ENDS. The raw subscription below is created with default rclpy
QoS, which is reliable, and the firmware publisher is reliable to match. A
best-effort publisher against a reliable subscriber is an incompatible pair:
DDS makes no match at all, both sides list the topic, `ros2 topic info` shows
one of each, and not one message is delivered. Change one end only with the
other.

A CELL THAT IS NOT REPORTING ARRIVES AS raw_min, NOT AS ZERO. The firmware
sends -8388608 for an amplifier that is unplugged, dead, or whose last
conversion is over 500 ms old, so it lands on the rail fault below and comes
out as NaN. It never sends zero to mean "no reading", because zero is what an
empty box reads.

Nothing is faked. With `source: auto` and no firmware talking, the topics are
all advertised and simply carry nothing, and the node says so once a second in
the log. `source: mock` is the explicit way to get numbers out of this without
a Teensy, for checking the topic layout or driving a display. There is no path
where the rover invents a weight and puts it on /load_cells/sand_box/weight
looking exactly like a real one.

EVERY CELL READS CONTINUOUSLY. All three weights are published at
publish_rate_hz whatever the rover is doing -- while a box is being filled,
while the bin travels, while the auger cuts. Nothing here waits for the rover
to hold still, and a cell with no reading publishes NaN at the same steady rate
rather than falling silent, because a topic that stops looks exactly like a
node that died. The bin's `valid` below LABELS its number; it never withholds
it.

THE BIN IS ONLY ON THE CELLS WHEN IT IS PARKED. The drill's sample bin rides
its rails between q = 0, parked forward of the mast, and q = -0.1304, back
under the auger (drill.xacro). The cell sits under the PARKED position, so the
bin rests on it at one end of that stroke and at the other end it is somewhere
else entirely, with nothing on the cell.

That is a trap, because a cell with nothing on it does not read "no data". It
reads ZERO -- which is exactly what a parked-and-empty bin reads. Publish it
unqualified and the operator watching the number sees the bin empty itself the
moment it slides under the auger to collect, and fill back up on the way out.
So `drill_container/valid` says whether the number means anything, and
`drill_container/weight_held` carries the last reading taken while it did.
Watch weight_held if you want to know what is in the bin; watch weight if you
want to know what the cell is reading.

WHERE THE BIN IS, is dead-reckoned from the rate commands on
/aries/drill_container_joint/cmd_vel, the same model drill_joystick.py runs its
limit switches on. It has to be: the bin's actuator is a DC motor with no
encoder, and on the real rover publish_wheel_joints.py publishes every drill
joint at a constant 0.0 so MoveIt has a complete robot state. Gate on THAT and
the answer is "parked" forever, including while the bin is under the auger --
the check would pass at exactly the moment it is wrong.

Dead reckoning drifts, and the honest fix is the bin's own forward end switch:
wire it to a std_msgs/Bool and name it in `parked_switch_topic`, and it takes
over completely. Until then two things bound the drift. Running the bin into
the parked end re-syncs the estimate the way homing against a hard stop does,
because the integrator clamps there. And `settle_s` keeps the reading
disqualified for a moment after the bin stops and after the auger stops, so a
bin still rocking on its rails, or a frame still ringing from the cut, is not
read as a sample mass.
"""

import json
import math
import random
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64, Int32, Int32MultiArray, String
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState


# An HX711 is a 24-bit signed converter. These are its rails: a reading that
# reaches one is a cell that is unplugged, wired backwards, or crushed, and
# converting it to kilograms would produce a confident enormous number.
HX711_MIN = -(1 << 23)
HX711_MAX = (1 << 23) - 1


class Cell:
    """One load cell: its calibration, its filter, and its last reading."""

    def __init__(self, name, scale, offset, invert, filter_samples,
                 raw_min, raw_max):
        self.name = name
        # Counts per kilogram. Never zero -- see _load_cells().
        self.scale = scale
        # The raw count with nothing on the cell. `tare` moves this.
        self.offset = offset
        self.sign = -1.0 if invert else 1.0
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.samples = deque(maxlen=max(1, filter_samples))
        self.raw = None
        self.stamp = None
        self.fault = None

    def update(self, raw, stamp):
        self.raw = int(raw)
        self.stamp = stamp
        if self.raw <= self.raw_min or self.raw >= self.raw_max:
            # Do not let a rail into the filter: it would poison the average
            # for filter_samples readings after the cell recovers.
            self.fault = f"raw {self.raw} at the converter's rail (cell unplugged?)"
            return
        self.fault = None
        self.samples.append(self.raw)

    def filtered_raw(self):
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)

    def weight(self):
        """Kilograms, or NaN when there is nothing trustworthy to report.

        NaN rather than 0.0, and rather than silence. Zero is a real weight and
        an empty box reads it; a topic that stops publishing is indistinguish-
        able from a node that died. NaN says "no reading" out loud, and every
        `if w > threshold` a consumer writes is False against it, which is the
        direction a scale should fail in.
        """
        if self.fault is not None:
            return float("nan")
        raw = self.filtered_raw()
        if raw is None:
            return float("nan")
        return self.sign * (raw - self.offset) / self.scale

    def tare(self):
        """Zero the cell at whatever is on it now. Returns the new offset."""
        raw = self.filtered_raw()
        if raw is None:
            return None
        self.offset = raw
        return self.offset


class LoadCells(Node):
    def __init__(self):
        super().__init__("load_cells")

        # --- what the cells are -------------------------------------------
        # Order matters: it is the element order of the firmware's
        # Int32MultiArray, and the firmware has no names to check itself
        # against. Change one and the sand box starts reporting the stone.
        self.declare_parameter("cells", ["sand_box", "stone_box", "drill_container"])
        self.declare_parameter("topic_namespace", "load_cells")

        # --- where the counts come from -----------------------------------
        # auto    use the firmware if it is there, otherwise publish nothing
        #         and say so. The rover default.
        # microros the same, without the "otherwise" -- warns harder.
        # mock    synthetic counts, for bringing this up with no Teensy.
        self.declare_parameter("source", "auto")
        self.declare_parameter("raw_topic", "/load_cells/raw")
        # The three-publisher alternative. Empty means "use raw_topic".
        self.declare_parameter("raw_topics", [""])
        # Counts older than this are not a weight any more. The firmware is
        # expected to publish continuously; an HX711 chain runs at 10 SPS by
        # default, so this is several missed conversions and not a hiccup.
        self.declare_parameter("timeout_s", 2.0)

        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("status_rate_hz", 2.0)

        g = self.get_parameter
        self.cell_names = [c for c in g("cells").value if c]
        if not self.cell_names:
            raise ValueError("load_cells: `cells` is empty, there is nothing to publish")

        ns = str(g("topic_namespace").value).strip("/")
        self.ns = f"/{ns}" if ns else ""
        self.source = str(g("source").value)
        if self.source not in ("auto", "microros", "mock"):
            raise ValueError(
                f"load_cells: source '{self.source}' is not one of auto, microros, mock")
        self.timeout_s = float(g("timeout_s").value)

        # --- per-cell calibration ------------------------------------------
        # Nested, one block per cell, because there are four numbers per cell
        # and a flat "name:value" list of them is unreadable. ROS 2 flattens
        # `cell: {sand_box: {scale: ...}}` to `cell.sand_box.scale`, which is a
        # declarable parameter name -- but only once the names are known, so
        # `cells` above has to be read first.
        self.cells = {}
        for name in self.cell_names:
            self.declare_parameter(f"cell.{name}.scale", 1.0)
            self.declare_parameter(f"cell.{name}.offset", 0.0)
            self.declare_parameter(f"cell.{name}.invert", False)
            self.declare_parameter(f"cell.{name}.filter_samples", 8)
            self.declare_parameter(f"cell.{name}.raw_min", float(HX711_MIN))
            self.declare_parameter(f"cell.{name}.raw_max", float(HX711_MAX))
            scale = float(g(f"cell.{name}.scale").value)
            if scale == 0.0:
                # Not a warning. Dividing by it gives inf on every reading, and
                # an uncalibrated cell reporting inf kilograms is worse than a
                # node that refuses to start with the reason on the console.
                raise ValueError(
                    f"load_cells: cell '{name}' has scale 0 -- it has never been "
                    f"calibrated. Hang a known mass on it and set "
                    f"cell.{name}.scale in load_cells.yaml to counts per kg.")
            if scale == 1.0:
                # An HX711 at gain 128 on a cell of this size runs to a few
                # hundred thousand counts per kilogram. Exactly 1.0 is the
                # value shipped in load_cells.yaml as a placeholder, so it
                # means nobody has hung a mass on this cell yet.
                self.get_logger().warn(
                    f"cell '{name}' still has the placeholder scale of 1.0 counts/kg - "
                    f"its weight topic is counts, not kilograms. Calibrate it: hang a "
                    f"known mass, read {self.ns}/{name}/raw, and set cell.{name}.scale.")
            self.cells[name] = Cell(
                name=name,
                scale=scale,
                offset=float(g(f"cell.{name}.offset").value),
                invert=bool(g(f"cell.{name}.invert").value),
                filter_samples=int(g(f"cell.{name}.filter_samples").value),
                raw_min=float(g(f"cell.{name}.raw_min").value),
                raw_max=float(g(f"cell.{name}.raw_max").value),
            )

        # --- the drill bin's gate ------------------------------------------
        # Which cell is under the bin. Empty disables all of this, for a rover
        # whose drill has been taken off.
        self.declare_parameter("container_cell", "drill_container")
        self.declare_parameter("container_rate_topic",
                               "/aries/drill_container_joint/cmd_vel")
        # Keep in step with drill.xacro: q = 0 is parked forward of the mast,
        # q = -0.1304 is back under the auger.
        self.declare_parameter("container_parked_position", 0.0)
        self.declare_parameter("container_lower", -0.1304)
        self.declare_parameter("container_upper", 0.0)
        # How close to the parked end still counts as on the cell. 5 mm: the
        # bin's feet are wider than that, so it is genuinely still seated.
        self.declare_parameter("container_parked_tolerance", 0.005)
        # Where the bin is assumed to be at startup. Parked is the stowed
        # position and the actuator does not backdrive, so this is where it was
        # left -- but it IS an assumption, and the end switch is what retires
        # it. Running the bin into the parked end also re-syncs the estimate.
        self.declare_parameter("container_initial_position", 0.0)
        # A std_msgs/Bool that is true when the bin is on its forward end
        # switch. Authoritative when wired: it overrides dead reckoning and
        # re-datums the estimate. Nothing publishes one yet.
        self.declare_parameter("parked_switch_topic", "")
        self.declare_parameter("parked_switch_active_high", True)
        # A measured position, if a real drill driver ever publishes one. NOT
        # the default: publish_wheel_joints.py puts every drill joint at a
        # constant 0.0 for MoveIt, which reads as "parked" forever.
        self.declare_parameter("container_joint_states_topic", "")
        self.declare_parameter("container_joint", "drill_container_joint")
        # Quiet time after the bin stops, and after the auger stops, before a
        # reading counts. A bin still rocking on its rails is not a sample mass.
        self.declare_parameter("container_settle_s", 1.5)
        self.declare_parameter("auger_rate_topic", "/aries/drill_bit_joint/cmd_vel")
        self.declare_parameter("rate_epsilon", 1e-6)
        # Rate topics go silent when idle -- drill_joystick.py sends its zeros
        # and stops. Silence is not a moving axis.
        self.declare_parameter("rate_timeout_s", 1.0)

        self.container_cell = str(g("container_cell").value)
        if self.container_cell and self.container_cell not in self.cells:
            raise ValueError(
                f"load_cells: container_cell '{self.container_cell}' is not in "
                f"`cells` ({', '.join(self.cell_names)})")
        self.container_parked = float(g("container_parked_position").value)
        self.container_lower = float(g("container_lower").value)
        self.container_upper = float(g("container_upper").value)
        self.container_tolerance = float(g("container_parked_tolerance").value)
        self.container_q = float(g("container_initial_position").value)
        self.container_settle_s = float(g("container_settle_s").value)
        self.rate_epsilon = float(g("rate_epsilon").value)
        self.rate_timeout_s = float(g("rate_timeout_s").value)
        self.parked_switch_active_high = bool(g("parked_switch_active_high").value)
        self.container_joint = str(g("container_joint").value)

        self.container_rate = 0.0
        self.container_rate_at = None
        self.auger_rate = 0.0
        self.auger_rate_at = None
        self.container_moved_at = None
        self.auger_ran_at = None
        self.parked_switch = None
        self.container_measured = None
        self.weight_held = {}
        self.held_at = {}
        self.last_integrate_at = None

        # --- mock -----------------------------------------------------------
        # Kilograms each mock cell should read. Same order as `cells`.
        self.declare_parameter("mock_kg", [0.35, 0.72, 0.12])
        # Noise on the mock signal, so the filter has something to do. In
        # KILOGRAMS, not counts: counts are only a mass once `scale` is
        # calibrated, and while it is still the 1.0 placeholder a noise
        # expressed in counts comes out as tens of kilograms of jitter.
        self.declare_parameter("mock_noise_kg", 0.002)
        self.mock_kg = [float(v) for v in g("mock_kg").value]
        if self.source == "mock" and len(self.mock_kg) != len(self.cell_names):
            raise ValueError(
                f"load_cells: mock_kg has {len(self.mock_kg)} values for "
                f"{len(self.cell_names)} cells; they pair up by position")
        self.mock_noise = float(g("mock_noise_kg").value)

        # --- publishers -----------------------------------------------------
        self.weight_pubs = {}
        self.raw_pubs = {}
        for name in self.cell_names:
            self.weight_pubs[name] = self.create_publisher(
                Float32, f"{self.ns}/{name}/weight", 10)
            # The raw count is republished for one reason: calibration. Hang a
            # known mass, watch this, do the arithmetic. Without it the only
            # way to see counts is a serial monitor on the Teensy, which means
            # unplugging it from the agent and taking the gripper down.
            self.raw_pubs[name] = self.create_publisher(
                Int32, f"{self.ns}/{name}/raw", 10)

        self.valid_pub = None
        self.held_pub = None
        if self.container_cell:
            self.valid_pub = self.create_publisher(
                Bool, f"{self.ns}/{self.container_cell}/valid", 10)
            self.held_pub = self.create_publisher(
                Float32, f"{self.ns}/{self.container_cell}/weight_held", 10)

        # One topic to echo when you want all of it at once, JSON in a String,
        # the way /aries_drive/status already does it.
        self.status_pub = self.create_publisher(String, f"{self.ns}/status", 10)

        # "No counts yet" is reported once in full and then only as a
        # reminder; see _publish_cb. Cleared when counts start, so a board that
        # comes up late says so instead of leaving a complaint as the last word.
        self._warned_no_counts = False
        self.no_counts_reminder_s = 120.0

        # --- subscriptions ---------------------------------------------------
        self.raw_topic = str(g("raw_topic").value)
        self.raw_topics = [t for t in g("raw_topics").value if t]
        if self.source != "mock":
            if self.raw_topics:
                if len(self.raw_topics) != len(self.cell_names):
                    raise ValueError(
                        f"load_cells: raw_topics has {len(self.raw_topics)} entries "
                        f"for {len(self.cell_names)} cells; they pair up by position")
                for name, topic in zip(self.cell_names, self.raw_topics):
                    self.create_subscription(
                        Int32, topic,
                        (lambda n: lambda m: self._raw_one(n, m.data))(name), 10)
            else:
                self.create_subscription(
                    Int32MultiArray, self.raw_topic, self._raw_array_cb, 10)

        if self.container_cell:
            self.create_subscription(
                Float64, str(g("container_rate_topic").value),
                self._container_rate_cb, 10)
            auger_topic = str(g("auger_rate_topic").value)
            if auger_topic:
                self.create_subscription(
                    Float64, auger_topic, self._auger_rate_cb, 10)
            switch_topic = str(g("parked_switch_topic").value)
            if switch_topic:
                self.create_subscription(
                    Bool, switch_topic, self._parked_switch_cb, 10)
            js_topic = str(g("container_joint_states_topic").value)
            if js_topic:
                self.create_subscription(
                    JointState, js_topic, self._joint_state_cb, 10)

        # --- services ---------------------------------------------------------
        # Zeroing a cell in the field otherwise means editing YAML and
        # relaunching the stack, which on a rover mid-task is not a thing
        # anyone does. The new offset is LOGGED, not written back: paste it
        # into load_cells.yaml to keep it across a restart.
        self.create_service(Trigger, f"{self.ns}/tare", self._tare_all_cb)
        for name in self.cell_names:
            self.create_service(
                Trigger, f"{self.ns}/{name}/tare",
                (lambda n: lambda req, res: self._tare_one_cb(n, req, res))(name))

        self.create_timer(1.0 / max(float(g("publish_rate_hz").value), 0.1),
                          self._publish_cb)
        self.create_timer(1.0 / max(float(g("status_rate_hz").value), 0.1),
                          self._status_cb)
        if self.source == "mock":
            self.create_timer(0.1, self._mock_cb)

        where = (f"mock counts, NO HARDWARE" if self.source == "mock"
                 else (", ".join(self.raw_topics) if self.raw_topics else self.raw_topic))
        self.get_logger().info(
            f"Load cells up on {self.ns}/<cell>/weight for "
            f"{', '.join(self.cell_names)}; counts from {where}")
        if self.source == "mock":
            self.get_logger().warn(
                "MOCK COUNTS - nothing here was measured. Note that the mock "
                "quantises to whole converter counts, as a real ADC does, so "
                "while `scale` is still the 1.0 placeholder every mock weight "
                "lands on a whole kilogram. Calibrate a cell to see fractions.")
        if self.container_cell:
            self.get_logger().info(
                f"{self.container_cell} is only on its cell when the bin is parked "
                f"at q={self.container_parked:+.4f} +/-{self.container_tolerance*1000:.0f} mm; "
                f"watch {self.ns}/{self.container_cell}/valid")
            if not str(g("parked_switch_topic").value):
                self.get_logger().info(
                    "Bin position is DEAD-RECKONED from the rate commands - no end "
                    "switch is wired. Fill in parked_switch_topic when one is.")

    # -- clock ---------------------------------------------------------------
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -- counts in -----------------------------------------------------------
    def _raw_array_cb(self, msg):
        values = list(msg.data)
        if len(values) != len(self.cell_names):
            self.get_logger().warn(
                f"{self.raw_topic} carried {len(values)} counts for "
                f"{len(self.cell_names)} cells; ignoring the message. The array's "
                f"element order is `cells`: {', '.join(self.cell_names)}",
                throttle_duration_sec=5.0)
            return
        now = self._now()
        for name, raw in zip(self.cell_names, values):
            self.cells[name].update(raw, now)

    def _raw_one(self, name, raw):
        self.cells[name].update(raw, self._now())

    def _mock_cb(self):
        now = self._now()
        for name, kg in zip(self.cell_names, self.mock_kg):
            cell = self.cells[name]
            noisy = kg + random.uniform(-self.mock_noise, self.mock_noise)
            counts = cell.offset + cell.sign * noisy * cell.scale
            cell.update(int(round(counts)), now)

    # -- where the bin is ------------------------------------------------------
    def _container_rate_cb(self, msg):
        self.container_rate = float(msg.data)
        self.container_rate_at = self._now()
        if abs(self.container_rate) > self.rate_epsilon:
            self.container_moved_at = self.container_rate_at

    def _auger_rate_cb(self, msg):
        self.auger_rate = float(msg.data)
        self.auger_rate_at = self._now()
        if abs(self.auger_rate) > self.rate_epsilon:
            self.auger_ran_at = self.auger_rate_at

    def _parked_switch_cb(self, msg):
        pressed = bool(msg.data) == self.parked_switch_active_high
        self.parked_switch = pressed
        if pressed:
            # A switch that has tripped is a measurement, and it is the only
            # one this axis produces. Re-datum on it: that is what retires the
            # dead reckoning's accumulated error.
            self.container_q = self.container_parked

    def _joint_state_cb(self, msg):
        for name, position in zip(msg.name, msg.position):
            if name == self.container_joint:
                self.container_measured = float(position)
                self.container_q = self.container_measured
                return

    def _integrate_container(self):
        """Advance the dead-reckoned bin position by the commanded rate.

        Clamped to the stroke, which is also how the estimate re-datums: drive
        the bin at the parked end for longer than the stroke takes and the
        clamp puts it exactly there, however much drift had built up. Homing
        against a stop, with the stop in software.
        """
        now = self._now()
        dt = 0.0 if self.last_integrate_at is None else now - self.last_integrate_at
        self.last_integrate_at = now
        if self.container_measured is not None or self.parked_switch:
            return
        rate = self.container_rate
        if (self.container_rate_at is None
                or (now - self.container_rate_at) > self.rate_timeout_s):
            rate = 0.0
        if dt <= 0.0 or abs(rate) <= self.rate_epsilon:
            return
        self.container_q = max(self.container_lower,
                               min(self.container_upper,
                                   self.container_q + rate * dt))

    def _container_state(self):
        """(valid, reason). Whether the bin's cell is reading the bin."""
        now = self._now()
        if self.parked_switch is not None:
            parked = self.parked_switch
        else:
            parked = abs(self.container_q - self.container_parked) <= self.container_tolerance
        if not parked:
            return False, f"bin at q={self.container_q:+.4f}, off the cell"

        quiet_since = [t for t in (self.container_moved_at, self.auger_ran_at)
                       if t is not None]
        if quiet_since:
            since = now - max(quiet_since)
            if since < self.container_settle_s:
                return False, f"settling, {self.container_settle_s - since:.1f}s to go"

        cell = self.cells[self.container_cell]
        if cell.fault is not None:
            return False, cell.fault
        if not self._fresh(cell):
            return False, "no counts"
        return True, "parked and settled"

    def _fresh(self, cell):
        return (cell.stamp is not None
                and (self._now() - cell.stamp) <= self.timeout_s)

    # -- out -------------------------------------------------------------------
    def _publish_cb(self):
        if self.container_cell:
            self._integrate_container()

        # Once per TICK, not once per cell. All three cells are fed from the
        # same raw topic, so "nothing has ever arrived" is one fact about one
        # topic; reporting it inside the loop below printed it three times.
        #
        # And said in full ONCE, then rarely. "The board is not attached" is a
        # static condition, not an event: on a bench run it is true for the
        # whole session, and at 10 s it printed ~360 times an hour into a
        # console someone is trying to read the rest of the stack in.
        if self.source != "mock":
            silent = [n for n in self.cell_names if self.cells[n].stamp is None]
            if silent and not self._warned_no_counts:
                self._warned_no_counts = True
                self.get_logger().warn(
                    f"No counts on {self.raw_topic} yet - is the Teensy's "
                    f"micro-ROS agent up, and does the firmware publish it? "
                    f"Run with source:=mock to exercise this without hardware. "
                    f"(Repeats every {self.no_counts_reminder_s:.0f}s while it "
                    f"lasts; the weights publish NaN meanwhile.)")
            elif silent:
                self.get_logger().warn(
                    f"still no counts on {self.raw_topic} "
                    f"({len(silent)}/{len(self.cell_names)} cells)",
                    throttle_duration_sec=self.no_counts_reminder_s)
            elif self._warned_no_counts:
                # Say so when it comes back, or the last thing in the log is a
                # complaint about hardware that has since started working.
                self._warned_no_counts = False
                self.get_logger().info(f"counts arriving on {self.raw_topic}")

        for name in self.cell_names:
            cell = self.cells[name]
            fresh = self._fresh(cell)
            # EVERY CELL, EVERY TICK, whatever the rover is doing. All three
            # weights are live readings and none of them is gated on the rover
            # holding still: watch a box fill while it is being filled, watch
            # the bin while the drill runs. The bin's `valid` below LABELS its
            # number, it does not withhold it.
            #
            # A stale or faulted cell publishes NaN rather than falling silent.
            # Silence is indistinguishable from a dead node, and an operator
            # display fed by this needs a value at a steady rate to show "no
            # reading" instead of freezing on the last good one. NaN is that
            # value, and it is not 0.0, which is a real weight an empty box has.
            kg = cell.weight() if fresh else float("nan")
            self.weight_pubs[name].publish(Float32(data=float(kg)))
            if fresh:
                # No NaN in an Int32, so raw is the one topic that does go
                # quiet. It is a calibration aid, not the reading.
                self.raw_pubs[name].publish(Int32(data=int(cell.raw)))

        if not self.container_cell:
            return
        valid, _ = self._container_state()
        self.valid_pub.publish(Bool(data=valid))
        if valid:
            kg = self.cells[self.container_cell].weight()
            if not math.isnan(kg):
                self.weight_held[self.container_cell] = kg
                self.held_at[self.container_cell] = self._now()
        held = self.weight_held.get(self.container_cell)
        self.held_pub.publish(Float32(data=float(held if held is not None else float("nan"))))

    def _status_cb(self):
        now = self._now()
        cells = {}
        for name in self.cell_names:
            cell = self.cells[name]
            fresh = self._fresh(cell)
            kg = cell.weight() if fresh else float("nan")
            cells[name] = {
                "kg": None if math.isnan(kg) else round(kg, 4),
                "g": None if math.isnan(kg) else round(kg * 1000.0, 1),
                "raw": cell.raw,
                "fresh": fresh,
                "fault": cell.fault,
            }
        status = {"source": self.source, "cells": cells}
        if self.container_cell:
            valid, reason = self._container_state()
            held = self.weight_held.get(self.container_cell)
            status["drill_container"] = {
                "valid": valid,
                "reason": reason,
                "bin_q": round(self.container_q, 4),
                "bin_position": ("measured" if self.container_measured is not None
                                 else "end switch" if self.parked_switch is not None
                                 else "dead reckoned"),
                "held_kg": None if held is None else round(held, 4),
                "held_age_s": (None if self.held_at.get(self.container_cell) is None
                               else round(now - self.held_at[self.container_cell], 1)),
            }
        self.status_pub.publish(String(data=json.dumps(status)))

    # -- tare --------------------------------------------------------------------
    def _tare_one_cb(self, name, request, response):
        offset = self.cells[name].tare()
        if offset is None:
            response.success = False
            response.message = f"{name}: no counts to tare against"
        else:
            response.success = True
            response.message = f"{name}: offset now {offset:.1f} counts"
            self.get_logger().info(
                f"Tared {name}: set cell.{name}.offset: {offset:.1f} in "
                f"load_cells.yaml to keep this across a restart")
        return response

    def _tare_all_cb(self, request, response):
        done, failed = [], []
        for name in self.cell_names:
            offset = self.cells[name].tare()
            (failed if offset is None else done).append(
                name if offset is None else f"{name}={offset:.1f}")
        response.success = not failed
        response.message = ("tared " + ", ".join(done)) if done else ""
        if failed:
            response.message += f"; no counts for {', '.join(failed)}"
        if done:
            self.get_logger().info(
                "Tared " + ", ".join(done)
                + " - copy these into load_cells.yaml as cell.<name>.offset")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LoadCells()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

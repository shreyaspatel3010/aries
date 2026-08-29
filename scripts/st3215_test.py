#!/usr/bin/python3
"""Bench test for a Feetech ST3215 bus servo on a USB bus-servo adapter.

Sweeps the horn 180 deg one way then 180 deg back, in position mode.
Nothing is written to EPROM, so the servo's stored config is untouched.

Runnable with any interpreter: the repo .venv has no pyserial, so if it is
missing the script re-execs itself under /usr/bin/python3 (apt python3-serial).

    python3 scripts/st3215_test.py
    python3 scripts/st3215_test.py --deg 90 --speed 400 --repeat 3
    python3 scripts/st3215_test.py --scan
    python3 scripts/st3215_test.py --where     # position + state, moves nothing
    python3 scripts/st3215_test.py --goto 120 --dir ccw   # long way round
    python3 scripts/st3215_test.py --monitor    # torque off, back-drive by hand

The port is found rather than assumed -- see resolve_port() below -- so none of
these needs --port once the adapter is plugged in.

Note: writing a goal position re-enables torque automatically, so "torque off"
only sticks if no goal is written afterwards.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

try:
    import serial
except ImportError:
    # The repo .venv is built with include-system-site-packages=false, so the
    # apt-installed pyserial in /usr/lib/python3/dist-packages is invisible from
    # inside it. Rather than make the caller remember which interpreter to use,
    # hand ourselves over to the system one. _ST3215_REEXEC stops this looping
    # if that interpreter has no pyserial either.
    import os

    # Do not compare interpreter paths to decide whether to re-exec: a venv's
    # bin/python3 is a symlink to the very same system binary, so the paths
    # match while sys.path does not. The guard variable is the only reliable
    # loop stopper.
    # Only hand off when this file IS the program. Re-execing on a plain import
    # would replace the importing process with this script -- which is exactly
    # what happened: `python3 -c "from st3215_test import Servo"` under the venv
    # silently ran the sweep instead.
    SYS_PY = "/usr/bin/python3"
    if __name__ == "__main__" and not os.environ.get("_ST3215_REEXEC") \
            and os.path.exists(SYS_PY):
        env = dict(os.environ, _ST3215_REEXEC="1")
        os.execve(SYS_PY, [SYS_PY, os.path.abspath(__file__)] + sys.argv[1:], env)
    raise ImportError(
        "pyserial not found. This module needs it; run under /usr/bin/python3 "
        "(apt python3-serial), or install it into the active environment with "
        "'pip install pyserial'.")

PING, READ, WRITE = 0x01, 0x02, 0x03

# SMS/STS control table
R_MODE, R_TORQUE, R_ACC = 33, 40, 41
R_GOAL_POS, R_GOAL_SPEED = 42, 46
R_PRES_POS, R_PRES_LOAD, R_VOLT, R_TEMP, R_MOVING = 56, 60, 62, 63, 66
R_PRES_SPEED, R_PRES_CURR = 58, 69
R_GOAL_TIME, R_LOCK = 44, 55

STEPS_PER_REV = 4096
CURRENT_LSB_MA = 6.5                 # present-current register unit


def sgn(v, bit):
    """STS sign convention: the direction flag sits at a per-register bit and the
    rest is magnitude -- it is NOT two's complement. Speed and current use bit 15;
    present LOAD uses bit 10 (it is a 0-1000 PWM duty, so bit 10 is free)."""
    if v is None:
        return None
    m = 1 << bit
    return -(v & ~m) if v & m else v
POS_MIN, POS_MAX = 50, 4045          # keep clear of the hard 0/4095 limits


class Servo:
    def __init__(self, port, baud, sid, timeout=0.05):
        self.ser = serial.Serial(port, baud, timeout=timeout, write_timeout=0.5)
        self.sid = sid
        self.bad_checksums = 0
        self.last_error = 0
        time.sleep(0.05)
        self.ser.reset_input_buffer()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def _txrx(self, inst, params=b"", expect=0, sid=None):
        sid = self.sid if sid is None else sid
        body = bytes([sid, len(params) + 2, inst]) + bytes(params)
        self.ser.reset_input_buffer()
        self.ser.write(b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF]))
        self.ser.flush()
        hdr = self.ser.read(5)                    # FF FF id len err
        if len(hdr) < 5 or hdr[0] != 0xFF or hdr[1] != 0xFF:
            return None
        rest = self.ser.read(hdr[3] - 1)          # params + checksum
        if len(rest) < hdr[3] - 1 or len(rest) - 1 != expect:
            return None
        # Verify the checksum. Without this a desynchronised packet is returned
        # as a plausible-looking value -- which silently corrupts fast polling
        # (it produced impossible velocities when differencing position).
        if (~(hdr[2] + hdr[3] + hdr[4] + sum(rest[:-1]))) & 0xFF != rest[-1]:
            self.bad_checksums += 1
            return None
        if hdr[4]:
            self.last_error = hdr[4]
        return rest[:-1]

    def ping(self, sid=None):
        return self._txrx(PING, b"", 0, sid) is not None

    def r8(self, addr):
        d = self._txrx(READ, bytes([addr, 1]), 1)
        return None if d is None else d[0]

    def r16(self, addr):
        d = self._txrx(READ, bytes([addr, 2]), 2)
        return None if d is None else d[0] | (d[1] << 8)

    def w8(self, addr, v):
        return self._txrx(WRITE, bytes([addr, v & 0xFF]), 0) is not None

    def w16(self, addr, v):
        v &= 0xFFFF
        return self._txrx(WRITE, bytes([addr, v & 0xFF, (v >> 8) & 0xFF]), 0) is not None

    def torque(self, on):
        """NOTE: torque does not stay off. Writing R_GOAL_POS re-enables it
        automatically (verified: reg 40 flips 0 -> 1 on a goal write), so to
        leave the horn genuinely free you must torque off and then write no
        goal at all."""
        return self.w8(R_TORQUE, 1 if on else 0)

    def telemetry(self):
        c = self.r16(R_PRES_CURR)
        return dict(
            pos=self.r16(R_PRES_POS),
            speed=sgn(self.r16(R_PRES_SPEED), 15),
            load=sgn(self.r16(R_PRES_LOAD), 10),
            curr=sgn(c, 15),
            ma=None if c is None else sgn(c, 15) * CURRENT_LSB_MA,
            volt=self.r8(R_VOLT),
            temp=self.r8(R_TEMP),
            torque=self.r8(R_TORQUE),
        )

    def goto(self, target, timeout=8.0, quiet=False, clamp=True):
        """Command a position (steps) and block until it settles.

        clamp=False reaches the hard 0/4095 stops, which POS_MIN/POS_MAX
        deliberately keep away from. Only --pos does that, and only because a
        datum of exactly 0 is sometimes what you want to assemble against.
        """
        target = max(POS_MIN, min(POS_MAX, int(target))) if clamp else \
            max(0, min(STEPS_PER_REV - 1, int(target)))
        self.w16(R_GOAL_POS, target)
        t0 = time.time()
        pos = None
        while time.time() - t0 < timeout:
            time.sleep(0.1)
            pos = self.r16(R_PRES_POS)
            if pos is None:
                continue
            if not quiet:
                print(f"    pos {pos:>4}  ({pos * 360 / STEPS_PER_REV:6.1f} deg)", end="\r")
            if self.r8(R_MOVING) == 0 and abs(pos - target) < 15:
                break
        if not quiet:
            print(f"    pos {pos:>4}  ({pos * 360 / STEPS_PER_REV:6.1f} deg)   ")
        return pos


def wrap_delta(d):
    """Encoder step difference across the 4095->0 seam, as a signed short move."""
    if d < -STEPS_PER_REV // 2:
        return d + STEPS_PER_REV
    if d > STEPS_PER_REV // 2:
        return d - STEPS_PER_REV
    return d


def plan_travel(start, target, turns=0, direction="short"):
    """Signed encoder steps to move, honouring a forced direction.

    "short" is the old behaviour: whichever way round is nearer, never more
    than 180 deg. "cw"/"ccw" force the sign instead, so a target just BEHIND
    the horn becomes a near-full revolution the long way rather than a small
    step back -- which is the entire point of asking for a direction.

    Direction is defined by the ENCODER: cw = counts increasing, ccw = counts
    decreasing. Whether that looks clockwise to you depends on which end of the
    output shaft you are looking at, so naming the servo's own convention is
    the only description that cannot be wrong.

    With a forced direction the sign of `turns` is redundant and would only
    contradict it, so its magnitude is used and the direction argument wins.
    """
    if direction == "short":
        return wrap_delta(target - start) + turns * STEPS_PER_REV
    if direction == "cw":
        return (target - start) % STEPS_PER_REV + abs(turns) * STEPS_PER_REV
    return -((start - target) % STEPS_PER_REV) - abs(turns) * STEPS_PER_REV


def set_mode(sv, mode):
    """Switch position(0) / wheel(1). This is an EPROM write: it persists across
    power cycles, so it is only ever done on an explicit --mode request."""
    sv.w8(R_LOCK, 0)
    time.sleep(0.03)
    sv.w8(R_MODE, mode)
    time.sleep(0.03)
    sv.w8(R_LOCK, 1)
    time.sleep(0.03)
    return sv.r8(R_MODE)


def rotate_to(sv, target_deg, turns=0, max_speed=1500, min_speed=80, tol=8,
              timeout=40.0, direction="short"):
    """Continuous-rotation move that still lands on a chosen angle.

    Wheel mode ignores goal positions, so the loop is closed here: unwrap the
    encoder across the 4095->0 seam to track true travel, drive speed
    proportional to the remaining error, and re-correct after the coast settles.
    `turns` adds whole revolutions before settling. `direction` forces which
    way round to approach: "short" (default, <=180 deg), "cw" or "ccw". See
    plan_travel().
    """
    target = int(round(target_deg * STEPS_PER_REV / 360.0)) % STEPS_PER_REV
    start = sv.r16(R_PRES_POS)
    if start is None:
        raise RuntimeError("no position feedback")

    remaining = plan_travel(start, target, turns, direction)
    if remaining == 0:
        return start, 0

    # Deceleration has to be immediate. With the accel register limiting the ramp
    # down, a full-speed approach coasted ~500 steps (44 deg) past the target and
    # no amount of slow creeping recovered it inside the timeout.
    sv.w8(R_ACC, 0)
    # Begin braking a long way out, so arrival is already slow.
    kp = max_speed / 1500.0
    sv.torque(True)

    travelled, prev, t0 = 0, start, time.time()
    try:
        while time.time() - t0 < timeout:
            p = sv.r16(R_PRES_POS)
            if p is None:
                continue
            travelled += wrap_delta(p - prev)
            prev = p
            err = remaining - travelled
            if abs(err) <= tol:
                sv.w16(R_GOAL_SPEED, 0)
                time.sleep(0.15)
                p = sv.r16(R_PRES_POS)
                if p is None:
                    continue
                travelled += wrap_delta(p - prev)
                prev = p
                if abs(remaining - travelled) <= tol:
                    break                       # settled inside tolerance
                continue                        # coasted out; keep correcting
            mag = min(max_speed, max(min_speed, abs(err) * kp))
            sv.w16(R_GOAL_SPEED, (int(mag) & 0x7FFF) | (0x8000 if err < 0 else 0))
    finally:
        sv.w16(R_GOAL_SPEED, 0)                 # never leave it spinning

    return prev, remaining - travelled


def spin(sv, speed, seconds):
    """Free continuous rotation at a fixed speed, with a hard stop."""
    sv.torque(True)
    sv.w16(R_GOAL_SPEED, (abs(speed) & 0x7FFF) | (0x8000 if speed < 0 else 0))
    start, prev, travelled, t0 = sv.r16(R_PRES_POS), None, 0, time.time()
    prev = start
    try:
        while time.time() - t0 < seconds:
            time.sleep(0.1)
            p = sv.r16(R_PRES_POS)
            if p is None:
                continue
            travelled += wrap_delta(p - prev)
            prev = p
            print(f"   {time.time()-t0:4.1f}s  {p*360/STEPS_PER_REV:6.1f} deg  "
                  f"total {travelled/STEPS_PER_REV:+6.2f} turns", end="\r")
    finally:
        sv.w16(R_GOAL_SPEED, 0)
    time.sleep(0.3)
    print()
    return travelled


def monitor(sv, hz=10.0):
    """Passive sensor readout with torque released -- turn the horn by hand and
    watch. Writes no goal position, because that would switch torque back on."""
    sv.torque(False)
    time.sleep(0.15)
    sv.torque(False)
    time.sleep(0.15)
    print(f"torque released (reg40={sv.r8(R_TORQUE)}). Turn the horn by hand. "
          f"Ctrl-C to stop.\n")
    print(f"  {'pos':>5} {'deg':>7} {'speed':>7} {'load':>6} {'duty':>6} "
          f"{'curr':>5} {'mA':>7} {'V':>5} {'C':>4}")
    base = None
    try:
        while True:
            t = sv.telemetry()
            if t["pos"] is None:
                continue
            if base is None:
                base = t["pos"]
            print(f"  {t['pos']:5} {t['pos']*360/STEPS_PER_REV:7.2f} "
                  f"{t['speed']:7} {t['load']:6} {abs(t['load'])/10:5.1f}% "
                  f"{t['curr']:5} {t['ma']:7.0f} {t['volt']/10:5.1f} {t['temp']:4}",
                  end="\r")
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        t = sv.telemetry()
        print(f"\n\nmoved {t['pos']-base:+d} steps "
              f"({(t['pos']-base)*360/STEPS_PER_REV:+.2f} deg) by hand since start")


# ── finding the adapter ──────────────────────────────────────────────────────
#
# --port used to default to /dev/ttyACM0, which is a guess twice over: the
# adapter may enumerate as ttyUSB* (CH340, CP210x, FT232) rather than ttyACM*,
# and either number moves with whatever else was plugged in first. When the
# guess is wrong pyserial raises a bare ENOENT that reads exactly like "the
# adapter is unplugged", so you cannot tell the two apart from the traceback.
#
# Resolution order: --port, then the devices.yaml path (normally the
# /dev/aries_servo_bus symlink from 99-aries-servo-bus.rules, installed by
# scripts/setup_system.sh), then whatever USB-serial bridge is actually
# attached. Anything explicit still wins.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEVICES_YAML = REPO_ROOT / "src" / "aries_common" / "config" / "devices.yaml"

# WCH CH340/CH343, Silabs CP210x, FTDI FT232 — the bridges these adapters ship
# with, and the same list 99-aries-servo-bus.rules matches on. Kept in sync by
# hand; they are chip IDs, so they change about never.
BRIDGE_VIDS = ("1a86", "10c4", "0403")


def devices_entry(key, default=None):
    """servo_bus.<key> from devices.yaml, best-effort.

    Deliberately NOT `from aries_common.devices import device`: this script
    re-execs itself under /usr/bin/python3 to reach the apt pyserial, and that
    interpreter has no ROS environment, so the import would fail exactly when
    the script is being useful. A bench tool must not need a sourced workspace.
    """
    path = Path(os.environ.get("ARIES_DEVICES_FILE") or DEVICES_YAML)
    try:
        import yaml

        entry = (yaml.safe_load(path.read_text()) or {}).get("servo_bus") or {}
    except Exception:
        return default
    value = entry.get(key)
    return default if value is None or value == "" else value


def usb_ids(dev):
    """(idVendor, idProduct) for a tty, or None if it is not on USB.

    Walks up from the tty's device node rather than assuming a fixed depth:
    a ttyUSB sits under usb-serial -> interface -> device, a ttyACM one level
    higher, and hubs add more.
    """
    node = Path("/sys/class/tty") / Path(dev).name / "device"
    try:
        node = node.resolve(strict=True)
    except OSError:
        return None
    for _ in range(8):
        vid = node / "idVendor"
        if vid.is_file():
            try:
                return vid.read_text().strip(), (node / "idProduct").read_text().strip()
            except OSError:
                return None
        if node.parent == node:
            break
        node = node.parent
    return None


def serial_ports():
    """Attached serial ttys, as (device, vendor_id, product_id)."""
    found = []
    for pattern in ("/dev/ttyUSB[0-9]*", "/dev/ttyACM[0-9]*"):
        for dev in sorted(glob.glob(pattern)):
            ids = usb_ids(dev)
            found.append((dev, *(ids or (None, None))))
    return found


def resolve_port(explicit):
    """The port to open, or a message explaining why there is not one."""
    if explicit:
        return explicit, None

    configured = devices_entry("port")
    if configured and os.path.exists(configured):
        return configured, None

    attached = serial_ports()
    bridges = [dev for dev, vid, _ in attached if vid in BRIDGE_VIDS]
    if len(bridges) == 1:
        return bridges[0], None
    if len(bridges) > 1:
        return None, (
            "More than one USB-serial adapter is attached and none of them is "
            + f"{configured or 'the configured port'}:\n  " + "\n  ".join(bridges)
            + "\nPass --port to choose, or set servo_bus.serial in "
              "devices.yaml and re-run scripts/setup_system.sh to pin one."
        )

    if attached:
        listed = "\n  ".join(f"{dev} (usb {vid}:{pid})" if vid else f"{dev} (not USB)"
                             for dev, vid, pid in attached)
        return None, (
            "No USB-serial bridge that looks like a bus-servo adapter is "
            f"attached. Serial ports that ARE here:\n  {listed}\n"
            "Pass --port explicitly if one of these is the adapter."
        )
    return None, (
        "No serial adapter is attached at all — /dev/ttyUSB* and /dev/ttyACM* "
        "are both empty.\nPlug the ST3215 bus-servo adapter in (and power the "
        "servo bus; the adapter enumerates on USB alone but the servo will not "
        "answer without bus voltage).\nIf it IS plugged in, check it enumerated:"
        "\n  journalctl -k -n 20 | grep -i tty"
    )


def read_state(sv, tries=3):
    """Telemetry plus mode, retried field by field.

    A single dropped reply is normal on this bus -- one in ~30 reads during
    bench testing -- and a one-shot readout that prints "?" because of it is
    worse than useless, because the next thing you do is doubt the servo.
    Keep the first non-None answer for each field and re-ask for the rest.
    """
    state = dict(mode=None)
    for _ in range(tries):
        fresh = sv.telemetry()
        fresh["mode"] = sv.r8(R_MODE)
        for key, value in fresh.items():
            if state.get(key) is None:
                state[key] = value
        if all(v is not None for v in state.values()):
            break
    return state


def report(sv, port, baud):
    """One-shot state readout. READS ONLY -- no goal position, no torque
    change, no EPROM write -- so it is safe on a servo holding a load."""
    s = read_state(sv)

    def deg(steps):
        return steps * 360 / STEPS_PER_REV

    def line(label, value, unit="", extra=""):
        print(f"  {label:<10} {value:>8}{unit}{extra}")

    pos, mode = s["pos"], s["mode"]
    print(f"ST3215 id={sv.sid} on {port} @ {baud}\n")
    line("position", pos if pos is None else f"{pos}", " steps",
         "" if pos is None else f"   {deg(pos):.1f} deg")
    line("speed", "?" if s["speed"] is None else f"{s['speed']}", " steps/s",
         "" if s["speed"] is None else f"   {deg(s['speed']):.1f} deg/s")
    line("load", "?" if s["load"] is None else f"{s['load']}", "",
         "" if s["load"] is None else f"         {abs(s['load'])/10:.1f}%")
    line("current", "?" if s["ma"] is None else f"{s['ma']:.0f}", " mA")
    line("voltage", "?" if s["volt"] is None else f"{s['volt']/10:.1f}", " V")
    line("temp", "?" if s["temp"] is None else f"{s['temp']}", " C")
    line("mode", "?" if mode is None else f"{mode}", "",
         "" if mode is None else
         f"         {'position' if mode == 0 else 'wheel' if mode == 1 else mode}")
    line("torque", "?" if s["torque"] is None else
         ("on" if s["torque"] else "off"))

    if pos is None:
        return 0

    # The 50-4045 window only binds in POSITION mode. In wheel mode the horn
    # turns continuously and the encoder just wraps, so printing a "largest
    # --deg that fits" there would invent a limit that does not exist -- and
    # --deg is refused in wheel mode anyway.
    if mode != 0:
        print("\n  wheel mode: rotation is continuous, no travel limit. "
              "Use --goto/--spin;\n  --deg needs position mode.")
        return 0

    # The sweep runs out-and-back from wherever the horn is, so what you
    # actually want to know before typing --deg is how much room is left on
    # each side. Printing the position alone makes you work that out by hand.
    up, down = POS_MAX - pos, pos - POS_MIN
    print(f"\n  travel {POS_MIN}-{POS_MAX} steps ({deg(POS_MAX - POS_MIN):.0f} deg); "
          f"room from here: +{deg(up):.0f} deg / -{deg(down):.0f} deg")
    print(f"  largest --deg that fits from here: {deg(max(up, down)):.0f}")
    if max(up, down) < POS_MAX - POS_MIN:
        print(f"  mid-travel is {(POS_MIN + POS_MAX) // 2} "
              f"({deg((POS_MIN + POS_MAX) // 2):.1f} deg)")
    return 0


def scan(port, baud, maxid=20):
    s = Servo(port, baud, 1, timeout=0.02)
    hits = [i for i in range(maxid + 1) if s.ping(i)]
    s.close()
    return hits


def main():
    ap = argparse.ArgumentParser(description="ST3215 bench sweep test")
    ap.add_argument("--port", default=None,
                    help="serial port. Default: servo_bus.port from "
                         "devices.yaml, else the attached USB-serial adapter.")
    ap.add_argument("--baud", type=int, default=devices_entry("baud", 1000000))
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--deg", type=float, default=180.0, help="sweep angle each way")
    ap.add_argument("--speed", type=int, default=600,
                    help="steps/s (4096 steps = 360 deg). Measured ceiling on this "
                         "unit at 10.4 V is ~2850 steps/s (41.7 rpm); higher values "
                         "are simply clipped by the motor.")
    ap.add_argument("--acc", type=int, default=20,
                    help="acceleration, 0 = unlimited. NOTE this also caps top speed "
                         "over a short move: measured peak was 650 steps/s at acc=1, "
                         "1950 at acc=10, and full ~2850 only at acc>=50.")
    ap.add_argument("--repeat", type=int, default=1, help="number of out-and-back cycles")
    ap.add_argument("--scan", action="store_true", help="just list responding IDs")
    ap.add_argument("--where", action="store_true",
                    help="print position and state, then exit. Reads only; "
                         "moves nothing and changes no setting.")
    ap.add_argument("--monitor", action="store_true",
                    help="release torque and stream sensors; back-drive by hand")
    ap.add_argument("--mode", choices=("position", "wheel"),
                    help="switch operating mode. EPROM write, persists across "
                         "power cycles. wheel = continuous rotation.")
    ap.add_argument("--pos", type=int, metavar="STEPS",
                    help="POSITION-MODE absolute move to a raw step count "
                         "(0-4095, 0 deg = step 0). This is the one that works "
                         "in the mode the gripper runs in; --goto below is "
                         "wheel-mode only. Prints the travel before moving.")
    ap.add_argument("--goto", type=float, metavar="DEG",
                    help="continuous-rotation move to an absolute angle (needs "
                         "wheel mode). Software-closed loop, so it can cross the "
                         "0/360 seam freely.")
    ap.add_argument("--dir", choices=("short", "cw", "ccw"), default="short",
                    help="which way --goto approaches the target. short (default) "
                         "takes the nearer way round, never more than 180 deg; "
                         "cw drives the encoder count up, ccw drives it down, "
                         "the long way round if that is what the target needs.")
    ap.add_argument("--turns", type=float, default=0,
                    help="whole extra revolutions before settling on --goto. "
                         "Sign chooses direction with --dir short; with --dir "
                         "cw/ccw the direction wins and only the count is used.")
    ap.add_argument("--spin", type=int, metavar="STEPS_PER_S",
                    help="free continuous spin at this signed speed (wheel mode)")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="how long --spin runs")
    args = ap.parse_args()

    args.port, why = resolve_port(args.port)
    if args.port is None:
        return why

    if args.scan:
        print(f"IDs responding on {args.port} @ {args.baud}: "
              f"{scan(args.port, args.baud) or 'none'}")
        return 0

    sv = Servo(args.port, args.baud, args.id)
    if not sv.ping():
        sv.close()
        return f"No response from ID {args.id} on {args.port} @ {args.baud} baud."

    if args.where:
        rc = report(sv, args.port, args.baud)
        sv.close()
        return rc

    if args.monitor:
        monitor(sv)
        sv.close()
        return 0

    if args.mode:
        want = 0 if args.mode == "position" else 1
        was = sv.r8(R_MODE)
        now = set_mode(sv, want)
        print(f"mode {was} -> {now} ({args.mode}); written to EPROM, persists.")
        if now != want:
            sv.close()
            return "mode write did not take effect"
        if args.goto is None and args.spin is None and args.pos is None:
            sv.close()
            return 0

    if args.pos is not None:
        mode = sv.r8(R_MODE)
        if mode != 0:
            sv.close()
            return (f"--pos needs POSITION mode; the servo is in mode {mode} "
                    "(wheel), where goal positions are ignored. "
                    "Run with --mode position first.")
        here = sv.r16(R_PRES_POS)
        if here is None:
            sv.close()
            return "no position feedback"
        target = max(0, min(STEPS_PER_REV - 1, args.pos))
        # Say how far it is about to move BEFORE moving. With the racks engaged
        # a long travel runs the jaws into an end stop, and the horn alone gives
        # no clue how much of the stroke is left.
        print(f"  now    {here:>4} ({here * 360 / STEPS_PER_REV:6.1f} deg)")
        print(f"  target {target:>4} ({target * 360 / STEPS_PER_REV:6.1f} deg)"
              f"   travel {target - here:+d} steps "
              f"({(target - here) * 360 / STEPS_PER_REV:+.1f} deg)")
        if not (POS_MIN <= target <= POS_MAX):
            print(f"  NOTE: {target} is outside the {POS_MIN}-{POS_MAX} working band; "
                  "that band exists to stay off the hard 0/4095 stops.")
        sv.w16(R_GOAL_SPEED, args.speed)
        sv.goto(target, clamp=False)
        t = sv.telemetry()
        print(f"  settled at {t['pos']} ({(t['pos'] or 0) * 360 / STEPS_PER_REV:.1f} deg), "
              f"load {t['load']}, {t['volt'] / 10.0 if t['volt'] else '?'} V, {t['temp']} C")
        sv.close()
        return 0

    if args.goto is not None or args.spin is not None:
        if sv.r8(R_MODE) != 1:
            sv.close()
            return ("This needs wheel mode; the servo is in position mode. "
                    "Run with --mode wheel first (an EPROM write that persists).")
        sv.w8(R_ACC, args.acc)
        try:
            if args.spin is not None:
                print(f"spinning at {args.spin} steps/s for {args.seconds}s")
                t = spin(sv, args.spin, args.seconds)
                print(f"travelled {t/STEPS_PER_REV:+.2f} turns "
                      f"({t*360/STEPS_PER_REV:+.0f} deg)")
            else:
                p0 = sv.r16(R_PRES_POS)
                if p0 is None:
                    raise RuntimeError("no position feedback")
                # Print the resolved plan, not just the request: with a forced
                # direction the travel can be a near-full turn for a target a
                # few degrees away, and that is worth seeing BEFORE it moves.
                target = int(round(args.goto * STEPS_PER_REV / 360.0)) % STEPS_PER_REV
                travel = plan_travel(p0, target, int(args.turns), args.dir)
                print(f"at {p0*360/STEPS_PER_REV:.1f} deg -> {args.goto:.1f} deg, "
                      f"dir {args.dir}, {args.turns:+g} extra turns")
                print(f"travel {travel*360/STEPS_PER_REV:+.1f} deg "
                      f"({travel:+d} steps, {'up' if travel >= 0 else 'down'} "
                      f"the encoder)")
                end, err = rotate_to(sv, args.goto, int(args.turns),
                                     max_speed=args.speed, direction=args.dir)
                print(f"landed {end*360/STEPS_PER_REV:.2f} deg "
                      f"(target {args.goto % 360:.2f}), error {err} steps "
                      f"= {abs(err)*360/STEPS_PER_REV:.2f} deg")
        finally:
            sv.w16(R_GOAL_SPEED, 0)
            sv.torque(False)
        sv.close()
        return 0

    volt, temp, mode = sv.r8(R_VOLT), sv.r8(R_TEMP), sv.r8(R_MODE)
    print(f"ST3215 id={args.id}  {volt / 10:.1f} V  {temp} C  "
          f"mode={mode} ({'position' if mode == 0 else 'wheel' if mode == 1 else mode})")
    if mode != 0:
        sv.close()
        return (f"Servo is in mode {mode}, not position mode. This sweep needs "
                f"position mode; a wheel-mode servo ignores goal positions.")

    span = int(round(args.deg * STEPS_PER_REV / 360.0))

    # Set the profile before any goal position: STS latches speed/accel when a
    # move begins, so rewriting them mid-flight is silently ignored.
    sv.w16(44, 0)                        # clear any goal-time override
    sv.w8(R_ACC, args.acc)
    sv.w16(R_GOAL_SPEED, args.speed)
    sv.torque(True)
    time.sleep(0.05)

    start = sv.r16(R_PRES_POS)
    # Sweep out from wherever the horn already is -- no repositioning first.
    # Travel is a bounded 0-4095, so take whichever direction still fits.
    if start + span <= POS_MAX:
        far, sign = start + span, "+"
    elif start - span >= POS_MIN:
        far, sign = start - span, "-"
    else:
        sv.torque(False)
        sv.close()
        return (f"{args.deg:.0f} deg ({span} steps) does not fit from position "
                f"{start} in either direction; travel is {POS_MIN}-{POS_MAX} steps. "
                f"Use a smaller --deg, or move the horn nearer mid-travel first.")

    print(f"start {start} ({start * 360 / STEPS_PER_REV:.1f} deg) -> "
          f"{sign}{args.deg:.0f} deg to {far} ({far * 360 / STEPS_PER_REV:.1f} deg) "
          f"and back, at {args.speed} steps/s "
          f"({args.speed * 360 / STEPS_PER_REV:.0f} deg/s)\n")

    end = start
    try:
        for n in range(args.repeat):
            print(f"  cycle {n + 1}/{args.repeat}: {sign}{args.deg:.0f} deg")
            sv.goto(far)
            print(f"  cycle {n + 1}/{args.repeat}: back to start")
            end = sv.goto(start)
        print(f"\ndone. started {start}, ended {end}, "
              f"error {abs(end - start)} steps "
              f"({abs(end - start) * 360 / STEPS_PER_REV:.2f} deg)")
    except KeyboardInterrupt:
        print("\ninterrupted -- holding here")
    finally:
        sv.torque(False)
        print("torque released")
        sv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

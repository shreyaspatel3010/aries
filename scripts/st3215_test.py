#!/usr/bin/python3
"""Bench test for a Feetech ST3215 bus servo on a USB bus-servo adapter.

Sweeps the horn 180 deg one way then 180 deg back, in position mode.
Nothing is written to EPROM, so the servo's stored config is untouched.

Runnable with any interpreter: the repo .venv has no pyserial, so if it is
missing the script re-execs itself under /usr/bin/python3 (apt python3-serial).

    python3 scripts/st3215_test.py
    python3 scripts/st3215_test.py --deg 90 --speed 400 --repeat 3
    python3 scripts/st3215_test.py --scan
    python3 scripts/st3215_test.py --monitor    # torque off, back-drive by hand

Note: writing a goal position re-enables torque automatically, so "torque off"
only sticks if no goal is written afterwards.
"""
import argparse
import sys
import time

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

    def goto(self, target, timeout=8.0, quiet=False):
        """Command a position (steps) and block until it settles."""
        target = max(POS_MIN, min(POS_MAX, int(target)))
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
              timeout=40.0):
    """Continuous-rotation move that still lands on a chosen angle.

    Wheel mode ignores goal positions, so the loop is closed here: unwrap the
    encoder across the 4095->0 seam to track true travel, drive speed
    proportional to the remaining error, and re-correct after the coast settles.
    `turns` adds whole revolutions before settling (sign picks the direction).
    """
    target = int(round(target_deg * STEPS_PER_REV / 360.0)) % STEPS_PER_REV
    start = sv.r16(R_PRES_POS)
    if start is None:
        raise RuntimeError("no position feedback")

    short = wrap_delta(target - start)
    remaining = short + turns * STEPS_PER_REV
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


def scan(port, baud, maxid=20):
    s = Servo(port, baud, 1, timeout=0.02)
    hits = [i for i in range(maxid + 1) if s.ping(i)]
    s.close()
    return hits


def main():
    ap = argparse.ArgumentParser(description="ST3215 bench sweep test")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1000000)
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
    ap.add_argument("--monitor", action="store_true",
                    help="release torque and stream sensors; back-drive by hand")
    ap.add_argument("--mode", choices=("position", "wheel"),
                    help="switch operating mode. EPROM write, persists across "
                         "power cycles. wheel = continuous rotation.")
    ap.add_argument("--goto", type=float, metavar="DEG",
                    help="continuous-rotation move to an absolute angle (needs "
                         "wheel mode). Software-closed loop, so it can cross the "
                         "0/360 seam freely.")
    ap.add_argument("--turns", type=float, default=0,
                    help="whole extra revolutions before settling on --goto; "
                         "sign chooses direction")
    ap.add_argument("--spin", type=int, metavar="STEPS_PER_S",
                    help="free continuous spin at this signed speed (wheel mode)")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="how long --spin runs")
    args = ap.parse_args()

    if args.scan:
        print(f"IDs responding on {args.port} @ {args.baud}: "
              f"{scan(args.port, args.baud) or 'none'}")
        return 0

    sv = Servo(args.port, args.baud, args.id)
    if not sv.ping():
        sv.close()
        return f"No response from ID {args.id} on {args.port} @ {args.baud} baud."

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
        if args.goto is None and args.spin is None:
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
                print(f"at {p0*360/STEPS_PER_REV:.1f} deg -> "
                      f"{args.goto:.1f} deg, {args.turns:+g} extra turns")
                end, err = rotate_to(sv, args.goto, int(args.turns),
                                     max_speed=args.speed)
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

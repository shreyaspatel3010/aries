"""
Provide acceleration-independent planar orientation estimation.

The filter estimates yaw from bias-corrected Z angular velocity and applies a
slow correction from the horizontal magnetometer direction. It intentionally
does not accept acceleration as an input.

The magnetometer is only trusted when it behaves like a rotating view of one
constant field: the hard-iron offset is estimated online, the field magnitude
must stay near the calibrated norm, and the correction is suspended while the
platform turns quickly. On this rover the raw horizontal field carries a
hard-iron offset half the size of the horizontal earth field, which alone
swings the apparent heading by +/-30 degrees over a turn.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple


Vector3 = Tuple[float, float, float]
QuaternionWxyz = Tuple[float, float, float, float]

_HARD_IRON_BIN_COUNT = 12
_HARD_IRON_MIN_BINS = 8
_DISPERSION_WINDOW = 25
# Largest bias error worth entertaining, as a multiple of the stillness rate
# gate. Bounds how much doubt a sustained steady turn can inject.
_MAX_PLAUSIBLE_BIAS_GATES = 5.0


class RollingDispersion:
    """Report how much the gyro is varying, independent of its offset.

    Stillness must be judged from dispersion, never from magnitude: a
    stationary gyro carries its bias, measured at -0.27 rad/s on this unit, so
    an absolute threshold is unsatisfiable and would reject every sample.
    """

    def __init__(self, window: int = _DISPERSION_WINDOW) -> None:
        self.window = max(2, window)
        self.reset()

    def reset(self) -> None:
        """Forget the buffered samples."""
        self._samples: List[Vector3] = []

    def add(self, angular_velocity: Vector3) -> None:
        """Buffer one sample, dropping the oldest beyond the window."""
        self._samples.append(tuple(angular_velocity))
        if len(self._samples) > self.window:
            del self._samples[0]

    def ready(self) -> bool:
        """Report whether enough samples exist to judge dispersion."""
        return len(self._samples) >= self.window

    def exceeds(self, threshold: float) -> bool:
        """Report whether any axis varies by more than ``threshold``."""
        if not self.ready():
            return False
        for axis in range(3):
            values = [sample[axis] for sample in self._samples]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            if math.sqrt(variance) > threshold:
                return True
        return False


class AdaptiveBiasEstimator:
    """Track gyro bias per axis with a gain that follows its own uncertainty.

    A fixed averaging window cannot be both immediate and accurate: it has to
    either commit on very few samples or make the caller wait. A recursive
    estimator that carries its own variance does both. The prior is deliberately
    wide, so the first stationary sample collapses it almost completely and the
    bias is usable at once; every later sample then refines it, shrinking the
    uncertainty as 1/sqrt(N) without any window to choose.

    Uncertainty is also allowed to REGROW, which is what keeps accumulated yaw
    error small on a long run. Two things drive it: a random walk in time, and
    the temperature change since the last stationary observation - the dominant
    real cause of MEMS bias drift, available here because the barometer reports
    temperature. So after a long drive or a thermal transient the estimator is
    uncertain again and relocks within a fraction of a second of stopping,
    rather than trusting a cold-start offset for the rest of the mission.

    The estimator never folds in a sample the caller has not certified as
    stationary, so motion cannot be averaged into the bias.
    """

    def __init__(
        self,
        noise_rad_s: float = 0.003,
        walk_rad_s: float = 0.0002,
        initial_uncertainty_rad_s: float = 0.5,
        temperature_sensitivity_rad_s_per_c: float = 0.002,
    ) -> None:
        self.measurement_variance = max(1.0e-12, float(noise_rad_s) ** 2)
        self.walk_variance_rate = max(0.0, float(walk_rad_s)) ** 2
        self.initial_variance = max(
            self.measurement_variance, float(initial_uncertainty_rad_s) ** 2)
        self.temperature_sensitivity = max(
            0.0, float(temperature_sensitivity_rad_s_per_c))
        self.reset()

    def reset(self) -> None:
        """Forget the estimate and return to the wide prior."""
        self.bias: Vector3 = (0.0, 0.0, 0.0)
        self._variance = [self.initial_variance] * 3
        self._temperature_c: Optional[float] = None
        self._temperature_at_measurement: Optional[float] = None
        self.measurements = 0

    def _thermal_variance(self) -> float:
        """Doubt admitted for temperature drift since the last measurement.

        Modelled as a systematic shift proportional to the total temperature
        change, not as a per-sample random walk: squaring a per-sample delta
        would make a slow warm-up vanish into rounding.
        """
        if (self.temperature_sensitivity <= 0.0
                or self._temperature_c is None
                or self._temperature_at_measurement is None):
            return 0.0
        delta = self._temperature_c - self._temperature_at_measurement
        return (self.temperature_sensitivity * delta) ** 2

    def variance(self, axis: int) -> float:
        """Total uncertainty for one axis, including the thermal term."""
        return self._variance[axis] + self._thermal_variance()

    @property
    def uncertainty_rad_s(self) -> Vector3:
        """Standard deviation of the bias estimate on each axis."""
        return tuple(math.sqrt(self.variance(axis)) for axis in range(3))

    def note_temperature(self, temperature_c: Optional[float]) -> None:
        """Record the latest temperature reading, if one is available."""
        if temperature_c is not None and math.isfinite(temperature_c):
            self._temperature_c = float(temperature_c)
            if self._temperature_at_measurement is None:
                self._temperature_at_measurement = self._temperature_c

    def predict(self, dt: float) -> None:
        """Grow uncertainty for elapsed time so the gain never reaches zero."""
        if dt <= 0.0 or self.walk_variance_rate <= 0.0:
            return
        growth = self.walk_variance_rate * dt
        self._variance = [value + growth for value in self._variance]

    def inflate(self, discrepancy: float, fraction: float) -> None:
        """Admit doubt worth ``discrepancy`` when the estimate looks wrong.

        Used when the stillness gate keeps rejecting samples. That is ambiguous
        evidence - either the platform is moving or this bias is wrong - so the
        doubt is admitted gradually. Once it approaches the size of the
        discrepancy the next accepted sample carries almost full weight, which
        is what lets a badly wrong bias recover instead of deadlocking.
        """
        if discrepancy <= 0.0 or fraction <= 0.0:
            return
        growth = (discrepancy ** 2) * fraction
        self._variance = [value + growth for value in self._variance]

    def measure(self, angular_velocity: Vector3) -> None:
        """Fold in one stationary sample, where the raw rate IS the bias."""
        bias = list(self.bias)
        thermal = self._thermal_variance()
        for axis in range(3):
            total = self._variance[axis] + thermal
            gain = total / (total + self.measurement_variance)
            bias[axis] += gain * (angular_velocity[axis] - bias[axis])
            # Absorb the thermal term as it is resolved by this observation.
            self._variance[axis] = (1.0 - gain) * total
        self.bias = tuple(bias)
        self._temperature_at_measurement = self._temperature_c
        self.measurements += 1


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PlanarFilterOutput:
    """One planar filter result."""

    quaternion: QuaternionWxyz
    angular_velocity: Vector3
    yaw: float
    calibrating: bool
    calibration_completed: bool
    magnetic_correction_used: bool
    hard_iron_offset: Vector3 = (0.0, 0.0, 0.0)
    hard_iron_calibrated: bool = False
    bias_confident: bool = True


class HardIronEstimator:
    """Fit the horizontal hard-iron offset from a rotating magnetometer.

    Solves the circle equation |m|^2 + D*mx + E*my + F = 0 by accumulating
    normal equations with exponential forgetting. The fit is only published
    once the samples span enough distinct headings, because a circle is not
    observable from a narrow arc.
    """

    def __init__(self, forgetting_factor: float = 0.9995) -> None:
        self.forgetting_factor = min(max(forgetting_factor, 0.9), 1.0)
        self.reset()

    def reset(self) -> None:
        """Discard the accumulated fit and heading coverage."""
        self._normal = [[0.0] * 3 for _ in range(3)]
        self._target = [0.0] * 3
        self._bins = set()
        self.offset: Vector3 = (0.0, 0.0, 0.0)
        self.field_norm: Optional[float] = None
        self.calibrated = False

    def seed(self, offset: Vector3, field_norm: Optional[float]) -> None:
        """Adopt a stored calibration so a restart does not relearn it."""
        self.offset = (float(offset[0]), float(offset[1]), float(offset[2]))
        if field_norm is not None and field_norm > 0.0:
            self.field_norm = float(field_norm)
            self.calibrated = True

    def add(self, mx: float, my: float) -> None:
        """Accumulate one horizontal sample into the circle fit."""
        row = (mx, my, 1.0)
        value = -(mx * mx + my * my)
        decay = self.forgetting_factor
        for i in range(3):
            self._target[i] = decay * self._target[i] + row[i] * value
            for j in range(3):
                self._normal[i][j] = decay * self._normal[i][j] + row[i] * row[j]

        bin_index = int(
            (math.atan2(my - self.offset[1], mx - self.offset[0]) + math.pi)
            / (2.0 * math.pi) * _HARD_IRON_BIN_COUNT) % _HARD_IRON_BIN_COUNT
        self._bins.add(bin_index)
        if len(self._bins) >= _HARD_IRON_MIN_BINS:
            self._solve()

    def _solve(self) -> None:
        solution = _solve_3x3(self._normal, self._target)
        if solution is None:
            return
        d, e, f = solution
        center_x, center_y = -0.5 * d, -0.5 * e
        radius_squared = center_x * center_x + center_y * center_y - f
        if not math.isfinite(radius_squared) or radius_squared <= 0.0:
            return
        radius = math.sqrt(radius_squared)
        if not all(math.isfinite(v) for v in (center_x, center_y, radius)):
            return
        self.offset = (center_x, center_y, self.offset[2])
        self.field_norm = radius
        self.calibrated = True


def _solve_3x3(matrix: List[List[float]], vector: List[float]):
    """Solve a 3x3 system by Gaussian elimination, or return None if singular."""
    augmented = [list(matrix[i]) + [vector[i]] for i in range(3)]
    for column in range(3):
        pivot_row = max(range(column, 3), key=lambda r: abs(augmented[r][column]))
        pivot = augmented[pivot_row][column]
        scale = max(abs(augmented[r][column]) for r in range(3))
        if scale <= 0.0 or abs(pivot) <= 1.0e-12 * scale:
            return None
        augmented[column], augmented[pivot_row] = \
            augmented[pivot_row], augmented[column]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, 4):
                augmented[row][k] -= factor * augmented[column][k]
    try:
        return tuple(augmented[i][3] / augmented[i][i] for i in range(3))
    except ZeroDivisionError:
        return None


class PlanarGyroMagFilter:
    """Estimate yaw without using linear acceleration."""

    def __init__(
        self,
        estimate_gyro_bias: bool = True,
        gyro_bias_stillness_rad_s: float = 0.08,
        gyro_noise_rad_s: float = 0.003,
        gyro_bias_walk_rad_s: float = 0.00005,
        gyro_bias_initial_uncertainty_rad_s: float = 0.5,
        gyro_bias_confident_rad_s: float = 0.001,
        gyro_bias_temperature_sensitivity_rad_s_per_c: float = 0.002,
        gyro_bias_max_corrected_rate_rad_s: float = 0.02,
        gyro_bias_relock_s: float = 60.0,
        magnetic_correction_time_constant_s: float = 5.0,
        magnetic_field_min_t: float = 10.0e-6,
        magnetic_field_max_t: float = 120.0e-6,
        magnetic_norm_tolerance: float = 0.25,
        magnetic_heading_rejection_rad: float = math.radians(45.0),
        magnetic_max_rate_rad_s: float = 0.5,
        hard_iron_offset_t: Vector3 = (0.0, 0.0, 0.0),
        hard_iron_field_norm_t: Optional[float] = None,
        estimate_hard_iron: bool = True,
        max_dt_s: float = 0.2,
    ) -> None:
        self.estimate_gyro_bias = bool(estimate_gyro_bias)
        self.gyro_bias_stillness_rad_s = max(0.0, gyro_bias_stillness_rad_s)
        # Uncertainty at or below this counts as calibrated. It is reached in a
        # fraction of a second, so there is no startup window to wait out.
        self.gyro_bias_confident_rad_s = max(0.0, gyro_bias_confident_rad_s)
        # Dispersion cannot see a slow STEADY turn - a constant rate has no
        # spread at all - so dispersion alone would let the estimator absorb a
        # gentle turn and stop reporting it. Once the bias is usable, require
        # the CORRECTED rate to be near zero as well. The absolute threshold
        # this file warns against applies to the RAW rate, which carries the
        # -0.27 rad/s offset; on the corrected rate it is exactly the right
        # test. Keep it below any commanded turn and above plausible bias error.
        self.gyro_bias_max_corrected_rate_rad_s = max(
            0.0, gyro_bias_max_corrected_rate_rad_s)
        # Escape hatch: if bias error ever exceeded the gate the estimator would
        # reject every sample and stay wrong forever. After this long without an
        # accepted measurement, take one anyway.
        self.gyro_bias_relock_s = max(0.0, gyro_bias_relock_s)
        self.bias_estimator = AdaptiveBiasEstimator(
            noise_rad_s=gyro_noise_rad_s,
            walk_rad_s=gyro_bias_walk_rad_s,
            initial_uncertainty_rad_s=gyro_bias_initial_uncertainty_rad_s,
            temperature_sensitivity_rad_s_per_c=(
                gyro_bias_temperature_sensitivity_rad_s_per_c),
        )
        self.magnetic_correction_time_constant_s = max(
            0.0, magnetic_correction_time_constant_s)
        self.magnetic_field_min_t = max(0.0, magnetic_field_min_t)
        self.magnetic_field_max_t = max(
            self.magnetic_field_min_t, magnetic_field_max_t)
        self.magnetic_norm_tolerance = max(0.0, magnetic_norm_tolerance)
        self.magnetic_heading_rejection_rad = max(
            0.0, magnetic_heading_rejection_rad)
        # A magnetometer sampled at 50 Hz aliases badly during a fast turn, and
        # motor current peaks with wheel torque. Integrate the gyro through
        # those intervals instead of steering yaw with a corrupted heading.
        self.magnetic_max_rate_rad_s = max(0.0, magnetic_max_rate_rad_s)
        self.max_dt_s = max(1.0e-3, max_dt_s)
        self.estimate_hard_iron = estimate_hard_iron
        self._seed_offset = tuple(float(v) for v in hard_iron_offset_t)
        self._seed_field_norm = hard_iron_field_norm_t
        self.hard_iron = HardIronEstimator()
        self._dispersion = RollingDispersion()
        self.reset()

    def reset(self, preserve_calibration: bool = False) -> None:
        """Reset the estimator, optionally keeping heading and calibration.

        A serial reconnect does not move the rover, so ``preserve_calibration``
        keeps yaw, the gyro bias, and the hard-iron fit. Relearning them would
        zero the heading mid-mission and average real motion into the bias.
        """
        self._first_time = None
        self._previous_time = None
        self._magnetic_reference = None
        self._dispersion.reset()
        if preserve_calibration and self._bias_calibrated_or_false():
            return

        self.yaw = 0.0
        self.bias_estimator.reset()
        # Latch: has a usable bias ever been reached? Distinct from the live
        # bias_confident, which tracks current uncertainty and so dips whenever
        # the platform moves. Arming the rate gate and preserving calibration
        # across a reconnect must not flicker with motion.
        self._has_calibrated = not self.estimate_gyro_bias
        self._last_bias_measurement_at = None
        self.hard_iron.reset()
        self.hard_iron.seed(self._seed_offset, self._seed_field_norm)

    @property
    def gyro_bias(self) -> Vector3:
        """Current bias estimate, zero when estimation is disabled."""
        if not self.estimate_gyro_bias:
            return (0.0, 0.0, 0.0)
        return self.bias_estimator.bias

    @property
    def gyro_bias_uncertainty(self) -> Vector3:
        """Standard deviation of the bias estimate on each axis."""
        return self.bias_estimator.uncertainty_rad_s

    @property
    def bias_confident(self) -> bool:
        """Whether the yaw-axis bias is known to within the threshold."""
        if not self.estimate_gyro_bias:
            return True
        return (self.bias_estimator.uncertainty_rad_s[2]
                <= self.gyro_bias_confident_rad_s)

    def _bias_calibrated_or_false(self) -> bool:
        return getattr(self, '_has_calibrated', False)

    @staticmethod
    def _yaw_quaternion(yaw: float) -> QuaternionWxyz:
        half_yaw = 0.5 * yaw
        return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))

    def _magnetic_body_angle(self, magnetic_field: Vector3):
        """Return the hard-iron corrected horizontal field angle, or None."""
        mx, my, mz = magnetic_field
        if not all(math.isfinite(value) for value in magnetic_field):
            return None
        magnitude = math.sqrt(mx * mx + my * my + mz * mz)
        if not self.magnetic_field_min_t <= magnitude <= self.magnetic_field_max_t:
            return None

        if self.estimate_hard_iron:
            self.hard_iron.add(mx, my)
        offset_x, offset_y, _ = self.hard_iron.offset
        corrected_x, corrected_y = mx - offset_x, my - offset_y
        horizontal = math.hypot(corrected_x, corrected_y)
        if horizontal < max(1.0e-9, self.magnetic_field_min_t * 0.2):
            return None

        # Under pure rotation a clean field traces a circle of constant radius.
        # A sample off that circle is local interference, not a heading.
        norm = self.hard_iron.field_norm
        if norm is not None and norm > 0.0 and self.magnetic_norm_tolerance > 0.0:
            if abs(horizontal - norm) > self.magnetic_norm_tolerance * norm:
                return None
        elif self.estimate_hard_iron and not self.hard_iron.calibrated:
            # Still learning the offset. An uncalibrated heading on this rover
            # is worth +/-30 deg of error, which is worse than integrating the
            # gyro alone, so withhold the correction until the fit converges.
            return None

        return math.atan2(corrected_y, corrected_x)

    def _corrected_angular_velocity(self, angular_velocity: Vector3) -> Vector3:
        return tuple(
            angular_velocity[index] - self.gyro_bias[index]
            for index in range(3)
        )

    def _output(self, corrected_angular, calibrating, calibration_completed,
                magnetic_correction_used) -> PlanarFilterOutput:
        return PlanarFilterOutput(
            quaternion=self._yaw_quaternion(self.yaw),
            angular_velocity=corrected_angular,
            yaw=self.yaw,
            calibrating=calibrating,
            calibration_completed=calibration_completed,
            magnetic_correction_used=magnetic_correction_used,
            hard_iron_offset=self.hard_iron.offset,
            hard_iron_calibrated=self.hard_iron.calibrated,
            bias_confident=getattr(self, 'bias_confident', True),
        )

    def update(
        self,
        timestamp_s: float,
        angular_velocity: Vector3,
        magnetic_field: Vector3,
        temperature_c: Optional[float] = None,
    ) -> PlanarFilterOutput:
        """Update yaw from gyro and magnetometer, never from acceleration."""
        if not math.isfinite(timestamp_s):
            raise ValueError('timestamp_s must be finite')
        if not all(math.isfinite(value) for value in angular_velocity):
            raise ValueError('angular_velocity values must be finite')

        if self._first_time is None:
            self._first_time = timestamp_s
            self._previous_time = timestamp_s

        previous_time = self._previous_time
        self._previous_time = timestamp_s
        dt = 0.0 if previous_time is None else timestamp_s - previous_time
        if dt < 0.0:
            dt = 0.0
        elif dt > self.max_dt_s:
            # Clamping keeps a bounded estimate of the rotation across a stall.
            # Zeroing dt would silently drop the turn that happened in the gap.
            dt = self.max_dt_s

        # Estimate the bias BEFORE integrating, so the very first stationary
        # window steers this sample rather than the next one.
        self._estimate_bias(timestamp_s, angular_velocity, dt, temperature_c)
        calibration_completed = self.bias_confident and not self._has_calibrated
        if calibration_completed:
            self._has_calibrated = True

        corrected_angular = self._corrected_angular_velocity(angular_velocity)
        self.yaw = wrap_angle(self.yaw + corrected_angular[2] * dt)

        magnetic_correction_used = False
        turning_fast = (
            self.magnetic_max_rate_rad_s > 0.0
            and abs(corrected_angular[2]) > self.magnetic_max_rate_rad_s
        )
        body_angle = None if turning_fast else self._magnetic_body_angle(magnetic_field)
        if body_angle is not None:
            if self._magnetic_reference is None:
                # alpha = body magnetic angle + world yaw. Capturing it here
                # makes the output yaw relative to the first valid field.
                self._magnetic_reference = wrap_angle(body_angle + self.yaw)
            measured_yaw = wrap_angle(self._magnetic_reference - body_angle)
            innovation = wrap_angle(measured_yaw - self.yaw)
            if abs(innovation) <= self.magnetic_heading_rejection_rad:
                if self.magnetic_correction_time_constant_s <= 0.0:
                    correction_fraction = 1.0
                else:
                    correction_fraction = 1.0 - math.exp(
                        -dt / self.magnetic_correction_time_constant_s)
                self.yaw = wrap_angle(
                    self.yaw + correction_fraction * innovation)
                magnetic_correction_used = correction_fraction > 0.0

        return self._output(
            corrected_angular, not self.bias_confident, calibration_completed,
            magnetic_correction_used)

    def _estimate_bias(self, timestamp_s, angular_velocity, dt,
                       temperature_c) -> None:
        """Refine the bias whenever the gyro is certifiably stationary.

        A sample is only offered to the estimator once the dispersion window is
        FULL. Before that there is no evidence of stillness, and folding in a
        turn that started at t=0 would bake it into the bias for the whole run -
        the failure this filter has always guarded against. That costs one
        dispersion window, a quarter second at 100 Hz, which is what makes
        startup effectively instant instead of a multi-second wait.

        Stillness then needs BOTH tests. Dispersion catches varying motion but
        is blind to a constant rate, so a slow steady turn shows zero spread;
        the corrected-rate gate catches exactly that case. Without it a gentle
        turn is absorbed into the bias within a few seconds and yaw stops
        reporting the turn at all.
        """
        if not self.estimate_gyro_bias:
            return
        self.bias_estimator.note_temperature(temperature_c)
        self.bias_estimator.predict(dt)
        self._dispersion.add(angular_velocity)
        if not self._dispersion.ready():
            return
        if self._dispersion.exceeds(self.gyro_bias_stillness_rad_s):
            return

        if self._has_calibrated and self.gyro_bias_max_corrected_rate_rad_s > 0.0:
            corrected = self._corrected_angular_velocity(angular_velocity)
            worst = max(abs(value) for value in corrected)
            if worst > self._allowed_corrected_rate(timestamp_s):
                # Either the platform is moving or this bias is wrong. Admit
                # doubt so a wrong bias regains weight rather than being locked
                # out by its own error, then leave the sample unused. Cap the
                # doubt at a few times the gate: a steady fast turn is motion,
                # not a plausible bias error, and taking it at face value would
                # throw away a good estimate every time the rover drives.
                if self.gyro_bias_relock_s > 0.0:
                    plausible = min(
                        worst, _MAX_PLAUSIBLE_BIAS_GATES
                        * self.gyro_bias_max_corrected_rate_rad_s)
                    self.bias_estimator.inflate(
                        plausible, dt / self.gyro_bias_relock_s)
                return

        self._last_bias_measurement_at = timestamp_s
        self.bias_estimator.measure(angular_velocity)

    def _allowed_corrected_rate(self, timestamp_s) -> float:
        """The rate gate, widened by how long it has been blocking.

        A fixed gate could deadlock: if the bias were ever wrong by more than
        the gate, every sample would look like motion and the estimate would
        stay wrong for the rest of the run. Widening it in proportion to the
        blocked interval guarantees a badly wrong bias is eventually admitted,
        while keeping the gate tight during normal operation - and any accepted
        sample resets the interval, so it re-narrows immediately.
        """
        gate = self.gyro_bias_max_corrected_rate_rad_s
        if self.gyro_bias_relock_s <= 0.0 or self._last_bias_measurement_at is None:
            return gate
        elapsed = timestamp_s - self._last_bias_measurement_at
        if elapsed <= 0.0:
            return gate
        return gate * (1.0 + elapsed / self.gyro_bias_relock_s)

"""YOLO model loading and background-thread inference.

Both nodes in this package need the same two things:

  * ``load_yolo_model`` — load the packaged weights on the requested device,
    run a CUDA warm-up so the first real frame is not slow, and fall back to
    CPU when the requested device fails.

  * ``YoloWorker`` — run inference off the rclpy executor thread. Model
    inference takes tens to hundreds of milliseconds; running it inside a
    timer/subscription callback blocks every other callback of a
    single-threaded executor (gripper ticks, action results, TF updates).
    The worker keeps only the newest submitted job and only the newest
    completed result, so consumers always see the freshest frame and stale
    frames are dropped instead of queueing.
"""

import os
import threading
import traceback
from typing import Any, Optional, Tuple

import numpy as np
from ament_index_python.packages import get_package_share_directory


# A frame-age magnitude no real session can reach (11.5 days). Past this the
# frame stamp and the consumer's clock are not on the same timeline at all —
# sim uptime counts from 0 while wall time is ~1.8e9 — so the age is
# meaningless rather than merely stale.
CLOCK_DOMAIN_MISMATCH_AGE_SEC = 1.0e6


def classify_frame_age(
    age_sec: float,
    using_sim_time: bool,
    max_age_sec: float,
    mismatch_age_sec: float = CLOCK_DOMAIN_MISMATCH_AGE_SEC,
) -> Tuple[str, str]:
    """Classify a frame age as ``'ok'``, ``'stale'`` or ``'clock_mismatch'``.

    Returns the verdict and, for the two reject verdicts, a human explanation.

    The mismatch check comes first and uses the *magnitude* of the age. Both
    details matter: an epoch-sized age reported against ``max_age_sec`` sends
    the reader off tuning a freshness bound that was never the problem, and the
    reverse mismatch (consumer on ``/clock``, frames stamped with wall time)
    makes the age hugely *negative*, which slips through a stale-frame test
    unnoticed and admits a frame whose stamp is meaningless.
    """
    if abs(age_sec) > mismatch_age_sec:
        if using_sim_time:
            cause = (
                'this node is on sim time (/clock) but the frames carry '
                'wall-clock stamps — relaunch with use_sim_time:=false, or '
                'point the camera topics at the simulator'
            )
        else:
            cause = (
                'this node is on wall-clock time but the frames carry sim '
                'stamps — relaunch with use_sim_time:=true'
            )
        return 'clock_mismatch', cause
    if age_sec > max_age_sec:
        return 'stale', (
            f'age={age_sec:.2f}s > inference_result_max_age_sec='
            f'{max_age_sec:.2f}s'
        )
    return 'ok', ''


def default_model_path() -> str:
    """Return the grasp model installed with this package."""
    return os.path.join(
        get_package_share_directory('aries_vision_grasp'), 'models', 'grasp.pt'
    )


def load_yolo_model(model_path: str, device: str = '', imgsz: int = 640, logger=None):
    """Load YOLO weights and warm them up; fall back to CPU on device failure.

    Returns ``(model, device)`` — device is the string that actually worked.
    Raises the underlying exception when the model cannot be loaded at all,
    so a misconfigured node fails loudly instead of idling forever.
    """
    from ultralytics import YOLO

    def _load(dev: str):
        model = YOLO(model_path)
        warmup = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        if dev:
            model(warmup, verbose=False, device=dev, imgsz=imgsz)
        else:
            model(warmup, verbose=False, imgsz=imgsz)
        return model

    try:
        model = _load(device)
        if logger is not None:
            logger.info(
                f'YOLO model ready: {model_path} | task={model.task} | '
                f'device={device or "auto"} | names={list(model.names.values())}'
            )
        return model, device
    except Exception:
        if device in ('', 'cpu'):
            raise
        if logger is not None:
            logger.warning(
                f'YOLO failed on device={device}, retrying on CPU:\n'
                f'{traceback.format_exc()}'
            )
        model = _load('cpu')
        if logger is not None:
            logger.info(
                f'YOLO model ready (CPU fallback): {model_path} | '
                f'task={model.task} | names={list(model.names.values())}'
            )
        return model, 'cpu'


class YoloWorker:
    """Latest-frame-wins background inference.

    ``submit(payload, image)`` stores the newest pending job (replacing any
    not-yet-started one). The worker thread runs the model on it and stores
    ``(payload, results)`` as the newest completed result, which
    ``take_result()`` hands to the consumer exactly once.

    ``payload`` is opaque to the worker; callers use it to keep the frames
    (color/depth/stamps) that belong to the inference result, so downstream
    processing never mixes a detection with newer sensor data.
    """

    def __init__(self, model, device: str = '', imgsz: Optional[int] = None,
                 conf: Optional[float] = None, logger=None):
        self._model = model
        self._device = device
        self._imgsz = imgsz
        self._conf = conf
        self._logger = logger
        self._cond = threading.Condition()
        self._pending: Optional[Tuple[Any, np.ndarray]] = None
        self._result: Optional[Tuple[Any, Any]] = None
        self._shutdown = False
        self._thread = threading.Thread(
            target=self._run, name='yolo-inference', daemon=True
        )
        self._thread.start()

    @property
    def model(self):
        return self._model

    def submit(self, payload: Any, image: np.ndarray) -> None:
        with self._cond:
            self._pending = (payload, image)
            self._cond.notify()

    def take_result(self) -> Optional[Tuple[Any, Any]]:
        """Return and clear the newest completed (payload, results) pair."""
        with self._cond:
            result = self._result
            self._result = None
            return result

    def stop(self) -> None:
        with self._cond:
            self._shutdown = True
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._shutdown:
                    self._cond.wait()
                if self._shutdown:
                    return
                payload, image = self._pending
                self._pending = None
            try:
                kwargs = {'verbose': False}
                if self._device:
                    kwargs['device'] = self._device
                if self._imgsz:
                    kwargs['imgsz'] = self._imgsz
                if self._conf is not None:
                    kwargs['conf'] = self._conf
                results = self._model(image, **kwargs)
            except Exception as exc:
                if self._logger is not None:
                    self._logger.error(
                        f'YOLO inference error: {exc}', throttle_duration_sec=5.0
                    )
                continue
            with self._cond:
                self._result = (payload, results)

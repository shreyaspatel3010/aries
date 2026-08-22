"""Pure-NumPy replacement for cv_bridge.

cv_bridge ships a C-extension (``cv_bridge_boost.so``) built against the system
NumPy 1.x ABI. Imported under a venv running NumPy 2.x it segfaults the process
(SIGSEGV) on the first image conversion. These helpers reimplement the small
slice of cv_bridge this project actually uses -- the bgr8/rgb8/mono8/16UC1/32FC1
encodings and ``passthrough`` -- with nothing but NumPy, so nodes can run on the
latest NumPy without rebuilding cv_bridge from source.
"""

import numpy as np
from sensor_msgs.msg import Image


class NumpyImageBridge:
    """Drop-in for the ``imgmsg_to_cv2`` / ``cv2_to_imgmsg`` calls we make."""

    # encoding -> (scalar dtype, channel count)
    _ENCODINGS = {
        'rgb8': (np.uint8, 3), 'bgr8': (np.uint8, 3),
        'rgba8': (np.uint8, 4), 'bgra8': (np.uint8, 4),
        'mono8': (np.uint8, 1), '8UC1': (np.uint8, 1), '8UC3': (np.uint8, 3),
        'mono16': (np.uint16, 1), '16UC1': (np.uint16, 1),
        '32FC1': (np.float32, 1),
    }

    def imgmsg_to_cv2(self, msg: Image, desired_encoding: str = 'passthrough') -> np.ndarray:
        src = msg.encoding
        if src not in self._ENCODINGS:
            raise ValueError(f'Unsupported image encoding: {src!r}')
        base, channels = self._ENCODINGS[src]
        dtype = np.dtype(base).newbyteorder('>' if msg.is_bigendian else '<')
        row_bytes = msg.width * channels * dtype.itemsize
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.step and msg.step > row_bytes:  # drop per-row padding
            raw = raw.reshape(msg.height, msg.step)[:, :row_bytes]
        raw = np.ascontiguousarray(raw).reshape(-1)
        arr = raw.view(dtype).reshape(msg.height, msg.width, channels)
        arr = arr.astype(base)  # copy into native byte order, writable
        if channels == 1:
            arr = arr[:, :, 0]
        return self._convert(arr, src, desired_encoding)

    def cv2_to_imgmsg(self, img: np.ndarray, encoding: str = 'passthrough') -> Image:
        img = np.ascontiguousarray(img)
        channels = 1 if img.ndim == 2 else int(img.shape[2])
        msg = Image()
        msg.height = int(img.shape[0])
        msg.width = int(img.shape[1])
        msg.encoding = encoding if encoding not in ('', 'passthrough') else self._infer_encoding(img, channels)
        msg.is_bigendian = 0
        msg.step = int(msg.width * channels * img.dtype.itemsize)
        msg.data = img.tobytes()
        return msg

    @staticmethod
    def _convert(arr: np.ndarray, src: str, desired: str) -> np.ndarray:
        if desired in ('', 'passthrough', src):
            return arr
        if desired == 'bgr8':
            if src == 'rgb8':
                return arr[:, :, ::-1].copy()
            if src in ('mono8', '8UC1'):
                return np.repeat(arr[:, :, None], 3, axis=2)
        elif desired == 'rgb8' and src == 'bgr8':
            return arr[:, :, ::-1].copy()
        elif desired in ('32FC1', '16UC1', 'mono8', 'mono16', '8UC1'):
            # single-channel numeric request matching source shape; hand back as-is
            return arr
        raise ValueError(f'Unsupported conversion {src!r} -> {desired!r}')

    @staticmethod
    def _infer_encoding(img: np.ndarray, channels: int) -> str:
        if channels == 1:
            return {'uint8': 'mono8', 'uint16': 'mono16',
                    'float32': '32FC1'}.get(str(img.dtype), 'passthrough')
        return {3: 'bgr8', 4: 'bgra8'}.get(channels, 'passthrough')

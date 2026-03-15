"""Webcam capture and async video recording utilities."""
import queue as _queue
import threading

import cv2
import numpy as np


class CameraThread:
    """Grab frames in a daemon thread; main loop calls read() to get the latest."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        ret, frame = cap.read()
        self._frame = frame
        self._ok = ret
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            ret, frame = self._cap.read()
            with self._lock:
                self._frame = frame
                self._ok = ret

    def read(self) -> tuple[bool, np.ndarray]:
        with self._lock:
            return self._ok, self._frame


class AsyncVideoWriter:
    """Drop-in wrapper around cv2.VideoWriter that writes on a background thread.

    Encodes and writes frames in a daemon thread so the main loop never blocks on
    VideoWriter.write() — important on slow CPUs (e.g. RPi 4) where encoding can
    take 20+ ms per frame.
    """

    def __init__(self, writer: cv2.VideoWriter, maxsize: int = 4):
        self._writer = writer
        self._q: _queue.Queue = _queue.Queue(maxsize=maxsize)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                break
            self._writer.write(frame)

    def write(self, frame: np.ndarray) -> None:
        """Queue a frame for writing. Drops frame if queue is full (keeps real-time)."""
        try:
            self._q.put_nowait(frame.copy())
        except _queue.Full:
            pass  # drop frame rather than stalling the main loop

    def release(self) -> None:
        self._q.put(None)   # sentinel to stop thread
        self._t.join(timeout=10)
        self._writer.release()

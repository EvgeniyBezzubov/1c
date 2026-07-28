#!/usr/bin/env python3
"""
Радарный обзор с вращающейся USB-камеры.

Камера делает 1 полный оборот за REVOLUTION_PERIOD секунд.
Кадры анализируются, объекты проецируются на схему «вид сверху»
в полярных координатах (как на радаре).

Запуск:
    python usb_radar_camera.py              # первая доступная камера
    python usb_radar_camera.py --camera 1   # конкретный индекс
    python usb_radar_camera.py --demo       # демо без камеры
    python usb_radar_camera.py --period 5   # период оборота, сек

Выход: клавиша Q или Esc.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

# --- параметры радара / камеры ---
REVOLUTION_PERIOD = 5.0  # секунд на полный оборот
CAMERA_FOV_DEG = 60.0  # горизонтальный угол обзора камеры
RADAR_SIZE = 640
MAX_RANGE = 1.0  # нормализованная дальность [0..1]
CONTACT_TTL = 6.0  # сколько секунд контакт живёт на радаре
MIN_CONTOUR_AREA = 400
SWEEP_TRAIL_DEG = 25.0


@dataclass
class Contact:
    """Точка на радаре: угол (рад), дальность [0..1], время обнаружения."""

    angle: float
    range_n: float
    timestamp: float
    strength: float = 1.0


class RotatingCameraRadar:
    def __init__(
        self,
        camera_index: int = 0,
        period: float = REVOLUTION_PERIOD,
        demo: bool = False,
        headless_frames: int = 0,
        output_path: Optional[str] = None,
    ) -> None:
        self.period = period
        self.demo = demo
        self.camera_index = camera_index
        self.headless_frames = headless_frames
        self.output_path = output_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=32, detectShadows=False
        )
        self.contacts: Deque[Contact] = deque(maxlen=4000)
        self.t0 = time.monotonic()
        self._demo_phase = 0.0

        # сглаженный кадр для стабильного вычитания фона при вращении
        self._prev_gray: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ camera
    def open(self) -> None:
        if self.demo:
            print("Режим демо: синтетический видеопоток.")
            return

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            # перебор нескольких индексов
            for idx in range(0, 6):
                if idx == self.camera_index:
                    continue
                trial = cv2.VideoCapture(idx)
                if trial.isOpened():
                    print(f"Камера {self.camera_index} недоступна, используем {idx}.")
                    self.cap.release()
                    self.cap = trial
                    self.camera_index = idx
                    break
                trial.release()

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                "Не удалось открыть USB-камеру. "
                "Проверьте подключение или запустите с --demo."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"USB-камера #{self.camera_index} подключена. "
              f"Оборот: {self.period:.1f} с.")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def read_frame(self) -> Optional[np.ndarray]:
        if self.demo:
            return self._synthetic_frame()
        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def _synthetic_frame(self) -> np.ndarray:
        """Имитация камеры, смотрящей наружу при вращении."""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (28, 28, 32)

        angle = self.current_angle()
        # «объекты» в мире: (угол_рад, дальность)
        world = [
            (0.4, 0.35),
            (1.8, 0.55),
            (3.5, 0.25),
            (4.9, 0.70),
            (5.6, 0.45),
        ]
        half_fov = math.radians(CAMERA_FOV_DEG / 2.0)
        for wa, wr in world:
            d = (wa - angle + math.pi) % (2 * math.pi) - math.pi
            if abs(d) > half_fov:
                continue
            # x — по азимуту внутри FOV, y — по дальности (ближе = ниже)
            x = int(w / 2 + (d / half_fov) * (w * 0.45))
            y = int(h * (0.25 + wr * 0.65))
            size = max(8, int(40 * (1.1 - wr)))
            color = (60, 180, 255) if wr < 0.5 else (80, 220, 120)
            cv2.rectangle(
                frame,
                (x - size, y - size),
                (x + size, y + size),
                color,
                -1,
            )
            cv2.rectangle(
                frame,
                (x - size, y - size),
                (x + size, y + size),
                (255, 255, 255),
                1,
            )

        # лёгкий шум
        noise = np.random.randint(0, 18, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        self._demo_phase += 0.02
        return frame

    # ------------------------------------------------------------------ angle
    def current_angle(self) -> float:
        """Текущий азимут камеры, рад [0..2π), растёт по часовой стрелке."""
        elapsed = time.monotonic() - self.t0
        return (elapsed % self.period) / self.period * 2.0 * math.pi

    def current_angle_deg(self) -> float:
        return math.degrees(self.current_angle()) % 360.0

    # ----------------------------------------------------------- detection
    def detect_objects(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """
        Возвращает список (угол_рад, дальность_норм, сила).

        Дальность оценивается по вертикали кадра:
        низ кадра ≈ близко, верх ≈ далеко (камера смотрит наружу).
        Азимут = угол вращения + смещение по горизонтали внутри FOV.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # при постоянном вращении MOG2 + разница кадров даёт устойчивые контуры
        fg = self.bg_subtractor.apply(gray)
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, diff_bin = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            fg = cv2.bitwise_or(fg, diff_bin)
        self._prev_gray = gray

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = gray.shape
        half_fov = math.radians(CAMERA_FOV_DEG / 2.0)
        base_angle = self.current_angle()
        results: List[Tuple[float, float, float]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw / 2.0
            cy = y + bh / 2.0

            # смещение азимута внутри поля зрения
            nx = (cx / w) * 2.0 - 1.0  # [-1..1]
            az = base_angle + nx * half_fov

            # дальность: низ = близко
            range_n = float(np.clip(cy / h, 0.05, 0.98))
            strength = float(np.clip(area / 5000.0, 0.3, 1.0))
            results.append((az % (2 * math.pi), range_n, strength))

        return results

    def update_contacts(self, detections: List[Tuple[float, float, float]]) -> None:
        now = time.monotonic()
        for angle, range_n, strength in detections:
            self.contacts.append(
                Contact(angle=angle, range_n=range_n, timestamp=now, strength=strength)
            )
        # удалить устаревшие
        while self.contacts and now - self.contacts[0].timestamp > CONTACT_TTL:
            self.contacts.popleft()

    # ------------------------------------------------------------- drawing
    def draw_camera_overlay(
        self, frame: np.ndarray, detections: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        for angle, range_n, strength in detections:
            # обратная проекция для подсветки (приблизительно)
            half_fov = math.radians(CAMERA_FOV_DEG / 2.0)
            d = (angle - self.current_angle() + math.pi) % (2 * math.pi) - math.pi
            if abs(d) > half_fov:
                continue
            x = int(w / 2 + (d / half_fov) * (w * 0.45))
            y = int(range_n * h)
            r = int(6 + 10 * strength)
            cv2.circle(out, (x, y), r, (0, 255, 255), 2)

        angle_deg = self.current_angle_deg()
        cv2.putText(
            out,
            f"Azimuth: {angle_deg:6.1f} deg | T={self.period:.1f}s/rev",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 180),
            1,
            cv2.LINE_AA,
        )
        return out

    def draw_radar(self) -> np.ndarray:
        size = RADAR_SIZE
        radar = np.zeros((size, size, 3), dtype=np.uint8)
        cx = cy = size // 2
        max_r = size // 2 - 16

        # фон
        radar[:] = (12, 18, 14)
        cv2.circle(radar, (cx, cy), max_r, (20, 45, 30), -1)

        # кольца дальности
        for k in (0.25, 0.5, 0.75, 1.0):
            cv2.circle(radar, (cx, cy), int(max_r * k), (40, 90, 55), 1)

        # азимутальные лучи каждые 30°
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            x2 = int(cx + max_r * math.sin(rad))
            y2 = int(cy - max_r * math.cos(rad))
            cv2.line(radar, (cx, cy), (x2, y2), (35, 70, 45), 1)

        # подписи направлений
        for deg, label in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            rad = math.radians(deg)
            tx = int(cx + (max_r + 10) * math.sin(rad) - 6)
            ty = int(cy - (max_r + 10) * math.cos(rad) + 5)
            cv2.putText(
                radar, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (90, 200, 120), 1, cv2.LINE_AA,
            )

        now = time.monotonic()
        sweep = self.current_angle()

        # след луча (затухание)
        trail_steps = 40
        for i in range(trail_steps):
            a = sweep - math.radians(SWEEP_TRAIL_DEG) * (i / trail_steps)
            alpha = 1.0 - i / trail_steps
            x2 = int(cx + max_r * math.sin(a))
            y2 = int(cy - max_r * math.cos(a))
            color = (
                int(30 * alpha),
                int(180 * alpha),
                int(60 * alpha),
            )
            cv2.line(radar, (cx, cy), (x2, y2), color, 2 if i == 0 else 1)

        # контакты
        for c in self.contacts:
            age = now - c.timestamp
            if age > CONTACT_TTL:
                continue
            fade = 1.0 - age / CONTACT_TTL
            fade *= c.strength
            r_px = int(c.range_n * max_r)
            x = int(cx + r_px * math.sin(c.angle))
            y = int(cy - r_px * math.cos(c.angle))
            radius = max(2, int(3 + 5 * fade))
            color = (
                int(40 * fade),
                int(255 * fade),
                int(80 * fade),
            )
            cv2.circle(radar, (x, y), radius, color, -1)
            if fade > 0.55:
                cv2.circle(radar, (x, y), radius + 3, (80, 255, 180), 1)

        # центр (камера)
        cv2.circle(radar, (cx, cy), 5, (0, 255, 255), -1)
        cv2.circle(radar, (cx, cy), 8, (0, 200, 200), 1)

        cv2.putText(
            radar,
            "TOP-DOWN RADAR",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 160),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            radar,
            f"contacts: {len(self.contacts)}  az: {self.current_angle_deg():.0f}",
            (12, size - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 200, 130),
            1,
            cv2.LINE_AA,
        )
        return radar

    # ----------------------------------------------------------------- main
    def compose_view(
        self, frame: np.ndarray, detections: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        cam_view = self.draw_camera_overlay(frame, detections)
        radar_view = self.draw_radar()
        rh = radar_view.shape[0]
        scale = rh / cam_view.shape[0]
        cam_resized = cv2.resize(cam_view, (int(cam_view.shape[1] * scale), rh))
        return np.hstack([cam_resized, radar_view])

    def run(self) -> None:
        self.open()
        frame_i = 0
        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    print("Кадр не получен, выход.")
                    break

                detections = self.detect_objects(frame)
                self.update_contacts(detections)
                combined = self.compose_view(frame, detections)
                frame_i += 1

                if self.headless_frames > 0:
                    if frame_i >= self.headless_frames:
                        if self.output_path:
                            cv2.imwrite(self.output_path, combined)
                            print(
                                f"Сохранён кадр: {self.output_path} "
                                f"(контактов: {len(self.contacts)})"
                            )
                        break
                    # имитация реального FPS в headless
                    time.sleep(1.0 / 30.0)
                    continue

                cv2.imshow("USB Camera Radar (Q/Esc — выход)", combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
        finally:
            self.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Схематичный радар (вид сверху) с вращающейся USB-камеры"
    )
    p.add_argument(
        "--camera", "-c", type=int, default=0,
        help="индекс USB-камеры (по умолчанию 0)",
    )
    p.add_argument(
        "--period", "-p", type=float, default=REVOLUTION_PERIOD,
        help="период полного оборота камеры в секундах (по умолчанию 5)",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="синтетическое видео без реальной камеры",
    )
    p.add_argument(
        "--headless-frames", type=int, default=0,
        help="без окна: обработать N кадров и выйти (для тестов)",
    )
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="путь для сохранения итогового кадра в headless-режиме",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.period <= 0:
        print("Период оборота должен быть > 0", file=sys.stderr)
        return 1
    app = RotatingCameraRadar(
        camera_index=args.camera,
        period=args.period,
        demo=args.demo,
        headless_frames=args.headless_frames,
        output_path=args.output,
    )
    try:
        app.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

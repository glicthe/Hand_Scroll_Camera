"""
Hand scroll + zoom.

Tracks one hand through the webcam.

SCROLL (two modes, both scroll up/down and sideways):
  palm     the page follows the movement of the palm center
  fingers  the page follows the movement of the index and middle fingertips,
           and only scrolls on an axis when both move together in the same
           direction on that axis. Ring finger, pinky and thumb are ignored.
The page follows the hand:
  hand moves up    -> content moves up    (scrolls down)
  hand moves down  -> content moves down  (scrolls up)
  hand moves right -> content moves right (scrolls left)
  hand moves left  -> content moves left  (scrolls right)
When one axis moves much more than the other, only that axis scrolls, so a
vertical movement does not leak sideways. Diagonal movement scrolls both.

Switch mode:
  --modePalm or --modeFingers at launch
  press m in the debug window, or type m, modePalm or modeFingers in the
  terminal and press Enter

PAUSE: make a fist to pause tracking. Open the hand to resume. The page does
not jump on resume, tracking re-anchors at the new hand position.
  palm mode:    fist = 3 or more of the 4 fingers curled
  fingers mode: fist = index and middle both curled, open = both extended
                (ring finger and pinky may stay curled)

ZOOM: pinch thumb and index finger together, then release.
  quick pinch (held under 5 s) then release -> zoom in
  pinch held 5 s or more, then release      -> zoom out
The zoom is decided at release, because the hold time is only known then.
Scrolling is frozen while pinching.
If the thumb cannot be seen (folded away, hidden behind the hand, or tracked
collapsed onto the palm), pinch detection is off, a running pinch is cancelled,
and the pinch no longer freezes scrolling. This applies in both modes.

Run (flags are single words, no spaces, and can be combined):
  python hand_scroll.py                   camera hidden, palm mode
  python hand_scroll.py --modeFingers     start in two-finger mode
  python hand_scroll.py --modePalm        start in palm mode
  python hand_scroll.py --debug           preview window, always on top
  python hand_scroll.py --invert          flip scroll direction (both axes)
  python hand_scroll.py --invertSideways  flip only the sideways direction
  python hand_scroll.py --noSideways      up/down scroll only
  python hand_scroll.py --zoomWheel       zoom with Ctrl+wheel instead of Ctrl +/-
  python hand_scroll.py --camera1         use camera index 1 (default 0)
  python hand_scroll.py --silent          no start / stop beeps
  python hand_scroll.py --modeFingers --debug

Run without a terminal (Windows): python create_shortcuts.py makes desktop
shortcuts that launch this script with pythonw.exe (no console window).
  --toggle      start if not running, stop if running (use this in the shortcut)
  --stop        stop the running instance
  --switchMode  switch palm / fingers mode of the running instance
Without a console, output goes to hand_scroll.log next to this script.
Only one instance can run at a time. A high beep means started, a low beep
means stopped.

Stop: Ctrl+C in the terminal, q / Esc in the focused debug window, or --stop.
"""

import argparse
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time

# No console (pythonw.exe, shortcut launch): sys.stdout and sys.stderr are None.
# Send them to a log file next to this script so errors are not lost.
if sys.stdout is None or sys.stderr is None:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_scroll.log")
    _log_mode = "w" if os.path.exists(_log_path) and os.path.getsize(_log_path) > 1_000_000 else "a"
    _log = open(_log_path, _log_mode, buffering=1, encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

CONTROL_PORT = 47653   # localhost-only port used to talk to a running instance


def send_command(cmd):
    """Send a command to a running instance. True if it answered ok."""
    try:
        with socket.create_connection(("127.0.0.1", CONTROL_PORT), timeout=1.0) as s:
            s.sendall(cmd.encode() + b"\n")
            return s.recv(16).strip() == b"ok"
    except OSError:
        return False


def early_commands(argv):
    """--stop, --toggle and --switchMode talk to a running instance. They run
    before the slow cv2 / mediapipe imports so they answer instantly."""
    if "--switchMode" in argv:
        print("mode switched" if send_command("mode") else "not running")
        sys.exit(0)
    if "--stop" in argv:
        print("stopped" if send_command("stop") else "not running")
        sys.exit(0)
    if "--toggle" in argv and send_command("stop"):
        print("stopped")
        sys.exit(0)
    # --toggle with nothing running falls through and starts normally


if __name__ == "__main__":
    early_commands(sys.argv[1:])

import cv2
import mediapipe as mp
import pyautogui

# ---------------- Tuning ----------------
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# scroll
SCROLL_GAIN = 30.0     # notches scrolled up/down per full frame height of hand travel
SIDE_GAIN = 30.0       # notches scrolled sideways per full frame width of hand travel
SMOOTHING = 0.7        # 0..1, higher = more responsive, lower = smoother
DEADZONE = 0.004       # min hand movement (fraction of frame size) before scrolling
MAX_STEP = 0.15        # clamp per-update movement, filters tracking glitches
AXIS_DOMINANCE = 1.5   # if one axis moves this many times more than the other,
                       # only that axis scrolls (stops vertical moves leaking sideways)
COHERENCE = 0.35       # fingers mode: both fingertips must move in the same direction
                       # by at least this fraction of their average movement

# fist / open (palm mode)
FIST_CURLED = 4        # curled fingers (of 4) that count as a fist
OPEN_CURLED = 2        # curled fingers (of 4) or fewer that count as an open hand
STATE_FRAMES = 2       # consecutive frames needed to switch tracking <-> paused

# pinch / zoom (ratio = thumb-index tip distance / palm length)
PINCH_CLOSE = 0.25     # below this = pinched
PINCH_OPEN = 0.45      # above this = released (gap vs PINCH_CLOSE prevents flicker)
PINCH_FREEZE = 0.35    # scrolling is frozen while ratio is below this
PINCH_FRAMES = 2       # consecutive frames to confirm a pinch
RELEASE_FRAMES = 2     # consecutive frames to confirm a release
ZOOM_OUT_HOLD = 0.5    # seconds a pinch must be held for the release to zoom out
ZOOM_COOLDOWN = 0.2    # min seconds between zoom actions

# thumb visibility. MediaPipe always outputs a thumb, even when it is hidden,
# so "cannot be seen" is estimated from these two measurements.
THUMB_MIN_LENGTH = 0.30   # thumb tip to thumb base distance / palm length; below = folded or hidden
THUMB_MAX_BEHIND = 0.05   # thumb tip depth beyond the palm center; above = behind the hand
THUMB_HIDDEN_FRAMES = 3   # consecutive hidden frames before a running pinch is cancelled

DEBUG_SIZE = (360, 270)
WINDOW = "hand_scroll debug"
# ----------------------------------------

IS_WIN = sys.platform.startswith("win")
NOTCH = 180 if IS_WIN else 1       # Windows wheel units: 120 per notch
MIN_EMIT = 20 if IS_WIN else 1     # smallest scroll event sent
MOD_KEY = "command" if sys.platform == "darwin" else "ctrl"

MODES = ("palm", "fingers")
PALM_IDS = (0, 5, 9, 13, 17)                        # wrist + finger bases
SCROLL_TIPS = (8, 12)                               # index and middle fingertips (fingers mode)
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))    # (tip, pip): index..pinky

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def palm_center(pts):
    x = sum(pts[i].x for i in PALM_IDS) / len(PALM_IDS)
    y = sum(pts[i].y for i in PALM_IDS) / len(PALM_IDS)
    return x, y


def palm_length(pts, aspect):
    """Wrist to middle finger base, used to normalize distances."""
    size = math.hypot((pts[0].x - pts[9].x) * aspect, pts[0].y - pts[9].y)
    return max(size, 1e-6)


def curled_flags(pts, aspect):
    """[index, middle, ring, pinky] booleans. A finger is curled when its tip
    is closer to the wrist than its middle joint. Thumb is ignored."""
    wx, wy = pts[0].x * aspect, pts[0].y
    flags = []
    for tip, pip in FINGERS:
        d_tip = math.hypot(pts[tip].x * aspect - wx, pts[tip].y - wy)
        d_pip = math.hypot(pts[pip].x * aspect - wx, pts[pip].y - wy)
        flags.append(d_tip < d_pip)
    return flags


def pinch_ratio(pts, aspect):
    """Thumb tip to index tip distance, divided by palm length."""
    d = math.hypot((pts[4].x - pts[8].x) * aspect, pts[4].y - pts[8].y)
    return d / palm_length(pts, aspect)


def thumb_metrics(pts, aspect):
    """(length, depth).
    length: thumb tip to thumb base (landmark 2) over palm length. Small when
            the thumb is folded, foreshortened, or collapsed onto the palm.
    depth:  thumb tip z minus palm center z. Positive means behind the palm."""
    d = math.hypot((pts[4].x - pts[2].x) * aspect, pts[4].y - pts[2].y)
    palm_z = sum(pts[i].z for i in PALM_IDS) / len(PALM_IDS)
    return d / palm_length(pts, aspect), pts[4].z - palm_z


def moving_together(deltas, mean):
    """True when every tracked point moved in the direction of the mean, by at
    least COHERENCE * |mean|. Always True for a single point."""
    if len(deltas) == 1:
        return True
    sign = 1 if mean > 0 else -1
    floor = abs(mean) * COHERENCE
    return all(d * sign >= floor for d in deltas)


class Axis:
    """Follows one axis (x or y) of one or more tracked points."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.smooth = None     # None means "no anchor, set one on the next step"
        self.last = None
        self.accum = 0.0

    def step(self, vals):
        """First call anchors and returns None. Later calls return
        (mean_delta, deltas) once movement since the last update passes the
        deadzone, otherwise None."""
        if self.smooth is None or len(self.smooth) != len(vals):
            self.smooth = list(vals)
            self.last = list(vals)
            self.accum = 0.0
            return None
        self.smooth = [s + SMOOTHING * (v - s) for s, v in zip(self.smooth, vals)]
        deltas = [s - l for s, l in zip(self.smooth, self.last)]
        mean = sum(deltas) / len(deltas)
        if abs(mean) < DEADZONE:
            return None
        self.last = list(self.smooth)   # re-anchor whether or not it scrolls
        return mean, deltas

    def emit(self, mean, gain):
        """Accumulate movement and return whole scroll units to send (0 if
        the amount is still below MIN_EMIT)."""
        mean = max(-MAX_STEP, min(MAX_STEP, mean))
        self.accum += mean * gain * NOTCH
        amount = int(self.accum)
        if abs(amount) < MIN_EMIT:
            return 0
        self.accum -= amount
        return amount


def follow_motion(axis_x, axis_y, xs, ys, sideways):
    """Advance both axes. Returns (move_x, move_y), each None or (mean, deltas).
    When one axis moves much more than the other, only the dominant axis is
    returned, so a vertical movement does not leak into sideways scrolling."""
    move_y = axis_y.step(ys)
    move_x = axis_x.step(xs) if sideways else None
    if move_x and move_y:
        if abs(move_x[0]) > AXIS_DOMINANCE * abs(move_y[0]):
            move_y = None
        elif abs(move_y[0]) > AXIS_DOMINANCE * abs(move_x[0]):
            move_x = None
    return move_x, move_y


def hscroll(amount):
    """Positive scrolls right, negative scrolls left."""
    try:
        pyautogui.hscroll(amount)
    except Exception:
        # platforms without native horizontal scroll: Shift + wheel
        pyautogui.keyDown("shift")
        try:
            pyautogui.scroll(-amount)
        finally:
            pyautogui.keyUp("shift")


def zoom(direction, use_wheel):
    """direction: +1 zoom in, -1 zoom out."""
    if use_wheel:
        pyautogui.keyDown(MOD_KEY)
        try:
            pyautogui.scroll(direction * NOTCH)
        finally:
            pyautogui.keyUp(MOD_KEY)
    else:
        pyautogui.hotkey(MOD_KEY, "=" if direction > 0 else "-")


def toggle_mode(state):
    state["mode"] = MODES[(MODES.index(state["mode"]) + 1) % len(MODES)]
    print(f"mode: {state['mode']}")


def beep(started):
    """High beep = started, low beep = stopped. Feedback when there is no window."""
    if not IS_WIN:
        return
    try:
        import winsound
        winsound.Beep(880 if started else 440, 120)
    except Exception:
        pass


def start_control_server(state):
    """Listen on localhost so --toggle, --stop and --switchMode can reach this
    instance. Returns None when the port is taken, meaning another instance is
    already running."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if not IS_WIN:
        # POSIX: lets a restart bind while the old port sits in TIME_WAIT.
        # Not set on Windows, where it would allow two instances to bind.
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("127.0.0.1", CONTROL_PORT))
    except OSError:
        srv.close()
        return None
    srv.listen(2)

    def loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return   # server closed
            with conn:
                try:
                    conn.settimeout(1.0)
                    cmd = conn.recv(64).strip().lower()
                    if cmd == b"stop":
                        state["quit"] = True
                        conn.sendall(b"ok\n")
                    elif cmd == b"mode":
                        toggle_mode(state)
                        conn.sendall(b"ok\n")
                    # Wait for the client to close first. The side that closes first
                    # keeps the socket in TIME_WAIT, so this keeps it off the control
                    # port and a restart right after a stop can always bind.
                    conn.recv(1)
                except OSError:
                    pass

    threading.Thread(target=loop, daemon=True).start()
    return srv


def stdin_listener(state):
    """Terminal commands (then Enter): m toggles, modePalm and modeFingers set the mode."""
    try:
        if sys.stdin is None:
            return
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in ("m", "mode"):
                toggle_mode(state)
            elif cmd == "modepalm":
                state["mode"] = "palm"
                print("mode: palm")
            elif cmd == "modefingers":
                state["mode"] = "fingers"
                print("mode: fingers")
    except Exception:
        pass


def make_topmost(title):
    """Keep the debug window above all other windows."""
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    if IS_WIN:
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
            user32.FindWindowW.restype = wintypes.HWND
            user32.SetWindowPos.argtypes = [
                wintypes.HWND, wintypes.HWND,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                wintypes.UINT,
            ]
            hwnd = user32.FindWindowW(None, title)
            if hwnd:
                HWND_TOPMOST = wintypes.HWND(-1)
                # SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE
                user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            subprocess.run(["wmctrl", "-r", title, "-b", "add,above"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except FileNotFoundError:
            pass


def parse_args(argv):
    """All flags are single words, no spaces. --camera<N> picks the camera index."""
    parser = argparse.ArgumentParser(
        description="Hand scroll + zoom",
        epilog="Camera index: --camera0, --camera1, ... (default %d)" % CAMERA_INDEX,
    )
    parser.add_argument("--modePalm", dest="mode", action="store_const", const="palm",
                        help="scroll by palm center (default)")
    parser.add_argument("--modeFingers", dest="mode", action="store_const", const="fingers",
                        help="scroll by index and middle fingertips moving together")
    parser.add_argument("--debug", action="store_true", help="show camera preview")
    parser.add_argument("--invert", action="store_true", help="invert scroll direction (both axes)")
    parser.add_argument("--invertSideways", dest="invert_sideways", action="store_true",
                        help="invert only the sideways scroll direction")
    parser.add_argument("--noSideways", dest="no_sideways", action="store_true",
                        help="disable sideways scrolling")
    parser.add_argument("--zoomWheel", dest="zoom_wheel", action="store_true",
                        help="zoom with Ctrl+wheel instead of Ctrl +/-")
    parser.add_argument("--silent", action="store_true", help="no start / stop beeps")
    parser.add_argument("--toggle", action="store_true",
                        help="start, or stop the running instance (for shortcuts)")
    parser.add_argument("--stop", action="store_true", help="stop the running instance")
    parser.add_argument("--switchMode", dest="switch_mode", action="store_true",
                        help="switch palm / fingers mode of the running instance")
    parser.set_defaults(mode="palm")

    camera = CAMERA_INDEX
    rest = []
    for a in argv:
        m = re.fullmatch(r"--camera(\d+)", a)
        if m:
            camera = int(m.group(1))
        else:
            rest.append(a)

    args = parser.parse_args(rest)
    args.camera = camera
    return args


def main():
    args = parse_args(sys.argv[1:])

    state = {"mode": args.mode, "quit": False}
    server = start_control_server(state)
    if server is None:
        print("Already running. Use --stop or --toggle to stop it.")
        return

    backend = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("Cannot open camera. Try --camera1.")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    if args.debug:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, *DEBUG_SIZE)

    threading.Thread(target=stdin_listener, args=(state,), daemon=True).start()
    prev_mode = state["mode"]

    # fist / open state
    tracking = True
    fist_frames = 0
    open_frames = 0

    # scroll state
    axis_x = Axis()
    axis_y = Axis()
    sideways = not args.no_sideways
    last_scroll = "-"
    last_side = "-"
    sync = True
    direction = -1 if args.invert else 1
    side_direction = -direction if args.invert_sideways else direction

    # pinch / zoom state
    pinched = False
    pinch_frames = 0
    release_frames = 0
    pinch_start = 0.0
    last_zoom_time = 0.0
    last_zoom = "-"

    # thumb visibility
    thumb_hidden_frames = 0
    thumb_ok = True
    thumb_len = 0.0
    thumb_depth = 0.0

    curled = 0
    ratio = 1.0
    frame_idx = 0

    print(f"Running, mode: {state['mode']}. Ctrl+C to stop. "
          "Terminal commands: m, modePalm, modeFingers (then Enter)."
          + (" Debug window: m switches mode, q or Esc quits." if args.debug else ""))
    if not args.silent:
        beep(True)

    try:
        while True:
            if state["quit"]:
                break
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)     # mirror: x grows toward the user's right
            h, w = frame.shape[:2]
            aspect = w / h
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            now = time.time()

            mode = state["mode"]
            if mode != prev_mode:
                prev_mode = mode
                axis_x.reset()
                axis_y.reset()
                sync = True

            status, color = "NO HAND", (160, 160, 160)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                pts = lm.landmark
                px, py = palm_center(pts)
                flags = curled_flags(pts, aspect)
                curled = sum(flags)
                others_curled = sum(flags[1:])
                ratio = pinch_ratio(pts, aspect)

                # ---- thumb visibility ----
                thumb_len, thumb_depth = thumb_metrics(pts, aspect)
                thumb_visible = thumb_len >= THUMB_MIN_LENGTH and thumb_depth <= THUMB_MAX_BEHIND
                thumb_hidden_frames = 0 if thumb_visible else thumb_hidden_frames + 1
                thumb_ok = thumb_hidden_frames < THUMB_HIDDEN_FRAMES

                # ---- fist / open state with debounce ----
                if mode == "fingers":
                    fist_now = flags[0] and flags[1]                  # index + middle curled
                    open_now = not flags[0] and not flags[1]          # index + middle extended
                    pinch_gate = not flags[1]                         # middle extended
                else:
                    fist_now = curled >= FIST_CURLED
                    open_now = curled <= OPEN_CURLED
                    pinch_gate = others_curled <= 1

                if fist_now:
                    fist_frames += 1
                    open_frames = 0
                elif open_now:
                    open_frames += 1
                    fist_frames = 0

                if tracking and fist_frames >= STATE_FRAMES:
                    tracking = False
                elif not tracking and open_frames >= STATE_FRAMES:
                    tracking = True

                # ---- pinch / zoom ----
                if not tracking or not thumb_ok:
                    # fist, or thumb hidden: no pinch, cancel any running pinch
                    pinched = False
                    pinch_frames = release_frames = 0
                elif not pinched:
                    # start only with the thumb visible and the pinch gate satisfied,
                    # so closing a fist is not read as a pinch
                    if thumb_visible and ratio < PINCH_CLOSE and pinch_gate:
                        pinch_frames += 1
                        if pinch_frames >= PINCH_FRAMES:
                            pinched = True
                            pinch_start = now
                            release_frames = 0
                    else:
                        pinch_frames = 0
                else:
                    if ratio > PINCH_OPEN:
                        release_frames += 1
                        if release_frames >= RELEASE_FRAMES:
                            held = now - pinch_start
                            pinched = False
                            pinch_frames = release_frames = 0
                            if now - last_zoom_time >= ZOOM_COOLDOWN:
                                if held >= ZOOM_OUT_HOLD:
                                    zoom(-1, args.zoom_wheel)
                                    last_zoom = f"OUT ({held:.1f}s)"
                                else:
                                    zoom(+1, args.zoom_wheel)
                                    last_zoom = f"IN ({held:.1f}s)"
                                last_zoom_time = now
                                print("zoom " + last_zoom)
                    else:
                        release_frames = 0

                # ---- scroll ----
                # with the thumb hidden the pinch ratio is meaningless, so it does not freeze scrolling
                pinching = thumb_ok and (pinched or ratio <= PINCH_FREEZE)
                scrolling = tracking and not pinching and open_now

                if scrolling:
                    status, color = "TRACKING", (0, 220, 0)
                    if mode == "palm":
                        xs, ys = [px], [py]
                    else:
                        xs = [pts[i].x for i in SCROLL_TIPS]
                        ys = [pts[i].y for i in SCROLL_TIPS]

                    move_x, move_y = follow_motion(axis_x, axis_y, xs, ys, sideways)
                    checks = []

                    if move_y:
                        together = moving_together(move_y[1], move_y[0])
                        checks.append(together)
                        if together:
                            # hand down -> positive -> scroll up
                            amount = axis_y.emit(move_y[0], SCROLL_GAIN * direction)
                            if amount:
                                pyautogui.scroll(amount)
                                last_scroll = f"{amount:+d}"

                    if move_x:
                        together = moving_together(move_x[1], move_x[0])
                        checks.append(together)
                        if together:
                            # hand right -> content right -> scroll left (negative)
                            amount = axis_x.emit(move_x[0], -SIDE_GAIN * side_direction)
                            if amount:
                                hscroll(amount)
                                last_side = f"{amount:+d}"

                    if checks:
                        sync = all(checks)
                else:
                    axis_x.reset()
                    axis_y.reset()
                    if not tracking:
                        status, color = "PAUSED", (0, 0, 230)
                    elif pinched:
                        status, color = f"PINCH {now - pinch_start:.1f}s", (0, 165, 255)
                    else:
                        status, color = "HOLD", (0, 220, 220)

                if args.debug:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    if mode == "palm":
                        cv2.circle(frame, (int(px * w), int(py * h)), 8, color, -1)
                    else:
                        for i in SCROLL_TIPS:
                            cv2.circle(frame, (int(pts[i].x * w), int(pts[i].y * h)), 8, color, -1)
                    if thumb_ok:
                        p4 = (int(pts[4].x * w), int(pts[4].y * h))
                        p8 = (int(pts[8].x * w), int(pts[8].y * h))
                        cv2.line(frame, p4, p8, (0, 165, 255) if pinched else (255, 255, 255), 2)
            else:
                # hand lost: reset everything, cancel any pending pinch
                tracking = True
                fist_frames = open_frames = 0
                axis_x.reset()
                axis_y.reset()
                pinched = False
                pinch_frames = release_frames = 0
                thumb_hidden_frames = 0
                thumb_ok = True
                thumb_len = thumb_depth = 0.0
                curled = 0
                ratio = 1.0

            if args.debug:
                cv2.putText(frame, status, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                mode_line = f"mode: {mode}"
                if mode == "fingers":
                    mode_line += f"  together: {'yes' if sync else 'no'}"
                side_text = last_side if sideways else "off"
                cv2.putText(frame, f"{mode_line}  v: {last_scroll}  h: {side_text}", (10, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(frame, f"curled: {curled}/4  pinch: {ratio:.2f}  zoom: {last_zoom}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                thumb_color = (0, 255, 255) if thumb_ok else (0, 0, 230)
                cv2.putText(frame,
                            f"thumb: {'ok' if thumb_ok else 'HIDDEN'}  len {thumb_len:.2f}  z {thumb_depth:+.2f}",
                            (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, thumb_color, 2)
                cv2.imshow(WINDOW, frame)
                if frame_idx % 150 == 0:
                    make_topmost(WINDOW)
                frame_idx += 1
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("m"):
                    toggle_mode(state)
    except KeyboardInterrupt:
        pass
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        try:
            server.shutdown(socket.SHUT_RDWR)   # wakes the accept loop so it stops listening
        except OSError:
            pass
        server.close()
        if not args.silent:
            beep(False)


if __name__ == "__main__":
    main()
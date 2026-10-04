"""
Create desktop shortcuts for hand_scroll.py (Windows).

Run once, from the same Python environment that has mediapipe installed:
  python create_shortcuts.py
  python create_shortcuts.py --modeFingers
  python create_shortcuts.py --modeFingers --debug --silent

Any flags you pass are saved into the toggle shortcut.

Two shortcuts are created on the Desktop:
  Hand Scroll        Ctrl+Alt+H   starts the program, press again to stop it
  Hand Scroll Mode   Ctrl+Alt+M   switches palm / fingers mode while running

They launch with pythonw.exe, so no terminal window opens. The hotkeys work
because the shortcuts are on the Desktop (Windows only honors shortcut hotkeys
for shortcuts on the Desktop or in the Start Menu).
"""

import os
import subprocess
import sys

TOGGLE_NAME = "Hand Scroll"
TOGGLE_HOTKEY = "CTRL+ALT+H"
MODE_NAME = "Hand Scroll Mode"
MODE_HOTKEY = "CTRL+ALT+M"

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "hand_scroll.py")


def pythonw_path():
    """pythonw.exe next to the current interpreter, so the same packages are used."""
    exe = sys.executable
    candidate = os.path.join(os.path.dirname(exe), "pythonw.exe")
    return candidate if os.path.exists(candidate) else exe


def ps_quote(text):
    """Quote a string for a PowerShell single-quoted literal."""
    return "'" + text.replace("'", "''") + "'"


def build_script(lnk_path, target, arguments, workdir, hotkey, description):
    return "\n".join([
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(%s)" % ps_quote(lnk_path),
        "$s.TargetPath = %s" % ps_quote(target),
        "$s.Arguments = %s" % ps_quote(arguments),
        "$s.WorkingDirectory = %s" % ps_quote(workdir),
        "$s.Hotkey = %s" % ps_quote(hotkey),
        "$s.Description = %s" % ps_quote(description),
        "$s.Save()",
    ])


def desktop_dir():
    """Ask Windows for the Desktop folder (it may be redirected, for example to OneDrive)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def make_shortcut(lnk_path, arguments, hotkey, description):
    script = build_script(lnk_path, pythonw_path(), arguments, HERE, hotkey, description)
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)


def main():
    if not sys.platform.startswith("win"):
        print("This script only creates Windows shortcuts.")
        return
    if not os.path.exists(SCRIPT):
        print("hand_scroll.py not found next to this script.")
        return

    extra = " ".join(sys.argv[1:])
    desktop = desktop_dir()

    toggle_args = ('"%s" --toggle %s' % (SCRIPT, extra)).strip()
    mode_args = '"%s" --switchMode' % SCRIPT

    toggle_lnk = os.path.join(desktop, TOGGLE_NAME + ".lnk")
    mode_lnk = os.path.join(desktop, MODE_NAME + ".lnk")

    make_shortcut(toggle_lnk, toggle_args, TOGGLE_HOTKEY, "Start or stop hand scroll")
    make_shortcut(mode_lnk, mode_args, MODE_HOTKEY, "Switch hand scroll mode")

    print("Created:")
    print("  %s  (%s)  ->  pythonw %s" % (toggle_lnk, TOGGLE_HOTKEY, toggle_args))
    print("  %s  (%s)  ->  pythonw %s" % (mode_lnk, MODE_HOTKEY, mode_args))


if __name__ == "__main__":
    main()
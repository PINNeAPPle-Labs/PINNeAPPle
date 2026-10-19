"""The screen GUI applications run on: a virtual X display (Xvfb) on a Linux server or container, screenshots of it,
and the three ways of driving a GUI when an application has no usable API.

* :class:`VirtualDisplay` starts Xvfb on a free display number and captures the whole screen with Pillow
  (``ImageGrab.grab(xdisplay=...)``), so no screenshot utility is needed.
* :class:`XdotoolDriver` finds, raises and types into X11 windows with ``xdotool``.
* :class:`PlaywrightDriver` drives a web UI (a solver's web front end, a licence portal) with Playwright.
* :class:`PywinautoDriver` is the interface for Windows desktop applications (Ansys Workbench, Abaqus/CAE, ...)
  through ``pywinauto``. It runs only on Windows: in the Linux container it reports itself unavailable.

API control (the application's own Python, the solver's command line) always comes first; these drivers are the
fallback and the observer.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["VirtualDisplay", "XdotoolDriver", "PlaywrightDriver", "PywinautoDriver", "GuiUnavailable"]


class GuiUnavailable(RuntimeError):
    """The GUI driver cannot run here (missing tool, wrong platform, no display)."""


class VirtualDisplay:
    """An Xvfb display for the duration of a ``with`` block (or until :meth:`stop`).

    If ``DISPLAY`` already points at a running X server and ``reuse=True``, that one is used instead of starting a
    new one (a desktop session, or a container that started Xvfb + noVNC itself)."""

    def __init__(self, width: int = 1600, height: int = 1000, depth: int = 24, reuse: bool = False):
        self.size = (width, height, depth)
        self.reuse = reuse
        self.display: str | None = None
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def available() -> tuple[bool, str]:
        if shutil.which("Xvfb"):
            return True, shutil.which("Xvfb") or ""
        return False, "Xvfb is not installed (apt install xvfb)"

    def start(self) -> VirtualDisplay:
        if self.reuse and os.environ.get("DISPLAY"):
            self.display = os.environ["DISPLAY"]
            return self
        ok, why = self.available()
        if not ok:
            raise GuiUnavailable(why)
        w, h, d = self.size
        for n in range(90, 200):
            if Path(f"/tmp/.X{n}-lock").exists() or Path(f"/tmp/.X11-unix/X{n}").exists():
                continue
            proc = subprocess.Popen(["Xvfb", f":{n}", "-screen", "0", f"{w}x{h}x{d}", "-nolisten", "tcp"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(50):
                if Path(f"/tmp/.X11-unix/X{n}").exists():
                    self.display, self._proc = f":{n}", proc
                    return self
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            proc.kill()
        raise GuiUnavailable("could not start Xvfb on any display :90-:199")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def __enter__(self) -> VirtualDisplay:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        if self.display:
            env["DISPLAY"] = self.display
        return env

    def screenshot(self, path: str | Path, box: tuple[int, int, int, int] | None = None) -> Path:
        """Capture the display (or ``box`` = left, top, right, bottom) to a PNG/JPEG file."""
        if not self.display:
            raise GuiUnavailable("the display is not started")
        from PIL import ImageGrab
        img = ImageGrab.grab(bbox=box, xdisplay=self.display)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(path, quality=88) if path.suffix.lower() in (".jpg", ".jpeg") else img.save(path)
        return path


class XdotoolDriver:
    """Find and act on X11 windows. Every call is recorded in :attr:`log` (the commands, for the evidence report)."""

    def __init__(self, display: VirtualDisplay):
        self.display = display
        self.log: list[str] = []

    @staticmethod
    def available() -> tuple[bool, str]:
        return (True, shutil.which("xdotool") or "") if shutil.which("xdotool") else \
            (False, "xdotool is not installed (apt install xdotool)")

    def _run(self, *args: str, check: bool = True) -> str:
        cmd = ["xdotool", *args]
        self.log.append(" ".join(cmd))
        r = subprocess.run(cmd, env=self.display.env(), capture_output=True, text=True, timeout=30)
        if check and r.returncode != 0:
            raise GuiUnavailable(f"{' '.join(cmd)} failed: {r.stderr.strip() or r.returncode}")
        return r.stdout.strip()

    def wait_window(self, name: str, timeout_s: float = 60.0) -> str:
        """Id of the first visible window whose title matches the regular expression ``name``."""
        end = time.time() + timeout_s
        while time.time() < end:
            out = self._run("search", "--onlyvisible", "--name", name, check=False)
            if out:
                return out.splitlines()[0]
            time.sleep(0.5)
        raise GuiUnavailable(f"no window named {name!r} appeared within {timeout_s:g} s")

    def title(self, window: str) -> str:
        return self._run("getwindowname", window)

    def geometry(self, window: str) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) of a window on the screen."""
        out = dict(line.split("=", 1) for line in self._run("getwindowgeometry", "--shell", window).splitlines()
                   if "=" in line)
        x, y, w, h = (int(out[k]) for k in ("X", "Y", "WIDTH", "HEIGHT"))
        return x, y, x + w, y + h

    def raise_window(self, window: str, width: int | None = None, height: int | None = None) -> None:
        self._run("windowactivate", "--sync", window, check=False)        # no window manager: may not apply
        if width and height:
            self._run("windowmove", window, "0", "0")
            self._run("windowsize", window, str(width), str(height))

    def key(self, window: str, *keys: str) -> None:
        self._run("key", "--window", window, *keys)

    def type(self, window: str, text: str) -> None:
        self._run("type", "--window", window, "--delay", "20", text)

    def click(self, x: int, y: int, button: int = 1) -> None:
        self._run("mousemove", "--sync", str(x), str(y))
        self._run("click", str(button))


class PlaywrightDriver:
    """Drive a web UI: open a page, click, fill, wait for text, take screenshots. Every action is logged."""

    def __init__(self, headless: bool = True, executable_path: str | None = None):
        self.headless = headless
        self.executable_path = executable_path or (os.environ.get("PLAYWRIGHT_CHROMIUM")
                                                   or ("/opt/pw-browsers/chromium"
                                                       if Path("/opt/pw-browsers/chromium").exists() else None))
        self.log: list[str] = []
        self._pw = self._browser = self.page = None

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import playwright  # noqa: F401
        except ImportError:
            return False, "playwright is not installed (pip install playwright)"
        return True, "playwright"

    def __enter__(self) -> PlaywrightDriver:
        ok, why = self.available()
        if not ok:
            raise GuiUnavailable(why)
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        kw = {"headless": self.headless}
        if self.executable_path and Path(self.executable_path).is_file():
            kw["executable_path"] = self.executable_path
        self._browser = self._pw.chromium.launch(**kw)
        self.page = self._browser.new_page(viewport={"width": 1400, "height": 900})
        return self

    def __exit__(self, *exc) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def goto(self, url: str) -> None:
        self.log.append(f"goto {url}")
        self.page.goto(url)

    def click(self, selector: str) -> None:
        self.log.append(f"click {selector}")
        self.page.click(selector)

    def fill(self, selector: str, value: str) -> None:
        self.log.append(f"fill {selector}")
        self.page.fill(selector, value)

    def wait_text(self, text: str, timeout_s: float = 30.0) -> None:
        self.log.append(f"wait for text {text!r}")
        self.page.get_by_text(text).first.wait_for(timeout=timeout_s * 1000)

    def screenshot(self, path: str | Path, full_page: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), full_page=full_page)
        return path


class PywinautoDriver:
    """Windows desktop GUIs through ``pywinauto`` (UI Automation backend).

    The interface mirrors :class:`XdotoolDriver`: ``start(cmd)``, ``wait_window(title)``, ``click(control)``,
    ``type(control, text)``, ``screenshot(path)``. It is documented here so a workflow written for a Windows
    application has a place to plug in; it is **not runnable in the Linux container** and raises
    :class:`GuiUnavailable` there.
    """

    def __init__(self, backend: str = "uia"):
        self.backend = backend
        self.app = None
        self.log: list[str] = []

    @staticmethod
    def available() -> tuple[bool, str]:
        if sys.platform != "win32":
            return False, "pywinauto drives Windows applications; this is " + sys.platform
        try:
            import pywinauto  # noqa: F401
        except ImportError:
            return False, "pywinauto is not installed (pip install pywinauto)"
        return True, "pywinauto"

    def _need(self) -> None:
        ok, why = self.available()
        if not ok:
            raise GuiUnavailable(why)

    def start(self, cmd: str) -> None:
        self._need()
        from pywinauto import Application
        self.log.append(f"start {cmd}")
        self.app = Application(backend=self.backend).start(cmd)

    def wait_window(self, title: str, timeout_s: float = 60.0):
        self._need()
        self.log.append(f"wait for window {title!r}")
        w = self.app.window(title_re=title)
        w.wait("visible", timeout=timeout_s)
        return w

    def click(self, window, control: str) -> None:
        self._need()
        self.log.append(f"click {control}")
        window.child_window(title=control).click_input()

    def type(self, window, control: str, text: str) -> None:
        self._need()
        self.log.append(f"type into {control}")
        window.child_window(title=control).type_keys(text, with_spaces=True)

    def screenshot(self, window, path: str | Path) -> Path:
        self._need()
        window.capture_as_image().save(str(path))
        return Path(path)

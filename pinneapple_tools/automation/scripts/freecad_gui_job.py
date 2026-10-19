# Runs inside the FreeCAD GUI (``freecad freecad_gui_job.py``); launched by pinneapple_tools.automation.adapters.freecad.
# The job file (path in PINNEAPPLE_FC_JOB) names a script to execute with ARGS (dict in) and RESULT (dict out).
# The job starts from a timer, inside FreeCAD's running event loop, so the 3D view is live. Afterwards the objects in
# "show"/"hide" are shown/hidden, the view is fitted, the screen position of the 3D view is written with the result,
# a "ready" file tells the adapter to take its screenshots, and FreeCAD exits once the adapter writes "release".
import json
import os
import time
import traceback

import FreeCAD
import FreeCADGui
from PySide import QtCore

job = json.load(open(os.environ["PINNEAPPLE_FC_JOB"], encoding="utf-8"))


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        FreeCADGui.updateGui()
        time.sleep(0.05)


def run_job():
    result, error, view_box = {}, None, None
    try:
        mw = FreeCADGui.getMainWindow()
        w, h = job.get("window", [1600, 1000])
        mw.setGeometry(0, 0, w, h)
        pump(0.3)
        code = open(job["script"], encoding="utf-8").read()
        exec(compile(code, job["script"], "exec"),
             {"__name__": "__main__", "__file__": job["script"], "ARGS": job["args"], "RESULT": result,
              "FreeCAD": FreeCAD, "App": FreeCAD, "FreeCADGui": FreeCADGui, "Gui": FreeCADGui})
    except Exception:
        error = traceback.format_exc()
    try:
        gdoc = FreeCADGui.activeDocument()
        if gdoc is not None:
            for name, visible in [(n, False) for n in job.get("hide", [])] + [(n, True) for n in job.get("show", [])]:
                obj = gdoc.getObject(name)
                if obj is not None:
                    obj.Visibility = visible
            pump(0.3)
            view = gdoc.activeView()
            getattr(view, job.get("view", "viewIsometric"))()
            pump(0.3)
            view.fitAll()
            pump(0.8)
            from PySide import QtGui
            for wdg in mw.findChildren(QtGui.QWidget):
                if wdg.metaObject().className() == "Gui::View3DInventorViewer" and wdg.isVisible():
                    p = wdg.mapToGlobal(QtCore.QPoint(0, 0))
                    view_box = [p.x(), p.y(), p.x() + wdg.width(), p.y() + wdg.height()]
    except Exception:
        result.setdefault("warnings", []).append("view: " + traceback.format_exc().splitlines()[-1])
    json.dump({"result": result, "error": error, "view_box": view_box}, open(job["result"], "w", encoding="utf-8"),
              indent=1, default=str)
    open(job["ready"], "w").write("ready")
    end = time.time() + float(job.get("hold_s", 60))
    while time.time() < end and not os.path.exists(job["release"]):
        pump(0.2)
    os._exit(0)


QtCore.QTimer.singleShot(300, run_job)

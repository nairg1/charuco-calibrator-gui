import os

import pytest

tk = pytest.importorskip("tkinter")


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a display (e.g. xvfb-run)")
def test_gui_builds_and_reads_options():
    from charuco_calibrator.gui import CharucoCalibratorApp

    root = tk.Tk()
    try:
        app = CharucoCalibratorApp(root)
        options = app._intrinsics_options()
        assert options.fix_k3 and not options.fix_aspect_ratio and options.auto_constrain
        assert app._sampling(app.multi_every).every == 1
        assert len(app.multi_rows) == 6
        app._rebuild_multi_camera_rows(3)
        assert len(app.multi_rows) == 3
    finally:
        root.destroy()

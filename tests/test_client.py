"""FaceClient tests with a fake HTTP session — no network, no torch.

Focus: client-side downscale must rescale the server's *sent-frame* coords back
UP to original-frame coords, and report the ORIGINAL shape.
"""
from __future__ import annotations

import numpy as np

from online_face._wire import decode_image
from online_face.client import FaceClient


class _Resp:
    def __init__(self, payload, headers=None):
        self._p = payload
        self.headers = headers or {"content-type": "application/json"}

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _EchoBoxSession:
    """Fake server that decodes the posted frame and echoes a fixed box (in
    sent-frame coords) plus a landmark at the box corners."""

    def __init__(self, box):
        self.box = list(box)
        self.last_sent_shape = None

    def post(self, url, files=None, timeout=None, **kw):
        img = decode_image(files["frame"][1])
        h, w = img.shape[:2]
        self.last_sent_shape = (h, w)
        x1, y1, x2, y2 = self.box
        return _Resp({"outputs": {
            "boxes": [self.box],
            "scores": [0.99],
            "landmarks": [[[x1, y1], [x2, y1], [(x1 + x2) / 2, (y1 + y2) / 2],
                           [x1, y2], [x2, y2]]],
            "shape": [h, w],
        }, "stats": {}})


def test_no_downscale_passes_boxes_through_and_reports_original_shape():
    sess = _EchoBoxSession([10, 20, 100, 120])
    client = FaceClient(session=sess)  # max_side=None
    frame = np.zeros((480, 640, 3), "uint8")
    r = client.predict(frame)
    assert sess.last_sent_shape == (480, 640)        # untouched
    assert r.shape == (480, 640)
    assert np.allclose(r.boxes[0], [10, 20, 100, 120], atol=1e-4)


def test_downscale_rescales_boxes_by_inverse_scale():
    sess = _EchoBoxSession([10.0, 20.0, 100.0, 120.0])
    client = FaceClient(session=sess, max_side=600)
    frame = np.zeros((1200, 1800, 3), "uint8")        # max side 1800 -> scale = 1/3 -> sent 600x400
    r = client.predict(frame)
    assert max(sess.last_sent_shape) == 600
    assert r.shape == (1200, 1800)                    # ORIGINAL shape, not sent
    # box maps back up by 1/scale = 3
    assert np.allclose(r.boxes[0], [30, 60, 300, 360], atol=1.0)
    # landmark corners map back too
    assert np.allclose(r.landmarks[0, 0], [30, 60], atol=1.0)
    assert np.allclose(r.landmarks[0, 4], [300, 360], atol=1.0)


def test_per_call_max_side_overrides_ctor():
    sess = _EchoBoxSession([0, 0, 50, 50])
    client = FaceClient(session=sess, max_side=None)
    frame = np.zeros((1000, 1000, 3), "uint8")
    client.predict(frame, max_side=250)
    assert max(sess.last_sent_shape) == 250

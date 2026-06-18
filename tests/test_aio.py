"""FaceStream async session via a fake WebSocket transport (no server, no torch).

Uses ``asyncio.run`` so no pytest-asyncio plugin is needed."""
from __future__ import annotations

import asyncio
import json

import numpy as np

from online_face.aio import FaceStream


def _reply(fid, h=8, w=8):
    return json.dumps({
        "id": fid,
        "outputs": {"boxes": [[0, 0, 1, 1]], "scores": [0.9],
                    "landmarks": [[[0, 0]] * 5], "shape": [h, w]},
        "server": {"infer_ms": 1.0, "queue_depth": 0, "t_recv": 0.0, "t_send": 0.001}})


class _FakeConn:
    """Echoes a reply per binary message (4-byte id + payload). Optionally delays
    even ids so replies arrive out of order."""

    def __init__(self, reorder=False):
        self._replies = asyncio.Queue()
        self._reorder = reorder

    async def send(self, data):
        fid = int.from_bytes(bytes(data[:4]), "big")
        if self._reorder and fid % 2 == 0:
            asyncio.ensure_future(self._delayed(fid))
        else:
            await self._replies.put(_reply(fid))

    async def _delayed(self, fid):
        await asyncio.sleep(0.03)
        await self._replies.put(_reply(fid))

    async def recv(self):
        return await self._replies.get()

    async def close(self):
        pass


def test_results_roundtrip_preserves_metadata():
    async def main():
        async def connect(url):
            return _FakeConn(reorder=True)
        got = []
        async with FaceStream("http://x", connect=connect, tick_s=0.05) as s:
            for i in range(12):
                await s.push(np.zeros((8, 8, 3), "uint8"), meta=i)
            async for res, meta in s.results():
                got.append(meta)
                assert res.boxes.shape[1] == 4
                if len(got) == 12:
                    break
        return sorted(got)

    assert asyncio.run(main()) == list(range(12))


def test_push_backpressure_blocks_when_full():
    async def main():
        class _DeadConn:
            async def send(self, d):
                await asyncio.sleep(3600)

            async def recv(self):
                await asyncio.sleep(3600)

            async def close(self):
                pass

        async def connect(url):
            return _DeadConn()

        s = FaceStream("http://x", connect=connect, max_queue=2, tick_s=10)
        # 1 pulled by the (stuck) sender + 2 fill the queue = 3 accepted; the 4th blocks.
        for _ in range(3):
            await s.push(np.zeros((4, 4, 3), "uint8"))
        blocked = False
        try:
            await asyncio.wait_for(s.push(np.zeros((4, 4, 3), "uint8")), timeout=0.3)
        except asyncio.TimeoutError:
            blocked = True
        s._closing = True
        return blocked

    assert asyncio.run(main()) is True

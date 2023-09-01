from __future__ import annotations

import multiprocessing as mp
import queue
import time

INIT_COMMAND = "<<I_SP>>"
INIT_DONE_MSG = ">>I_D_MSG<<"
WORK_COMMAND = "<<W>>"
RETURN_MODELS = "<<RM>>"
NEW_MODEL = "<<NM>>"


class MySimpleQueue:
    def __init__(self):
        self.q = mp.SimpleQueue()

    def get(self, block: bool = True, timeout: float | None = None):
        msg = None
        if block:
            timeout = 9999 if timeout is None else timeout
            t = time.time()
            while True:
                if time.time() - t > timeout:
                    raise queue.Empty
                if self.q.empty():
                    time.sleep(0.000001)
                else:
                    msg = self.q.get()
                    break
        else:
            if not self.q.empty():
                msg = self.q.get()

        return msg

    def get_nowait(self):
        return self.get(block=False)

    def put(self, obj_):
        self.q.put(obj_)
        time.sleep(0.000001)


class MyManagedQueue:
    def __init__(self):
        self.q = mp.Manager().Queue()

    def get(self, block: bool = True, timeout: float | None = None):
        return self.q.get(block, timeout)

    def get_nowait(self):
        return self.q.get_nowait()

    def put(self, obj_):
        self.q.put(obj_)
        time.sleep(0.000001)


class MyBasicPipe:
    def __init__(self):
        self.dst, self.src = mp.Pipe(duplex=False)

    def put(self, obj_):
        self.src.send(obj_)
        time.sleep(0.00001)

    def get(self, block: bool = True, timeout: float | None = None):
        if timeout is None:
            timeout = 0.00001

        if self.dst.poll(timeout=timeout if not block else 600):
            msg = self.dst.recv()
        else:
            msg = None

        return msg

    def get_nowait(self):
        return self.get(block=False)

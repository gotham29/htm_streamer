from __future__ import annotations

import logging
import os
import itertools
import multiprocessing as mp


import queue
import time
from typing import Dict, Callable, Tuple, Any, Optional, List

import tqdm
from htm.bindings.sdr import SDR

from htm_source.model.htm_model import HTMModule
from htm_source.utils import dict_zip


INIT_COMMAND = "<<I_SP>>"
INIT_DONE_MSG = ">>I_D_MSG<<"
WORK_COMMAND = "<<W>>"
RETURN_MODELS = "<<RM>>"
NEW_MODEL = "<<NM>>"


class HTMProcessHive:
    def __init__(self, num_workers: int = mp.cpu_count() - 1):
        self.workers: Dict[str, HTMProcess] = {}
        self._model2worker: Dict[str, str] = {}
        self._n_workers = num_workers
        self._works_pending = 0
        self._started = False
        self._create_workers()
        self.start()
        self.worker_belt = self._cycle_workers()

    def send_sdr(self, model_key: str, sdr: SDR):
        payload = (WORK_COMMAND, {'key': model_key, 'sdr': sdr})
        w_name = self._model2worker[model_key]
        worker = self.workers[w_name]
        worker.in_q.put(payload)
        self._works_pending += 1

    def collect_results(self) -> Dict[str: SDR]:
        results = {}
        while self._works_pending:
            _, (key, sdr) = self._recv_from_belt_v2()  # blocks until at least 1 worker is done
            results[key] = sdr
            self._works_pending -= 1

        return results

    def add_model(self, key: str, model: HTMModule):
        worker = next(self.worker_belt)
        worker.in_q.put((NEW_MODEL, {'key': key, 'model': model}))
        self._model2worker[key] = worker.name

    def init_all_sp(self):
        if not self._started:
            self.start()

        for worker in self.workers.values():
            worker.in_q.put((INIT_COMMAND, None))

        finished = 0
        while finished != self._n_workers:
            pid, msg = self._recv_from_belt_v2()
            if msg == INIT_DONE_MSG:
                finished += 1
            else:
                raise RuntimeError(f"Expected `INIT_DONE_MSG` from worker, got: {msg} from worker `{pid}`")

    def return_all_models(self) -> Dict[str, HTMModule]:
        if self._works_pending:
            raise RuntimeError(f"Cannot return models, {self._works_pending} works pending")

        for worker in self.workers.values():
            worker.in_q.put((RETURN_MODELS, None))

        results = {}
        for model_key, w_name in self._model2worker.items():
            if model_key in results:
                continue

            worker = self.workers[w_name]
            try:
                models = worker.out_q.get(block=True, timeout=2)
            except queue.Empty:
                raise RuntimeError(f"Worker {w_name} model-return timed out.")
            except Exception:
                raise
            else:
                results.update(models)

        return results

    def start(self):
        for worker in self.workers.values():
            worker.start()

        self._started = True

    def join(self):
        for worker in self.workers.values():
            worker.join()

    def terminate(self):
        for worker in self.workers.values():
            worker.terminate()

    def _recv_from_belt(self) -> Tuple[str, Any]:
        while True:
            try:
                worker = next(self.worker_belt)
                recv = worker.out_q.get_nowait()
            except queue.Empty:
                time.sleep(0.000001)
            except Exception:
                raise
            else:
                w_name = worker.name
                break

        return w_name, recv

    def _recv_from_belt_v2(self) -> Tuple[str, Any]:
        while True:
            worker = next(self.worker_belt)
            recv = worker.out_q.get_nowait()
            if recv is None:
                pass
            else:
                w_name = worker.name
                break

        return w_name, recv

    def _cycle_workers(self):
        for worker in itertools.cycle(self.workers.values()):
            yield worker

    def _create_workers(self):
        for _ in tqdm.tqdm(range(self._n_workers), desc="Initiating hive.."):
            proc = HTMProcess()
            self.workers[proc.name] = proc


class HTMProcess(mp.Process):
    def __init__(self):
        super().__init__()
        # self.in_q, self.out_q = mp.Manager().Queue(), mp.Manager().Queue()
        self.in_q, self.out_q = MySimpleQueue(), MySimpleQueue()
        # self.in_q, self.out_q = MySimpleQueue(), MySimpleQueue()
        # self.in_q, self.out_q = MyBasicPipe(), MyBasicPipe()
        self.models: Dict[str, HTMModule] = {}
        # self.logger = logging.Logger(self.name)

    def run(self) -> None:
        # os.nice(-10)
        # self.logger.debug("START")
        print(f"{self.name} START ")
        while True:
            command, kwargs = self.in_q.get(block=True)
            if command == NEW_MODEL:
                # self.logger.debug(f"Got new model: {kwargs['key']}")
                self.models[kwargs['key']] = kwargs['model']

            elif command == WORK_COMMAND:
                result = self.models[kwargs['key']](kwargs['sdr'])
                self.out_q.put((kwargs['key'], result))

            elif command == RETURN_MODELS:
                self.out_q.put(self.models)

            elif command == INIT_COMMAND:
                print(f"{self.name} START INIT FOR {len(self.models)} models")
                t = time.perf_counter()
                for model in self.models.values():
                    model._init_sp()
                self.out_q.put(INIT_DONE_MSG)
                print(f"{self.name} DONE INIT FOR {len(self.models)} models in {time.perf_counter()-t:.2f} Sec")

            else:
                raise ValueError(f"UNKNOWN COMMAND `{command}`")


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
                    break
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


def mp_pool_model_dict_apply(model_dict: Dict,
                             func: Callable,
                             extra_args: Dict = None,
                             cpu_count: int = mp.cpu_count()) -> Optional[Dict]:
    ret_dict = {}
    with mp.Pool(cpu_count) as pool:
        returned_items = pool.starmap(func, dict_zip(model_dict, extra_args))
        for k, v, *r in returned_items:
            model_dict[k] = v
            if r:
                ret_dict[k] = r

    return ret_dict if ret_dict else None


def mp_proc_model_dict_apply(model_dict: Dict,
                             func: Callable,
                             src: mp.Queue,
                             dst: mp.Queue,
                             extra_args: Dict = None) -> Optional[Dict]:

    ret_dict = {}
    n_jobs = len(model_dict)
    done_jobs = 0

    for args in dict_zip(model_dict, extra_args):
        src.put((func, args))

    while done_jobs != n_jobs:
        returned_items = dst.get(block=True)
        done_jobs += 1
        for k, v, *r in returned_items:
            model_dict[k] = v
            if r:
                ret_dict[k] = r

    return ret_dict if ret_dict else None


def model_dict_apply(model_dict: Dict,
                     func: Callable,
                     extra_args: Dict = None) -> Optional[Dict]:
    ret_dict = {}
    returned_items = itertools.starmap(func, dict_zip(model_dict, extra_args))
    for k, v, *r in returned_items:
        model_dict[k] = v
        if r:
            ret_dict[k] = r

    return ret_dict if ret_dict else None


def multiprocess_model_list_apply(model_list: List,
                                  func: Callable,
                                  extra_args: List = None,
                                  cpu_count: int = mp.cpu_count()) -> Optional[List]:
    ret_list = []
    with mp.Pool(cpu_count) as pool:
        returned_items = pool.starmap(func, zip(model_list, extra_args)) if extra_args else pool.map(func, model_list)
        for idx, item in enumerate(returned_items):
            if not isinstance(item, HTMModule):
                model, ret = item[0], item[1:]
                ret_list.append(ret)
            else:
                model = item

            model_list[idx] = model

    return ret_list if ret_list else None


def mp_init_sp(key: str, model: HTMModule) -> Tuple[str, HTMModule]:
    model._init_sp()
    return key, model


def mp_forward(key: str, model: HTMModule, data: SDR) -> Tuple[str, HTMModule, SDR]:
    pred = model(data)
    return key, model, pred


def process_job(src: mp.Queue, dst: mp.Queue):
    while True:
        func, args = src.get(block=True)
        result = func(*args)
        dst.put([result], block=False)

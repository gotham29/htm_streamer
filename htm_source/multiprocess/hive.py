from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import time
from typing import Dict, Tuple, Any

import tqdm
from htm.bindings.sdr import SDR

from htm_source.model.htm_model import HTMModule
from htm_source.multiprocess.communication import INIT_COMMAND, INIT_DONE_MSG, WORK_COMMAND, RETURN_MODELS, NEW_MODEL
from htm_source.multiprocess.process import HTMProcess


class HTMProcessHive:
    def __init__(self, num_workers: int = mp.cpu_count() - 1):
        self._workers: Dict[str, HTMProcess] = {}
        self._model2worker: Dict[str, str] = {}
        self._n_workers = num_workers
        self._works_pending = 0
        self._started = False

        self._create_workers()
        self._start()
        self._worker_belt = self._cycle_workers()

    def send_sdr(self, model_key: str, sdr: SDR):
        """ Send an input sdr to the model with the given `model_key` """
        payload = (WORK_COMMAND, {'key': model_key, 'sdr': sdr})
        w_name = self._model2worker[model_key]
        worker = self._workers[w_name]
        worker.in_q.put(payload)
        self._works_pending += 1

    def collect_results(self) -> Dict[str: SDR]:
        """ Collect the results (output) of all models """
        results = {}
        while self._works_pending:
            _, (key, sdr) = self._recv_from_belt_v2()  # blocks until at least 1 worker is done
            results[key] = sdr
            self._works_pending -= 1

        return results

    def add_model(self, key: str, model: HTMModule):
        """ Register a model in the hive with a given key """
        worker = next(self._worker_belt)
        worker.in_q.put((NEW_MODEL, {'key': key, 'model': model}))
        self._model2worker[key] = worker.name

    def init_all_sp(self):
        """ Initialize spatial-pooling layers in all models """
        if not self._started:
            self._start()

        for worker in self._workers.values():
            worker.in_q.put((INIT_COMMAND, None))

        finished = 0
        while finished != self._n_workers:
            w_name, msg = self._recv_from_belt_v2()
            if msg == INIT_DONE_MSG:
                finished += 1
            else:
                raise RuntimeError(f"Expected `INIT_DONE_MSG` from worker, got: {msg} from worker `{w_name}`")

    def return_all_models(self) -> Dict[str, HTMModule]:
        """ Return all the models from the hive's other processes """
        if self._works_pending:
            raise RuntimeError(f"Cannot return models, {self._works_pending} works pending")

        for worker in self._workers.values():
            worker.in_q.put((RETURN_MODELS, None))

        results = {}
        for model_key, w_name in self._model2worker.items():
            if model_key in results:
                continue

            worker = self._workers[w_name]
            try:
                models = worker.out_q.get(block=True, timeout=3)
            except queue.Empty:
                raise RuntimeError(f"Worker {w_name} model-return timed out.")
            except Exception:
                raise
            else:
                results.update(models)

        return results

    def _start(self):
        for worker in self._workers.values():
            worker.start()

        self._started = True

    def join(self):
        for worker in self._workers.values():
            worker.join()

    def terminate(self):
        for worker in self._workers.values():
            worker.terminate()

    def _recv_from_belt(self) -> Tuple[str, Any]:
        while True:
            try:
                worker = next(self._worker_belt)
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
            worker = next(self._worker_belt)
            recv = worker.out_q.get_nowait()
            if recv is None:
                pass
            else:
                w_name = worker.name
                break

        return w_name, recv

    def _cycle_workers(self):
        """ Creates an infinite worker belt-style generator """
        for worker in itertools.cycle(self._workers.values()):
            yield worker

    def _create_workers(self):
        for _ in tqdm.tqdm(range(self._n_workers), desc="Initiating hive.."):
            proc = HTMProcess()
            self._workers[proc.name] = proc

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Dict

from htm.bindings.sdr import SDR

from htm_source.model.htm_model import HTMModule
from htm_source.multiprocess.communication import INIT_COMMAND, INIT_DONE_MSG, WORK_COMMAND, RETURN_MODELS, NEW_MODEL, \
    MySimpleQueue


class HTMProcess(mp.Process):
    def __init__(self):
        super().__init__()
        # self.in_q, self.out_q = mp.Manager().Queue(), mp.Manager().Queue()
        self.in_q, self.out_q = MySimpleQueue(), MySimpleQueue()
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
                x = kwargs['sdr']
                model = self.models[kwargs['key']]
                result = model(x) if isinstance(x, SDR) else model(*x)
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

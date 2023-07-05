from __future__ import annotations

import math
import time
from itertools import chain
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp

from htm.bindings.sdr import SDR
from tqdm import tqdm

from htm_source.data.data_streamer import DataStreamer
from htm_source.model.htm_model import HTMModule
from htm_source.pipeline.multiprocess import HTMProcessHive, mp_pool_model_dict_apply, mp_init_sp
from htm_source.utils.general import split_features, choose_features, get_joined_name
from htm_source.utils.sdr import sdr_merge


class ModelPyramid:
    def __init__(self,
                 data: pd.DataFrame,
                 feat_cfg: dict,
                 enc_cfg: dict,
                 sp_cfg: dict | None,
                 tm_cfg: dict,
                 inputs_per_layer: int = 3,  # | tuple | list
                 train_percent: float = 1.0,
                 prepare_data: bool = True,
                 feature_join_str: str = '+',
                 dummy_feat: str = 'None',
                 mode: str = 'split',
                 merge_mode: str = 'u',
                 num_choices: int = None,
                 seed: int = 0,
                 max_pool: int = 1,
                 flatten: bool = False,
                 learn_schedule: slice = None,
                 anomaly_score: bool = False,
                 lazy_init: bool = True):

        if dummy_feat in data.columns.tolist():
            raise AssertionError(f"Data cannot contain a feature named `{dummy_feat}`. If this is an "
                                 f"issue, change the `dummy_feat` parameter")
        if mode == 'split':
            bottom_layer = split_features(data.columns.tolist(), size=inputs_per_layer)
        elif mode == 'choose':
            if not isinstance(num_choices, int):
                raise TypeError("int")

            bottom_layer = choose_features(data.columns.tolist(), n_choices=num_choices, choice_size=inputs_per_layer)
        else:
            raise ValueError("mode")

        print(f"Features chosen: {bottom_layer}")

        self._ds = DataStreamer(data,
                                features_cfg=feat_cfg,
                                encoders_cfg=enc_cfg,
                                train_percent=train_percent,
                                concat=bottom_layer,
                                prepare=prepare_data,
                                feature_join_str=feature_join_str)

        self.layer_size = inputs_per_layer
        self.max_pool = max_pool
        self.flat = flatten
        self.calc_anomaly = anomaly_score
        self.lr_sch = learn_schedule
        self.lazy = lazy_init
        self.merge_mode = merge_mode
        self._fjs = feature_join_str
        self._configuration = {'sp': sp_cfg, 'tm': tm_cfg}
        self._seed = seed
        self.n_jobs = mp.cpu_count()
        self._dummy = dummy_feat
        self._n_models = 0
        self._n_layers = 0
        self._par2ch = {}
        self._ch2par = {}
        self._layer_dict: Dict[int, Dict[str, HTMModule]] | Dict[int, List[str]] = {}

        self.hive = HTMProcessHive(5)
        self.build(bottom_layer)

    def build(self, bottom_layer: List[Tuple[str]]):
        t0 = time.perf_counter()
        print("Building model... ", end='')

        model_dict = {}
        current_layer = bottom_layer
        num_layers = math.ceil(math.log(len(bottom_layer), self.layer_size)) + 1

        for layer_idx in range(num_layers):
            new_layer = []

            for inputs in current_layer:
                # make connection
                if len(inputs) == 1:
                    inputs = tuple([inputs[0], self._dummy])

                target = get_joined_name(inputs, self.fjs)
                self._add_connection(inputs, target)
                new_layer.append(target)

                # get model input dim
                if layer_idx == 0:
                    input_dims = self.ds.shape[target]
                else:
                    input_dims = model_dict[inputs[0]].output_dim

                # init model
                self._n_models += 1
                model = HTMModule(input_dims=input_dims,
                                  sp_cfg=self.config['sp'],
                                  tm_cfg=self.config['tm'],
                                  seed=self.seed * self._n_models,
                                  anomaly_score=self.calc_anomaly,
                                  learn_schedule=self.lr_sch,
                                  max_pool=self.max_pool,
                                  flatten=self.flat,
                                  lazy_init=self.lazy)

                model_dict[target] = model
                self.hive.add_model(key=target, model=model)

            current_layer = split_features(new_layer, size=self.layer_size)
            # TODO ????
            # if len(current_layer) > 1 and any(filter(lambda x: len(x) == 1, current_layer)):
            #     current_layer = self._redistribute_layer(current_layer)

            self._layer_dict[layer_idx] = list(chain(*current_layer))

        if self.lazy:
            self.hive.init_all_sp()
            # mp_pool_model_dict_apply(model_dict, mp_init_sp)

        self._n_layers = len(self._layer_dict)
        # self._fuse_layers_models(model_dict)

        print(f"Done in {time.perf_counter() - t0:.1f} sec")
        print(f"The model is {self._n_layers} layers deep with a total of {self._n_models} HTMs")
        time.sleep(0.5)

    def _add_connection(self, inputs: Tuple[str], target: str):
        self._par2ch[target] = inputs
        for inp in inputs:
            if inp != self._dummy:
                self._ch2par[inp] = target

    def _fuse_layers_models(self, model_dict: Dict[str, HTMModule]):
        fused_dict = {}
        for idx, layer in self._layer_dict.items():
            fused_dict[idx] = {}
            for model_name in layer:
                fused_dict[idx][model_name] = model_dict[model_name]

        self._layer_dict = fused_dict

    def _run_layer_with_hive(self, prev_layer_results: Dict[str, SDR]) -> Dict[str, SDR]:
        for model_key, sdr in prev_layer_results.items():
            self.hive.send_sdr(model_key, sdr)

        results = self.hive.collect_results()
        return results

    def run(self, iterations: int = None):
        if iterations is None:
            iterations = len(self._ds)

        iterations = min(len(self._ds), iterations)
        for idx, item in tqdm(enumerate(self._ds), total=iterations, desc="Running model"):
            if idx == iterations:
                break

            prev_layer_results = item
            for layer_idx in self._layer_dict:

                # unmerged_results = model_dict_apply(layer_dict, mp_forward, extra_args=prev_layer_results)
                unmerged_results = self._run_layer_with_hive(prev_layer_results)

                # for each model in next layer, find the results to merge and do it, store in `prev_layer_results`
                if layer_idx + 1 < self._n_layers:
                    prev_layer_results = {}
                    next_layer = self._layer_dict[layer_idx + 1]
                    for parent_model in next_layer:
                        to_merge = [unmerged_results[child] for child in self.children(parent_model) if
                                    child != self._dummy]

                        merged = sdr_merge(*to_merge, mode=self.merge_mode)
                        prev_layer_results[parent_model] = merged

        trained_models = self.hive.return_all_models()
        self._fuse_layers_models(trained_models)
        self.hive.terminate()

    def children(self, parent: str) -> Tuple[str]:
        return self._par2ch[parent]

    def parent(self, child: str) -> str:
        return self._ch2par[child]

    def plot_results(self, gt=None, thresh=0.8):
        for layer_idx, layer_dict in self._layer_dict.items():
            fig, axs = plt.subplots(nrows=len(layer_dict), figsize=(15, 5 * len(layer_dict)))
            if len(layer_dict) == 1:
                axs = [axs]

            fig.suptitle(f"Layer {layer_idx}")
            for i, (name, model) in enumerate(layer_dict.items()):
                axs[i].set_title(name)
                scores = np.array(model.anomaly['score'])
                scores[scores <= thresh] = 0
                axs[i].plot(scores, label=f'Anomaly > {thresh}')
                if gt is not None:
                    axs[i].plot(gt, label='GT')

                axs[i].legend()

            fig.show()

    @property
    def ds(self) -> DataStreamer:
        return self._ds

    @property
    def fjs(self) -> str:
        return self._fjs

    @property
    def config(self) -> dict:
        return self._configuration.copy()

    @property
    def seed(self) -> int:
        return self._seed

    @staticmethod
    def _redistribute_layer(current_layer: List[Tuple[str]]):
        single = current_layer.pop(-1)
        first = current_layer.pop(0)
        new_first = tuple(list(first) + list(single))
        return [new_first] + current_layer

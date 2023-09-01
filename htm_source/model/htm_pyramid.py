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
from htm_source.multiprocess.hive import HTMProcessHive
from htm_source.utils.general import split_features, choose_features, get_joined_name, manual_seed
from htm_source.utils.sdr import sdr_merge, concat_shapes


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
                 feature_join_str: str = '_',
                 dummy_feat: str = 'None',
                 feature_mode: str = 'split',
                 merge_mode: str = 'u',
                 concat_axis: int = 0,
                 num_choices: int = None,
                 seed: int = 0,
                 max_pool: int = 1,
                 flatten: bool = False,
                 learn_schedule: slice = None,
                 al_learn_period: int = 1000,
                 anomaly_score: bool = False,
                 lazy_init: bool = True):
        """
        TODO
        """

        manual_seed(seed)

        if dummy_feat in data.columns.tolist():
            raise AssertionError(f"Data cannot contain a feature named `{dummy_feat}`. If this is an "
                                 f"issue, change the `dummy_feat` parameter")
        if feature_mode == 'split':
            bottom_layer = split_features(data.columns.tolist(), size=inputs_per_layer)
        elif feature_mode == 'choose':
            if not isinstance(num_choices, int):
                raise TypeError(f"`num_choices` must be an integer, got: {num_choices}")

            bottom_layer = choose_features(data.columns.tolist(), n_choices=num_choices, choice_size=inputs_per_layer)
        else:
            raise ValueError(f"`feature_mode` could be either `choose` or `split`, got: {feature_mode}")

        print(f"Features chosen: {bottom_layer}")

        self._ds = DataStreamer(data,
                                features_cfg=feat_cfg,
                                encoders_cfg=enc_cfg,
                                train_percent=train_percent,
                                concat=bottom_layer,
                                prepare=prepare_data,
                                feature_join_str=feature_join_str)

        self.n_jobs = min(mp.cpu_count() - 1, len(bottom_layer))
        self.layer_size = inputs_per_layer
        self.max_pool = max_pool
        self.flat = flatten
        self.calc_anomaly = anomaly_score
        self.lr_sch = learn_schedule
        self.lazy = lazy_init
        self.merge_mode = merge_mode
        self.al_learn_period = al_learn_period
        self.concat_axis = concat_axis

        self._fjs = feature_join_str
        self._configuration = {'sp': sp_cfg, 'tm': tm_cfg}
        self._seed = seed
        self._head = None
        self._dummy = dummy_feat
        self._n_models = 0
        self._n_layers = 0
        self._current_iter = 0
        self._par2ch = {}
        self._ch2par = {}
        self._models_stats: Dict[str, str] = {}
        self._layer_dict: Dict[int, Dict[str, HTMModule]] | Dict[int, List[str]] = {}

        self.hive = HTMProcessHive(self.n_jobs)
        self.build(bottom_layer)

    def build(self, bottom_layer: List[Tuple[str]]):
        """ Build the entire HTMPyramid bottom-up, creating the individual modules and connecting them appropriately """

        t0 = time.perf_counter()
        print("Building model... ")

        model_dict = {}
        current_layer = bottom_layer
        num_layers = math.ceil(math.log(len(bottom_layer), self.layer_size)) + 1

        for layer_idx in range(num_layers):
            is_head = layer_idx + 1 == num_layers
            new_layer = []

            for inputs in current_layer:

                if len(inputs) == 1:
                    inputs = tuple([inputs[0], self._dummy])

                # make connection
                target = get_joined_name(inputs, self.fjs)
                self._add_connection(inputs, target)
                new_layer.append(target)

                # get model input dim
                input_dims = self._get_input_dim(model_dict, layer_idx, target, inputs)

                # init model
                self._n_models += 1
                model = HTMModule(input_dims=input_dims,
                                  sp_cfg=self.config['sp'],
                                  tm_cfg=self.config['tm'],
                                  seed=self.seed * self._n_models,
                                  anomaly_score=self.calc_anomaly,
                                  anomaly_likelihood=self.calc_anomaly and is_head,
                                  al_learn_period=self.al_learn_period,
                                  learn_schedule=self.lr_sch,
                                  max_pool=self.max_pool,
                                  flatten=self.flat,
                                  lazy_init=self.lazy)

                model_dict[target] = model
                self._models_stats[target] = model.summary()
                self.hive.add_model(key=target, model=model)

            current_layer = split_features(new_layer, size=self.layer_size)
            self._layer_dict[layer_idx] = list(chain(*current_layer))

        if self.lazy:
            self.hive.init_all_sp()

        self._n_layers = len(self._layer_dict)

        print(f"Done in {time.perf_counter() - t0:.1f} sec")
        print(f"The model is {self._n_layers} layers deep with a total of {self._n_models} HTMs")
        time.sleep(0.5)

    def _get_input_dim(self, model_dict, layer_idx: int, target, inputs):
        if layer_idx == 0:
            input_dims = self.ds.shape[target]
        else:
            if self.merge_mode == 'c':
                shapes = [model_dict[x].output_dim for x in inputs if x != self._dummy]
                input_dims = concat_shapes(*shapes, axis=self.concat_axis)
            else:
                input_dims = model_dict[inputs[0]].output_dim

        return input_dims

    def _add_connection(self, inputs: Tuple[str], target: str):
        """ Register all models in `inputs` as inputs to `target` model """
        self._par2ch[target] = inputs
        for inp in inputs:
            if inp != self._dummy:
                self._ch2par[inp] = target

    def _fuse_layers_models(self, model_dict: Dict[str, HTMModule]):
        """ Fuse the model dict into the layered model-name dict """
        fused_dict = {}
        for idx, layer in self._layer_dict.items():
            fused_dict[idx] = {}
            for model_name in layer:
                fused_dict[idx][model_name] = model_dict[model_name]

        self._layer_dict = fused_dict

    def _run_layer_with_hive(self, prev_layer_results: Dict[str, SDR]) -> Dict[str, SDR]:
        """ Run the current layer (given previous layer results) with the hive and return the outputs """
        for model_key, sdr in prev_layer_results.items():
            self.hive.send_sdr(model_key, sdr)

        results = self.hive.collect_results()
        return results

    def _merge_layer_results(self, unmerged_results: Dict[str, SDR], next_layer: Dict[str, HTMModule]) -> Dict[str, SDR]:
        """ For each model in the next layer, get its inputs from the current layer's outputs and merge them """
        prev_layer_results = {}

        for parent_model in next_layer:
            to_merge = [unmerged_results[child] for child in self.children(parent_model) if
                        child != self._dummy]

            merged = sdr_merge(*to_merge, mode=self.merge_mode, axis=self.concat_axis)
            if len(next_layer) == 1:
                # if next layer is head,
                # get the actual data point and feed the model (for anomaly likelihood)
                data_point = self.ds.raw_data(self._current_iter)
                prev_layer_results[parent_model] = (merged, data_point)
            else:
                prev_layer_results[parent_model] = merged

        return prev_layer_results

    def run(self, iterations: int = None):
        """
        Run the HTMPyramid for `iterations` steps. (entire dataset by default)

        Note that the Pyramid can only run once, at which point the trained models are retrieved and the _workers killed.
        """

        _iterations = len(self._ds) if iterations is None else min(len(self._ds), iterations)

        for ds_idx, item in tqdm(enumerate(self._ds), total=_iterations, desc="Running model"):
            if self._current_iter >= _iterations:
                break

            prev_layer_results = item
            for layer_idx in self._layer_dict:
                # run layer and get results per model
                unmerged_results = self._run_layer_with_hive(prev_layer_results)

                # if not last layer (head)
                if layer_idx + 1 < self._n_layers:
                    next_layer = self._layer_dict[layer_idx + 1]
                    # find the results to merge and do it, store in `prev_layer_results`
                    prev_layer_results = self._merge_layer_results(unmerged_results, next_layer)

            self._current_iter += 1

        # get trained models back
        trained_models = self.hive.return_all_models()
        # fuse back into self dict, set head
        self._fuse_layers_models(trained_models)
        self._head = list(self._layer_dict[self._n_layers - 1].values()).pop()
        # terminate processes
        self.hive.terminate()

    def children(self, parent: str) -> Tuple[str]:
        """ Returns the names of the children (input) models of `parent` """
        return self._par2ch[parent]

    def parent(self, child: str) -> str:
        """ Returns the name of the parent (next layer in hierarchy) model of `child` """
        return self._ch2par[child]

    def model_summary(self):
        print()
        for idx, layer in self._layer_dict.items():
            print((idx == 0) * "INPUT " + "LAYER " + f"{idx}" * (idx > 0))
            longest = max(map(len, layer))
            max_pad = longest + 15
            for name in layer:
                dots = "." * (max_pad - len(name))
                summary = self._models_stats[name]
                print(f"{name}{dots}{summary}")

            print()

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

                if model == self.head:
                    likl = np.array(model.anomaly['likelihood'])
                    likl[likl <= thresh] = 0
                    axs[i].plot(likl, label='A LIKL')

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

    @property
    def head(self) -> HTMModule:
        return self._head

    @staticmethod
    def _redistribute_layer(current_layer: List[Tuple[str]]):
        single = current_layer.pop(-1)
        first = current_layer.pop(0)
        new_first = tuple(list(first) + list(single))
        return [new_first] + current_layer

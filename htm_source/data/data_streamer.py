from __future__ import annotations

from typing import Mapping
import time
from warnings import warn

import numpy as np
import pandas as pd
import multiprocessing as mp
from htm.bindings.sdr import SDR
from htm_source.config import build_enc_params
from htm_source.data import Feature
from htm_source.utils import dict_zip
from htm_source.utils.sdr import concat_shapes
from htm_source.utils.general import get_joined_name, get_base_names


class DataStreamer:

    def __init__(self,
                 df: pd.DataFrame,
                 *,
                 features_cfg: dict,
                 encoders_cfg: dict,
                 train_percent: float = 1.0,
                 concat: bool | tuple | list = False,
                 prepare: bool = False,
                 feature_join_str: str = '+'):
        """
        TODO
        """
        self._dataframe = df.reset_index(drop=True)
        self._prepared = False
        self._encoded_data = None
        self._fjs = feature_join_str

        self._params = build_enc_params(features_cfg=features_cfg,
                                        encoders_cfg=encoders_cfg,
                                        data_samples=df.iloc[:int(len(df) * train_percent)])
        self.features = {name: Feature(name, params) for name, params in self._params.items()}

        self._concat = concat
        self._check_concat()

        self._encoding_width = self._get_encoding_width()

        if prepare:
            _s = time.perf_counter()
            print("Encoding data.. ", end='')
            with mp.Pool(mp.cpu_count()) as pool:
                self._encoded_data = pool.map(self.__getitem__, range(len(self)))
            print(f"Done in {time.perf_counter() - _s:.3f} sec")
            self._prepared = True

    def __len__(self) -> int:
        return len(self._dataframe)

    def __getitem__(self, idx: int) -> SDR | dict[str, SDR]:
        if self._prepared:
            item = self._encoded_data[idx]
        else:
            row = self._dataframe.iloc[idx]
            item = self.get_encoding(row)

        return item

    def get_encoding(self, data_row: Mapping) -> SDR | dict[str, SDR]:
        """ Encode the feature or features in the given dataframe row into SDR(s)
            If the dataset is a single feature, or `concat` is True, returns a single SDR
            Otherwise, returns a dict mapping feature name (or names if concatenated as a group) to SDR """

        if self._concat is True:
            # Get encodings for all features
            all_encodings = [SDR(0)] + [feature.encode(data) for _, data, feature in dict_zip(data_row, self.features)]

            # Combine all features encodings into one for Spatial Pooling
            encoding = SDR(self._encoding_width).concatenate(all_encodings)

        elif isinstance(self._concat, (list, tuple)):
            encoding = dict()

            # get encoding for each feature separately
            temp_encoding = {f_name: feature.encode(data) for f_name, data, feature in dict_zip(data_row,
                                                                                                self.features)}
            for j_f_name, shape in self._encoding_width.items():
                base_names = get_base_names(j_f_name, self.fjs)

                # concatenate this combination of features into a new encoding
                temp_sdrs = [temp_encoding[b_n] for b_n in base_names]
                if len(temp_sdrs) == 1:
                    new_encoding = temp_sdrs[0]
                else:
                    new_encoding = SDR(shape).concatenate(temp_sdrs)

                encoding[j_f_name] = new_encoding

        else:
            encoding = {f_name: feature.encode(data) for f_name, data, feature in dict_zip(data_row, self.features)}

        return encoding

    def _get_encoding_width(self) -> np.ndarray | dict[str, np.ndarray]:
        """ Return the encoding width(s).
            If the dataset is a single feature, or `concat` is True, returns a 1 element np.array
            Otherwise, returns a dict mapping feature name (or names if concatenated as a group) to encoding width """

        if self._concat is True:
            return concat_shapes(*(f_enc.encoding_dim for f_enc in self.features.values()))

        elif isinstance(self._concat, (list, tuple)):
            shapes = dict()
            for features in self._concat:
                joined_name = get_joined_name(features, self.fjs)
                length = concat_shapes(*(self.features[f_name].encoding_dim for f_name in features))
                shapes[joined_name] = length

            return shapes

        else:
            return {f_name: np.array(f_enc.encoding_dim) for f_name, f_enc in self.features.items()}

    def _check_concat(self):
        """ Checks that the `concat` argument has a valid value """
        if self._concat not in (True, False, None):
            features = set()
            for conc_feats in self._concat:
                for cf in conc_feats:
                    if self._fjs in cf:
                        raise AssertionError(f"Feature names cannot contain `{self._fjs}`, got: `{cf}`. If this is an "
                                             f"issue, change the `feature_join_str` parameter")
                features.update(conc_feats)

            if features != self.features.keys():
                missing = set(self.features.keys()).difference(features)
                warn(f"Not all features present in concatenation configuration, missing: {missing}")

    def raw_data(self, idx: int) -> np.ndarray:
        return self._dataframe.iloc[idx].values.copy()

    @property
    def combos(self) -> list[str] | None:
        if self._concat not in (True, False, None):
            return list(self.shape.keys())

    @property
    def shape(self) -> np.ndarray | dict[str, np.ndarray]:
        return self._encoding_width

    @property
    def fjs(self):
        return self._fjs

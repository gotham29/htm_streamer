from __future__ import annotations

import os
import sys
import time
import pandas as pd
from htm.bindings.sdr import SDR

from htm_source.data.data_streamer import DataStreamer
from htm_source.utils.general import choose_features, split_features

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from htm_source.utils import get_args
from htm_source.model.htm_model import HTMModule
from htm_source.model.htm_pyramid import ModelPyramid
from htm_source.utils.sdr import sdr_max_pool, sdr_merge, concat_shapes, flatten_shape
from htm_source.data.types import HTMType, to_htm_type

from logger import setup_applevel_logger

log = setup_applevel_logger(file_name=os.path.join(_SOURCE_DIR, 'logs.log'))


def new_run(data: pd.DataFrame,
            data_cfg: dict,
            model_cfg: dict,
            iter_print: int = 100,
            train_percent: float = 1.0,
            learn_schedule=None):
    ds = DataStreamer(data,
                      features_cfg=data_cfg['features'],
                      encoders_cfg=model_cfg['models_encoders'],
                      train_percent=train_percent,
                      prepare=True, concat=False)

    model_x = HTMModule(input_dims=ds.shape['x'],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'],
                        anomaly_score=True,
                        learn_schedule=learn_schedule)

    model_y = HTMModule(input_dims=ds.shape['y'],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'] * 2,
                        anomaly_score=True,
                        learn_schedule=learn_schedule)

    xy_shape = flatten_shape(model_y.output_dim)
    maxpool = 2
    xy_shape[0] //= maxpool
    model_xy = HTMModule(input_dims=xy_shape,
                         sp_cfg=model_cfg['models_params']['sp'],
                         tm_cfg=model_cfg['models_params']['tm'],
                         seed=model_cfg['models_params']['seed'] * 3,
                         anomaly_score=True,
                         learn_schedule=learn_schedule)

    for idx, item in enumerate(ds):
        x_hat = model_x(item['x'])
        y_hat = model_y(item['y'])
        mp_x = sdr_max_pool(x_hat, maxpool).flatten()
        mp_y = sdr_max_pool(y_hat, maxpool).flatten()
        xy_sdr = sdr_merge(mp_x, mp_y, mode='sd')
        model_xy(xy_sdr)

        if idx % iter_print == 0:
            print(f"Done {idx} Iterations")

    return model_x, model_y, model_xy


def water_run(data: pd.DataFrame,
              data_cfg: dict,
              model_cfg: dict,
              iter_print: int = 100,
              train_percent: float = 1.0,
              learn_schedule=None):
    ds = DataStreamer(data,
                      features_cfg=data_cfg['features'],
                      encoders_cfg=model_cfg['models_encoders'],
                      train_percent=train_percent,
                      prepare=True, concat=False)

    model_l = HTMModule(input_dims=ds.shape['level'],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'],
                        anomaly_score=True,
                        learn_schedule=learn_schedule)

    f_shape = concat_shapes(ds.shape['f_in'], ds.shape['f_out'])
    model_f = HTMModule(input_dims=f_shape,
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'] * 2,
                        anomaly_score=True,
                        learn_schedule=learn_schedule)

    xy_shape = flatten_shape(model_l.output_dim)
    maxpool = 2
    xy_shape[0] //= maxpool
    model_xy = HTMModule(input_dims=xy_shape,
                         sp_cfg=model_cfg['models_params']['sp'],
                         tm_cfg=model_cfg['models_params']['tm'],
                         seed=model_cfg['models_params']['seed'] * 3,
                         anomaly_score=True,
                         learn_schedule=learn_schedule)

    for idx, item in enumerate(ds):
        # hat = model(item)
        item_f = SDR(f_shape).concatenate([item['f_in'], item['f_out']])
        f_hat = model_f(item_f)
        l_hat = model_l(item['level'])
        mp_f = sdr_max_pool(f_hat, maxpool).flatten()
        mp_l = sdr_max_pool(l_hat, maxpool).flatten()
        xy_sdr = sdr_merge(mp_f, mp_l, mode='u')
        model_xy(xy_sdr)

        if idx % iter_print == 0:
            print(f"Done {idx} Iterations")

    return model_f, model_l, model_xy


def corr_run(data: pd.DataFrame,
             data_cfg: dict,
             model_cfg: dict,
             iter_print: int = 100,
             train_percent: float = 1.0,
             learn_schedule=None):
    ds = DataStreamer(data,
                      features_cfg=data_cfg['features'],
                      encoders_cfg=model_cfg['models_encoders'],
                      train_percent=train_percent,
                      prepare=True, concat=False)

    model_1 = HTMModule(input_dims=ds.shape['seq1'],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'],
                        anomaly_score=True,
                        learn_schedule=learn_schedule)

    model_2 = HTMModule(input_dims=ds.shape['seq2'],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'] * 2,
                        anomaly_score=True,
                        learn_schedule=learn_schedule)

    xy_shape = flatten_shape(model_1.output_dim)
    maxpool = 2
    xy_shape[0] //= maxpool
    model_xy = HTMModule(input_dims=xy_shape,
                         sp_cfg=model_cfg['models_params']['sp'],
                         tm_cfg=model_cfg['models_params']['tm'],
                         seed=model_cfg['models_params']['seed'] * 3,
                         anomaly_score=True,
                         learn_schedule=learn_schedule)

    for idx, item in enumerate(ds):
        hat_1 = model_1(item['seq1'])
        hat_2 = model_2(item['seq2'])
        mp_1 = sdr_max_pool(hat_1, maxpool).flatten()
        mp_2 = sdr_max_pool(hat_2, maxpool).flatten()
        xy_sdr = sdr_merge(mp_1, mp_2, mode='u')
        model_xy(xy_sdr)

        if idx % iter_print == 0:
            print(f"Done {idx} Iterations")

    return model_1, model_2, model_xy


def corr_run2(data: pd.DataFrame,
              data_cfg: dict,
              model_cfg: dict,
              iter_print: int = 100,
              train_percent: float = 1.0,
              learn_schedule=None):
    ds = DataStreamer(data,
                      features_cfg=data_cfg['features'],
                      encoders_cfg=model_cfg['models_encoders'],
                      train_percent=train_percent,
                      prepare=True)

    model_1 = HTMModule(input_dims=ds.shape['seq1'],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'],
                        anomaly_score=True,
                        learn_schedule=learn_schedule)

    model_2 = HTMModule(input_dims=ds.shape['seq2'],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'] * 2,
                        anomaly_score=True,
                        learn_schedule=learn_schedule)

    model_3 = HTMModule(input_dims=ds.shape['seq3'],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'] * 3,
                        anomaly_score=True,
                        learn_schedule=learn_schedule)

    model_4 = HTMModule(input_dims=ds.shape['seq4'],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'] * 4,
                        anomaly_score=True,
                        learn_schedule=learn_schedule)
    xy_shape = flatten_shape(model_1.output_dim)
    maxpool = 2
    xy_shape[0] //= maxpool
    model_xy = HTMModule(input_dims=xy_shape,
                         sp_cfg=model_cfg['models_params']['sp'],
                         tm_cfg=model_cfg['models_params']['tm'],
                         seed=model_cfg['models_params']['seed'] * 5,
                         anomaly_score=True,
                         learn_schedule=learn_schedule)

    model_zw = HTMModule(input_dims=xy_shape,
                         sp_cfg=model_cfg['models_params']['sp'],
                         tm_cfg=model_cfg['models_params']['tm'],
                         seed=model_cfg['models_params']['seed'] * 6,
                         anomaly_score=True,
                         learn_schedule=learn_schedule)

    model_wxyz = HTMModule(input_dims=xy_shape,
                           sp_cfg=model_cfg['models_params']['sp'],
                           tm_cfg=model_cfg['models_params']['tm'],
                           seed=model_cfg['models_params']['seed'] * 7,
                           anomaly_score=True,
                           learn_schedule=learn_schedule)

    for idx, item in enumerate(ds):
        hat_1 = model_1(item['seq1'])
        hat_2 = model_2(item['seq2'])
        hat_3 = model_3(item['seq3'])
        hat_4 = model_4(item['seq4'])

        mp_1 = sdr_max_pool(hat_1, maxpool).flatten()
        mp_2 = sdr_max_pool(hat_2, maxpool).flatten()
        mp_3 = sdr_max_pool(hat_3, maxpool).flatten()
        mp_4 = sdr_max_pool(hat_4, maxpool).flatten()

        xy_sdr = sdr_merge(mp_1, mp_2, mode='u')
        zw_sdr = sdr_merge(mp_3, mp_4, mode='u')
        hat_xy = model_xy(xy_sdr)
        hat_zw = model_zw(zw_sdr)

        mp_5 = sdr_max_pool(hat_xy, maxpool).flatten()
        mp_6 = sdr_max_pool(hat_zw, maxpool).flatten()
        wxyz_sdr = sdr_merge(mp_5, mp_6, mode='u')
        model_wxyz(wxyz_sdr)

        if idx % iter_print == 0:
            print(f"Done {idx} Iterations")

    return model_1, model_2, model_3, model_4, model_xy, model_zw, model_wxyz


def occupancy_run(data: pd.DataFrame,
                  data_cfg: dict,
                  model_cfg: dict,
                  iter_print: int = 100,
                  train_percent: float = 1.0,
                  merge_mode: str = 'u',
                  learn_schedule=None,
                  lazy_init: bool = False):
    ds = DataStreamer(data,
                      features_cfg=data_cfg['features'],
                      encoders_cfg=model_cfg['models_encoders'],
                      train_percent=train_percent,
                      concat=choose_features(data.columns.tolist(), 4, 2),
                      prepare=True)

    combos = ds.combos
    print(f"Features chosen: {combos}")

    t0 = time.perf_counter()
    print("Building models... ", end='')

    model_1 = HTMModule(input_dims=ds.shape[combos[0]],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'],
                        anomaly_score=True,
                        learn_schedule=learn_schedule,
                        max_pool=2,
                        flatten=True,
                        lazy_init=lazy_init)

    model_2 = HTMModule(input_dims=ds.shape[combos[1]],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'] * 2,
                        anomaly_score=True,
                        learn_schedule=learn_schedule,
                        max_pool=2,
                        flatten=True,
                        lazy_init=lazy_init)

    model_3 = HTMModule(input_dims=ds.shape[combos[2]],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'] * 3,
                        anomaly_score=True,
                        learn_schedule=learn_schedule,
                        max_pool=2,
                        flatten=True,
                        lazy_init=lazy_init)

    model_4 = HTMModule(input_dims=ds.shape[combos[3]],
                        sp_cfg=model_cfg['models_params']['sp'],
                        tm_cfg=model_cfg['models_params']['tm'],
                        seed=model_cfg['models_params']['seed'] * 4,
                        anomaly_score=True,
                        learn_schedule=learn_schedule,
                        max_pool=2,
                        flatten=True,
                        lazy_init=lazy_init)

    model_xy = HTMModule(input_dims=model_1.output_dim,
                         sp_cfg=model_cfg['models_params']['sp'],
                         tm_cfg=model_cfg['models_params']['tm'],
                         seed=model_cfg['models_params']['seed'] * 5,
                         anomaly_score=True,
                         learn_schedule=learn_schedule,
                         max_pool=2,
                         flatten=True,
                         lazy_init=lazy_init)

    model_zw = HTMModule(input_dims=model_3.output_dim,
                         sp_cfg=model_cfg['models_params']['sp'],
                         tm_cfg=model_cfg['models_params']['tm'],
                         seed=model_cfg['models_params']['seed'] * 6,
                         anomaly_score=True,
                         learn_schedule=learn_schedule,
                         max_pool=2,
                         flatten=True,
                         lazy_init=lazy_init)

    model_wxyz = HTMModule(input_dims=model_zw.output_dim,
                           sp_cfg=model_cfg['models_params']['sp'],
                           tm_cfg=model_cfg['models_params']['tm'],
                           seed=model_cfg['models_params']['seed'] * 7,
                           anomaly_score=True,
                           learn_schedule=learn_schedule,
                           max_pool=2,
                           flatten=True,
                           lazy_init=lazy_init)

    model_dict = {combos[0]: model_1,
                  combos[1]: model_2,
                  combos[2]: model_3,
                  combos[3]: model_4,
                  f"{combos[0]}+{combos[1]}": model_xy,
                  f"{combos[2]}+{combos[3]}": model_zw,
                  "Final": model_wxyz}

    print(f"Done in {time.perf_counter() - t0:.1f} sec")

    for idx, item in enumerate(ds):
        bottom_layer = [model_dict[key](item[key]) for key in item.keys()]

        xy_sdr = sdr_merge(*bottom_layer[:2], mode=merge_mode)
        zw_sdr = sdr_merge(*bottom_layer[2:], mode=merge_mode)

        hat_xy = model_xy(xy_sdr)
        hat_zw = model_zw(zw_sdr)

        wxyz_sdr = sdr_merge(hat_xy, hat_zw, mode=merge_mode)

        model_wxyz(wxyz_sdr)

        if idx % iter_print == 0:
            print(f"Done {idx} Iterations")

    return model_dict

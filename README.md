## Setup Module

## 1) htm.core
### Pip
N/A (as of now)
### Git
From command line:
* **Clone htm.core repo**: `git clone https://github.com/htm-community/htm.core.git`
* **CD to htm.core dir**: `cd htm.core`
* **Create fresh env**: `conda create -n htm_env python=3.9.7`
* **Switch to fresh env**: `conda activate htm_env`
* **Install packages**: `pip install -r requirements.txt`
* **Run setup.py**: `python setup.py install`

## 2) htm_streamer
### Pip
From command line:
* **Install packages**: `pip install git+https://github.com/gotham29/htm_streamer.git`
### Git
From command line:
* **Clone htm_streamer repo**: `git clone https://github.com/gotham29/htm_streamer.git`
* **CD to htm_streamer dir**:
  * `cd ..`
  * `cd htm_streamer`
* **Install packages**: `pip install -r requirements.txt`

## 3) Run Integration Test
From command line:
* **Run**: `python tests/integration_test.py`
* **Get results**: `htm_streamer/tests`

<br/>

## Quickstart

From Python:
* **Import Functions**:
  * `import os`
  * `import pandas as pd`
  * `from htm_source.config import load_config`
  * `from htm_source.pipeline.htm_batch_runner import run_batch`
* **Load Config & Data**:
  * `config_path = os.path.join(os.getcwd(), 'data', 'config.yaml')`
  * `data_path = os.path.join(os.getcwd(), 'data', 'batch', 'sample_timeseries.csv')`
  * `config = load_config(config_path)`
  * `data = pd.read_csv(data_path)`
* **Set Config**:
  * `my_features = data.columns[1:4]`
  * `timestep_tostop_sampling = 40`
  * `timestep_tostop_learning = 4000`
  * `timestep_tostop_running = 5000`
  * `model_for_each_feature = True`
  * `features_invalid = [ f for f in my_features if f not in data]`
  * `assert len(features_invalid) == 0, f"features not found --> {sorted(features_invalid)}"`
  * `config['features'] = my_features`
  * `config['timesteps_stop']['sampling'] = timestep_tostop_sampling`
  * `config['timesteps_stop']['learning'] = timestep_tostop_learning`
  * `config['timesteps_stop']['running'] = timestep_tostop_running`
  * `config['models_state']['model_for_each_feature'] = model_for_each_feature`
* **Train New HTM Models**:
  * `features_models, features_outputs = run_batch(cfg=config, config_path=None, learn=True, data=data, iter_print=100, features_models={})`
* **Run Existing HTM Models**:
  * `features_models, features_outputs = run_batch(cfg=config, config_path=None, learn=False, data=data, iter_print=100, features_models=features_models)`
* **Collect Models and Outputs**:
  * `f1 = my_features[0]`
  * `f1_model = features_models[ f1 ]`
  * `f1_anomaly_scores = features_outputs[f1]['anomaly_score']`
  * `f1_anomaly_liklihoods = features_outputs[f1]['anomaly_likelihood']`
  * `f1_prediction_counts = features_outputs[f1]['pred_count']`

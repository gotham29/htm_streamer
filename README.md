## Steps to run locally

### Prerequisites:
* xxx

### 1) Setup htm.core module
From command line:
* **Clone htm.core repo**: `git clone https://github.com/htm-community/htm.core.git`
* **CD to htm.core dir**: `cd htm.core`
* **Create fresh env**: `conda create -n htm_env python=3.9.7`
* **Switch to fresh env**: `conda activate htm_env`
* **Install packages**: `pip install -r requirements.txt`
* **Run setup.py**: `python setup.py --install`

### 2) Setup htm_streamer module
From command line:
* **Clone htm_streamer repo**: `git clone https://github.com/gotham29/htm_streamer.git`
* **CD to htm_streamer dir**:
  * `cd ..`
  * `cd htm_streamer`

### 3) Get Sample Data
* **Download from Box**: https://xxx
* **Drag data folder to repo root**: Create htm_streamer/data

### 4) Run htm_streamer Integration Tests
From command line:
* **CD to tests dir**: `cd tests`
* **Run**: `python integration_tests.py`
* **Get results**: Find in htm_streamer/results
  * **Training**:
    * Complete when .pkl models for all targets found in --> `htm_streamer/models`
  * **Inference**:
    * Complete when `xxx` file found in --> `htm_streamer/results`

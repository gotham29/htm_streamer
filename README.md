## Steps to run locally

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
  * `cd htm_streamer`
  
### 4) Run Integration Test
From command line:
* **Run**: `python tests/integration_test.py`
* **Get results**: Find in: `htm_streamer/tests`
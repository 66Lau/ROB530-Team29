# Skateboard InEKF environment #


## Installation
```bash
cd rsl_rl
pip install -e .
cd legged_gym
pip install -e .
```

### Usage ###


1. Play base policy:
```bash
roscore
conda activate rob530-final
export LD_LIBRARY_PATH=/home/lau/anaconda3/envs/rob530-final/lib/
# collect data
python play.py --exptid=01-12-10
# replay data
python play_msgs.py enu_data.jsonl
```

2. runing drift
```bash
cd drift
./build_ros.sh
rosrun mini_cheetah
```

4. log data
```bash
python log_msgs.py
```

5. plot data
```bash
python plot_fig.py
```
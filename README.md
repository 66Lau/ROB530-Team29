# ROB530-Team29 #

Jikai Zhang, Wenhan Jiangm, Hang Liu

## Installation


plz follow the drift installation and compilation here : https://github.com/UMich-CURLY/drift  
and follow isaac gym installation here: https://github.com/leggedrobotics/legged_gym


### Usage ###


1. Play base policy:
```bash
roscore
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
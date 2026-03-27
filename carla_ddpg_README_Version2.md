# CARLA DDPG Autonomous Driving

Goals:
- Lane keeping on straight/curved roads
- Overtake parked vehicles
- Overtake slow vehicles while avoiding oncoming traffic

## Setup
1. Install CARLA 0.9.14+ and start the server:
   ```
   ./CarlaUE4.sh -carla-server -quality-level=Epic -fps=20
   ```
2. Create a venv and install deps:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Training
```
python train.py
```

## Evaluation
```
python evaluate.py
```

## Architecture (ASCII)
```
                +---------------------+
                |   CARLA Simulator   |
                +----------+----------+
                           |
                 obs (RGB, LiDAR, ego state)
                           |
                     +-----v-----+
                     |  Actor    |----> action (steer, throttle, brake)
                     +-----+-----+
                           |
                     +-----v-----+
                     |  Critic   |<----- action, obs
                     +-----+-----+
                           |
                 experience tuples -> Replay Buffer
```

## Notes
- Synchronous mode with fixed delta (0.05s).
- Overtaking safety layer relaxes lane penalty when oncoming lane clear (LiDAR left sectors).
- Dense reward (speed, lateral/heading penalties, comfort, overtake bonus, collision/lane penalties) configurable in `utils/config.py`.
- All actors are cleaned up in `close()`/finally blocks to prevent CARLA leaks.
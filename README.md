# BalatroBotLearning

Reinforcement Learning for Balatro

## Setup Instructions

1. Install [Steamodded](https://github.com/Steamodded/smods/wiki/Installing-Steamodded-windows)
2. Clone the repository to your local machine
3. Using Python 3.11, install requirements with `pip install -r requirements.txt` and `pip install -r requirements-ppo.txt`
4. Ensure that the game path in `balatro_connection.start_balatro_instance` is where your Balatro game executable is located (You must already own Balatro & have it installed).
5. Test the game connection by running `balatro_connection.py`, which should launch Balatro, and after around a 10 sec delay, will start playing the game automatically.

## Training Instructions

Training the model starts through `train_ppo.py`, and it can either automaticallly start the game or use an existing connection depending on what works for you.

1. Optional: start Balatro manually with bot port (e.g. 12348), or let the env start it.
2. Run: `python train_ppo.py [--no-run-balatro] [--port 12348] [--total-timesteps 50000]`. Currently it's very slow as it runs only one Balatro instance on your computer and uses it to train. It will print a status update every 128 trainsteps.
3. Run `tensorboard --logdir logs/ppo_balatro` to start a live Tensorboard server for live metrics.

Logs will automatically be saved to `tests/` directory, and models will be saved by default as `ppo_balatro.zip` once the training process is stopped. We keep the models trained in the `models/` directory, and specific logs we want to show in git should be renamed so that it doesn't start with MaskablePPO.

The balatrobot mod & API was developed by [besteon](https://github.com/besteon/balatrobot)
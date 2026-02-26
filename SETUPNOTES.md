## Setup Process

Cloned khang branch balatroo sim repo
Created new conda env & installed from requirements.txt
Ran rl_training.py

Got error about directory not created, updated storage path in rl_training
Ran rl_training.py
It seems to work, though I got a few warning errors it is running.

Only thing that seems strange is that no samples returned from remote workers.
Status shows up as RUNNING. Output looks like this:


2026-02-24 22:42:29,712 WARNING tune.py:219 -- Stop signal received (e.g. via SIGINT/Ctrl+C), ending Ray Tune run. This will try to checkpoint the experiment state one last time. Press CTRL+C (or send SIGINT/SIGKILL/SIGTERM) to skip.
2026-02-24 22:42:29,737 INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/Users/abhin/balatrobot-simulation/run_data/blind_shop' in 0.0251s.
Trial status: 1 RUNNING
Current time: 2026-02-24 22:42:29. Total running time: 2min 40s
Logical resource usage: 6.0/16 CPUs, 0/0 GPUs
╭───────────────────────────────────╮
│ Trial name               status   │
├───────────────────────────────────┤
│ blind_shop_c81d5_00000   RUNNING  │
╰───────────────────────────────────╯

Stopped the current instance, reran it to see if it would work twice. It did work.


"""
Train a PPO policy on a single Balatro instance using MaskablePPO (sb3-contrib).

Usage:
  1. Install: pip install -r requirements-ppo.txt  (gymnasium, stable-baselines3, sb3-contrib)
  2. Optional: start Balatro manually with bot port (e.g. 12348), or let the env start it.
  3. Run: python train_ppo.py [--no-run-balatro] [--port 12348] [--total-timesteps 50000]
  4. View metrics: tensorboard --logdir logs/ppo_balatro  (or set --tensorboard-log "" to disable)

The env starts Balatro.exe by default; use --no-run-balatro if you start it yourself.
Custom Balatro metrics (round reached, chips vs blind, etc.) are logged via callbacks.

For evaluation / rollout with masking:
  from sb3_contrib.common.maskable.utils import get_action_masks
  obs, info = env.reset()
  while True:
      action_masks = get_action_masks(env)
      action, _ = model.predict(obs, action_masks=action_masks)
      obs, reward, terminated, truncated, info = env.step(action)
      if terminated or truncated:
          break
"""

from __future__ import annotations

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Balatro env")
    parser.add_argument("--port", type=int, default=12346, help="Bot port for Balatro")
    parser.add_argument("--no-run-balatro", action="store_true", help="Do not start Balatro; connect to existing")
    parser.add_argument("--total-timesteps", type=int, default=50_000, help="Training steps")
    parser.add_argument("--save-path", type=str, default="ppo_balatro", help="Path to save model")
    parser.add_argument("--deck", type=str, default="Blue Deck")
    parser.add_argument("--stake", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Truncate episode after N steps (optional)")
    parser.add_argument("--tensorboard-log", type=str, default="logs/ppo_balatro", help="TensorBoard log dir (set to '' to disable)")
    parser.add_argument("--reward-chips-scale", type=float, default=0.001, help="Scale for chips-per-hand reward (0 to disable)")
    parser.add_argument("--reward-round-clear", type=float, default=1.0, help="Bonus when beating a blind (0 to disable)")
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.evaluation import evaluate_policy
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    except ImportError as e:
        raise ImportError(
            "Train script requires: pip install gymnasium stable-baselines3 sb3-contrib"
        ) from e

    from balatro_env import BalatroEnv
    from callbacks import BalatroTensorboardCallback

    run_balatro = not args.no_run_balatro
    env = BalatroEnv(
        bot_port=args.port,
        deck=args.deck,
        stake=args.stake,
        seed=str(args.seed) if args.seed is not None else None,
        run_balatro=run_balatro,
        reward_chips_per_hand_scale=args.reward_chips_scale,
        reward_round_clear=args.reward_round_clear,
        max_steps_per_episode=args.max_steps,
    )

    tensorboard_log = args.tensorboard_log if args.tensorboard_log else None
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=args.seed,
        tensorboard_log=tensorboard_log,
    )

    callbacks = [BalatroTensorboardCallback()]
    if tensorboard_log:
        print(f"TensorBoard logs: {os.path.abspath(tensorboard_log)}  (run: tensorboard --logdir {tensorboard_log})")
    print("Starting training. Episodes are full runs; Ctrl+C to stop.")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        model.save(args.save_path)
        print(f"Model saved to {args.save_path}")

    env.close()


if __name__ == "__main__":
    main()

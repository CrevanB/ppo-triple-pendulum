"""
Train triple inverted pendulum -- Full State Space (Unified)

Linearly ramps init_noise from near-upright to fully hanging down (pi).
No early angle termination.

Two Modes (--balance-mode flag):
  - Swing-up (default): the env resets with cart position tightly bounded
    (+/-0.05) and angles/velocities widened together as init_noise ramps,
    covering the full range from near-upright out to hanging straight down.
    The policy has to learn to pump energy in and swing the pendulum up.
  - Balance (--balance-mode): the env resets using separate, narrower
    constants (BALANCE_INIT_X_NOISE, BALANCE_INIT_VEL_NOISE) for cart
    position and velocity, while init_noise still ramps the angles via the
    same curriculum. The pendulum always starts near upright with a
    meaningful starting velocity from step one -- there's no swing-up
    phase, only fine stabilization. Everything else (dynamics, reward,
    step, termination) is identical between the two modes; only reset()
    differs.

Retains critical infrastructure:
  - EvalCallback.best_mean_reward seed on resume (prevents silent overwrites).
  - Matched VecNormalize saves on checkpoints and best-model.
  - Train-env difficulty sync on resume.
  - Adaptive entropy coefficient: nudges ent_coef up when policy std drops
    too low (risk of premature convergence / std collapse) and down when
    std rises too high (too noisy to exploit what's been learned), instead
    of using one fixed ent_coef for the whole run.

============================================================================
 HOW TO RUN
============================================================================

--- Swing-up (from scratch, hanging -> upright) ---
2M-step curriculum ramp + 3M more steps at full difficulty = 5M total.

    python train.py \
        --timesteps 5000000 \
        --curriculum-start 0.05 \
        --curriculum-end 3.14159 \
        --curriculum-timesteps 2000000 \
        --learning-rate 3e-4 \
        --disable-adaptive-entropy \
        --run-name swingup \
        --save-path ppo_triple_pendulum_swingup

--- Balance (near-upright stabilization, balance_mode=True) ---
1M-step curriculum ramp to 0.25 rad + 4M more steps at full difficulty = 5M total.

    python train.py \
        --timesteps 5000000 \
        --curriculum-start 0.01 \
        --curriculum-end 0.25 \
        --curriculum-timesteps 1000000 \
        --learning-rate 5e-5 \
        --balance-mode \
        --run-name balance \
        --save-path ppo_triple_pendulum_balance
============================================================================
"""

import argparse
import os
import math
import csv

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback, CallbackList

from triple_pendulum_env import TripleInvertedPendulumEnv


def make_env_fn(balance_mode):
    def _init():
        return TripleInvertedPendulumEnv(balance_mode=balance_mode)
    return _init

class SaveVecNormalizeOnNewBest(BaseCallback):
    def __init__(self, save_path, meta_path=None, curriculum_callback=None, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.meta_path = meta_path
        self.curriculum_callback = curriculum_callback

    def _on_step(self):
        try:
            self.training_env.save(self.save_path)
            if self.verbose:
                print(f"[best-model] saved matching VecNormalize -> {self.save_path}")
        except Exception as e:
            print(f"[best-model] WARNING: failed to save VecNormalize alongside best_model.zip: {e}")
        if self.meta_path is not None:
            try:
                import json
                meta = {"num_timesteps": self.num_timesteps}
                if self.curriculum_callback is not None:
                    meta["init_noise"] = self.curriculum_callback.current
                with open(self.meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                print(f"[best-model] WARNING: failed to save {self.meta_path}: {e}")
        return True


class VecNormalizeCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="pendulum",
                 curriculum_callback=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.curriculum_callback = curriculum_callback

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            base = f"{self.name_prefix}_{self.num_timesteps}_steps"
            vecnorm_path = os.path.join(self.save_path, f"{base}_vecnormalize.pkl")
            meta_path = os.path.join(self.save_path, f"{base}_meta.json")
            try:
                self.training_env.save(vecnorm_path)
            except Exception as e:
                print(f"[vecnorm-checkpoint] WARNING: failed to save {vecnorm_path}: {e}")
            try:
                import json
                meta = {"num_timesteps": self.num_timesteps}
                if self.curriculum_callback is not None:
                    meta["init_noise"] = self.curriculum_callback.current
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                print(f"[vecnorm-checkpoint] WARNING: failed to save {meta_path}: {e}")
        return True


class TimeBasedCurriculum(BaseCallback):
    """
    Linearly ramps init_noise from `start` to `end` over `ramp_timesteps`.
    Cannot stall, cannot get stuck, just steadily increases the difficulty
    across the entire state space.
    """
    def __init__(self, start, end, ramp_timesteps, eval_env=None, check_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.ramp_timesteps = ramp_timesteps
        self.eval_env = eval_env
        self.check_freq = check_freq
        self._next_check = check_freq
        self.current = start

    def _on_step(self):
        if self.num_timesteps >= self._next_check:
            self._next_check += self.check_freq
            frac = min(1.0, self.num_timesteps / self.ramp_timesteps)
            self.current = self.start + frac * (self.end - self.start)

            self.training_env.env_method("set_difficulty", self.current)
            if self.eval_env is not None:
                self.eval_env.env_method("set_difficulty", self.current)

            if self.verbose:
                print(f"[curriculum] timestep={self.num_timesteps} init_noise -> {self.current:.3f}")
        return True


class AdaptiveEntropyCallback(BaseCallback):
    """
    Option to use dynamically adjustment model.ent_coef based on the policy's current action
    std, instead of using one fixed value for the whole run.

    - If std drops below `target_std_min`, ent_coef is nudged UP (multiply by
      adjust_factor) to push exploration back up before the policy locks
      onto a narrow, possibly premature strategy (the std-collapse failure
      mode seen previously: std -> 0.28, approx_kl and clip_fraction blowing
      past healthy ranges, episode length flatlining).
    - If std rises above `target_std_max`, ent_coef is nudged DOWN (divide by
      adjust_factor) so the policy is allowed to exploit and sharpen a
      strategy it has found, rather than staying permanently noisy.
    - Within the [target_std_min, target_std_max] band, ent_coef is left
      untouched.

    ent_coef is clamped to [ent_coef_min, ent_coef_max] so the feedback loop
    can't run away in either direction. Adjustments are checked every
    `check_freq` environment steps (across all parallel envs), which should
    be on the order of one rollout (n_steps_per_env * n_envs) so each check
    reflects a full batch of fresh data rather than a noisy partial one.
    """
    def __init__(self, target_std_min=0.3, target_std_max=0.8,
                 adjust_factor=1.05, ent_coef_min=0.001, ent_coef_max=0.05,
                 check_freq=8192, verbose=1):
        super().__init__(verbose)
        self.target_std_min = target_std_min
        self.target_std_max = target_std_max
        self.adjust_factor = adjust_factor
        self.ent_coef_min = ent_coef_min
        self.ent_coef_max = ent_coef_max
        self.check_freq = check_freq
        self._next_check = check_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_check:
            return True
        self._next_check += self.check_freq

        log_std = self.model.policy.log_std.detach().cpu().numpy()
        current_std = float(np.exp(log_std).mean())

        old_coef = self.model.ent_coef
        if current_std < self.target_std_min:
            new_coef = min(old_coef * self.adjust_factor, self.ent_coef_max)
        elif current_std > self.target_std_max:
            new_coef = max(old_coef / self.adjust_factor, self.ent_coef_min)
        else:
            new_coef = old_coef

        self.model.ent_coef = new_coef

        if self.verbose and new_coef != old_coef:
            print(f"[adaptive_ent] timestep={self.num_timesteps} std={current_std:.3f}  "
                  f"ent_coef {old_coef:.4f} -> {new_coef:.4f}")

        self.logger.record("adaptive/std", current_std)
        self.logger.record("adaptive/ent_coef", new_coef)
        return True

class EvalAnalyzerCallback(BaseCallback):
    """
    Runs a deterministic eval episode periodically, logs reward components
    to TensorBoard, and saves the trajectory as a CSV for analyze_reward_components.py

    TensorBoard logging and CSV saving are decoupled — each has its own
    frequency. Only one eval episode is run per triggered step even if both
    fire on the same timestep.
    """
    def __init__(self, eval_env, tb_freq=10_000, csv_freq=100_000,
                 save_path="./eval_rollouts", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tb_freq = tb_freq
        self.csv_freq = csv_freq
        self.save_path = save_path
        self._next_tb = tb_freq
        self._next_csv = csv_freq

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        do_tb = self.num_timesteps >= self._next_tb
        do_csv = self.num_timesteps >= self._next_csv
        if not (do_tb or do_csv):
            return True

        if do_tb:
            self._next_tb += self.tb_freq
        if do_csv:
            self._next_csv += self.csv_freq

        # Run one deterministic episode
        obs = self.eval_env.reset()
        episode_data = []

        for _ in range(1500):  # max_steps fallback
            action, _ = self.model.predict(obs, deterministic=True)

            obs, rewards, dones, infos = self.eval_env.step(action)
            info = infos[0]

            episode_data.append({
                "step": len(episode_data),

                # --- Rewards ---
                "height_bonus": round(info["height_bonus"], 2),
                "alignment_reward": round(info["alignment_reward"], 2),
                "stillness_reward": round(info["stillness_reward"], 2),

                # --- Penalties ---
                "position_penalty": round(info["position_penalty"], 2),
                "effort_penalty": round(info["effort_penalty"], 2),
                "cart_velocity_penalty": round(info["cart_velocity_penalty"], 2),
                "omega_limit_penalty": round(info["omega_limit_penalty"], 2),
                "energy_limit_penalty": round(info["energy_limit_penalty"], 2),

                # --- Diagnostics ---
                "kinetic_energy": round(info["kinetic_energy"], 2),
                "potential_energy": round(info["potential_energy"], 2),
                "total_energy": round(info["total_energy"], 2),
                "pendulum_energy": round(info["pendulum_energy"], 2),
                "x": round(info["cart_position"], 2),
                "cart_velocity": round(info["cart_velocity"], 2),
                "th1": round(info["th1"], 2), "th2": round(info["th2"], 2), "th3": round(info["th3"], 2),
                "th1_dot": round(info["th1_dot"], 2), "th2_dot": round(info["th2_dot"], 2), "th3_dot": round(info["th3_dot"], 2),

                "action": round(float(action[0]), 2),
                "reward": round(float(rewards[0]), 2),
            })

            if dones[0]:
                break

        if do_tb:
            for key in [
                # --- Rewards (positive contributions) ---
                "height_bonus",
                "alignment_reward",
                "stillness_reward",

                # --- Penalties (negative contributions) ---
                "position_penalty",
                "effort_penalty",
                "cart_velocity_penalty",
                "omega_limit_penalty",
                "energy_limit_penalty",

                # --- Diagnostics (not part of reward, just physics state) ---
                "kinetic_energy",
                "potential_energy",
                "total_energy",
                "pendulum_energy",
                "cart_velocity",
                "th1_dot", "th2_dot", "th3_dot",
            ]:
                if key in info:
                    self.logger.record(f"eval_components/{key}", float(info[key]))

            if self.verbose:
                print(f"[analyzer] Logged eval components to TensorBoard at step {self.num_timesteps}")

        if do_csv:
            csv_path = os.path.join(self.save_path, f"eval_timestep_{self.num_timesteps}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=episode_data[0].keys())
                writer.writeheader()
                writer.writerows(episode_data)

            if self.verbose:
                print(f"[analyzer] Saved eval rollout at step {self.num_timesteps} -> {csv_path}")

        return True



def main():
    parser = argparse.ArgumentParser(description="Triple Pendulum -- Full State Space Linear Ramp")

    # Curriculum
    parser.add_argument("--analyzer-tb-freq", type=int, default=10_000,
                         help="How often (in timesteps) to log reward-component "
                              "breakdown to TensorBoard.")
    parser.add_argument("--analyzer-csv-freq", type=int, default=100_000,
                         help="How often (in timesteps) to save a full eval-episode "
                              "trajectory CSV to disk.")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                         help="Constant PPO learning rate. Suggested: 3e-4 for swing-up "
                              "from scratch, 5e-5 for balance fine-tuning.")
    parser.add_argument("--balance-mode", action="store_true",
                         help="Train the balance-mode env (narrow init distribution near "
                              "upright) instead of full swing-up.")
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                         help="Total training timesteps. 30M+ recommended for full state space.")
    parser.add_argument("--curriculum-start", type=float, default=0.01,
                         help="init_noise at start (radians). 0.05 is near upright.")
    parser.add_argument("--curriculum-end", type=float, default=math.pi,
                         help="init_noise at end of ramp (radians). pi is hanging straight down.")
    parser.add_argument("--curriculum-timesteps", type=int, default=None,
                         help="Timesteps over which to ramp. Defaults to --timesteps if not set.")

    # PPO Hyperparameters
    parser.add_argument("--ent-coef", type=float, default=0.02,
                         help="Initial entropy bonus. Adjusted dynamically at runtime unless "
                              "--disable-adaptive-entropy is set.")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-steps-per-env", type=int, default=2048,
                         help="Rollout length. 2048 is highly recommended for swing-up credit assignment.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--net-size", type=int, default=256)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--target-kl", type=float, default=0.03,
                         help="Early-stops a PPO update epoch if approx_kl exceeds this, preventing "
                              "the kind of runaway policy update seen when std collapsed previously.")

    # Adaptive entropy
    parser.add_argument("--disable-adaptive-entropy", action="store_true",
                         help="Disable dynamic ent_coef adjustment and just use a fixed --ent-coef "
                              "for the whole run.")
    parser.add_argument("--adaptive-ent-std-min", type=float, default=0.5,
                         help="If policy std drops below this, ent_coef is nudged up.")
    parser.add_argument("--adaptive-ent-std-max", type=float, default=0.9,
                         help="If policy std rises above this, ent_coef is nudged down.")
    parser.add_argument("--adaptive-ent-factor", type=float, default=1.05,
                         help="Multiplicative nudge applied to ent_coef per check.")
    parser.add_argument("--adaptive-ent-min", type=float, default=0.002,
                         help="Floor clamp for dynamically adjusted ent_coef.")
    parser.add_argument("--adaptive-ent-max", type=float, default=0.05,
                         help="Ceiling clamp for dynamically adjusted ent_coef.")
    parser.add_argument("--adaptive-ent-check-freq", type=int, default=8192,
                         help="How often (in timesteps) to check std and possibly adjust ent_coef. "
                              "Should be on the order of one rollout (n_steps_per_env * n_envs).")

    
    # Paths
    parser.add_argument("--save-path", type=str, default="ppo_triple_pendulum_balance")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--resume-vecnormalize", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="full_state",
                         help="Namespaces checkpoints/, logs/, best_model_*/, eval_rollouts/, "
                              "and the TensorBoard run so swing-up and balance runs don't "
                              "collide. e.g. 'swingup' or 'balance'.")
    args = parser.parse_args()
    
    best_model_dir = f"./best_model_{args.run_name}"
    checkpoint_dir = f"checkpoints/{args.run_name}"
    log_dir = f"./logs/{args.run_name}"
    eval_rollouts_dir = f"./eval_rollouts/{args.run_name}"
    
    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else None
    train_env = make_vec_env(make_env_fn(args.balance_mode), n_envs=args.n_envs, vec_env_cls=vec_env_cls)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    if args.resume_vecnormalize:
        if not os.path.exists(args.resume_vecnormalize):
            raise FileNotFoundError(f"--resume-vecnormalize file not found: {args.resume_vecnormalize}")
        print(f"Loading VecNormalize stats from {args.resume_vecnormalize}...")
        loaded_norm = VecNormalize.load(args.resume_vecnormalize, train_env.venv)
        train_env.obs_rms = loaded_norm.obs_rms
        train_env.ret_rms = loaded_norm.ret_rms
        print("Successfully loaded VecNormalize stats.")
    
    eval_env = make_vec_env(make_env_fn(args.balance_mode), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.env_method("set_difficulty", args.curriculum_start)
    
    analyzer_callback = EvalAnalyzerCallback(
        eval_env=eval_env,
        tb_freq=args.analyzer_tb_freq,
        csv_freq=args.analyzer_csv_freq,
        save_path=eval_rollouts_dir,
    )
    
    save_vecnorm_on_best = SaveVecNormalizeOnNewBest(
        save_path=f"{best_model_dir}/vecnormalize.pkl",
        meta_path=f"{best_model_dir}/meta.json",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        callback_on_new_best=save_vecnorm_on_best,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix=args.run_name,
    )
    
    ramp_timesteps = args.curriculum_timesteps if args.curriculum_timesteps is not None else args.timesteps
    curriculum_callback = TimeBasedCurriculum(
        start=args.curriculum_start,
        end=args.curriculum_end,
        ramp_timesteps=ramp_timesteps,
        eval_env=eval_env,
    )
    save_vecnorm_on_best.curriculum_callback = curriculum_callback
    
    vecnorm_checkpoint_callback = VecNormalizeCheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix=args.run_name,
        curriculum_callback=curriculum_callback,
    )

    callback_list = [curriculum_callback, eval_callback, checkpoint_callback, vecnorm_checkpoint_callback, analyzer_callback ]

    if not args.disable_adaptive_entropy:
        adaptive_entropy_callback = AdaptiveEntropyCallback(
            target_std_min=args.adaptive_ent_std_min,
            target_std_max=args.adaptive_ent_std_max,
            adjust_factor=args.adaptive_ent_factor,
            ent_coef_min=args.adaptive_ent_min,
            ent_coef_max=args.adaptive_ent_max,
            check_freq=args.adaptive_ent_check_freq,
        )
        callback_list.append(adaptive_entropy_callback)

    callbacks = CallbackList(callback_list)

    policy_kwargs = dict(net_arch=dict(pi=[args.net_size, args.net_size], vf=[args.net_size, args.net_size]))

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,  
        n_steps=args.n_steps_per_env,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=args.target_kl,
        tensorboard_log="../tb_logs/",
        verbose=1,
    )

    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume-from checkpoint not found: {args.resume_from}")
        print(f"Loading pre-trained model from {args.resume_from}...")

        model = PPO.load(args.resume_from, env=train_env, tensorboard_log="../tb_logs/")
        print("Successfully loaded model and optimizer state.")
        
        # Apply CLI overrides that PPO.load() does NOT restore from the CLI
        # (it restores the checkpoint's SAVED hyperparameters instead).
        model.ent_coef = args.ent_coef
        model.target_kl = args.target_kl
        model.learning_rate = args.learning_rate
        print(f"Overriding resumed model: ent_coef={args.ent_coef}, target_kl={args.target_kl}, lr={model.learning_rate}")

        from stable_baselines3.common.evaluation import evaluate_policy
        print("Evaluating resumed model to seed EvalCallback's best-reward tracker...")
        resumed_mean_reward, resumed_std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=args.n_eval_episodes, deterministic=True
        )
        eval_callback.best_mean_reward = resumed_mean_reward
        print(f"Resumed model's current eval reward: {resumed_mean_reward:.2f} "
              f"+/- {resumed_std_reward:.2f}")

    print(f"\n{'='*60}")
    print(f"  TRIPLE PENDULUM -- {'BALANCE' if args.balance_mode else 'SWING-UP'} ({args.run_name})")
    print(f"  Total Timesteps:      {args.timesteps:,}")
    print(f"  Curriculum Ramp:      {args.curriculum_start:.3f} -> {args.curriculum_end:.3f} rad")
    print(f"  Ramp Duration:        {ramp_timesteps:,} steps")
    print(f"  Rollout Length:       {args.n_steps_per_env}")
    print(f"  Network:              [{args.net_size}, {args.net_size}]")
    print(f"  Entropy Coef (init):  {args.ent_coef}")
    print(f"  Learning Rate:        {args.learning_rate}")
    print(f"  Adaptive Entropy:     {'disabled' if args.disable_adaptive_entropy else 'enabled'}")
    if not args.disable_adaptive_entropy:
        print(f"    std target band:    [{args.adaptive_ent_std_min}, {args.adaptive_ent_std_max}]")
        print(f"    ent_coef clamp:      [{args.adaptive_ent_min}, {args.adaptive_ent_max}]")
        print(f"    check every:         {args.adaptive_ent_check_freq:,} steps")
    print(f"  Target KL:            {args.target_kl}")
    print(f"{'='*60}\n")

    # CRITICAL: Sync train env difficulty on resume
    train_env.env_method("set_difficulty", args.curriculum_start)

    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=args.run_name)

    model.save(args.save_path)
    train_env.save(args.save_path + "_vecnormalize.pkl")
    print(f"\nSaved: {args.save_path}.zip")


if __name__ == "__main__":
    main()

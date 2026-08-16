PPO Triple Inverted Pendulum Controller.

A reinforcement-learning controller for a triple inverted pendulum on a cart, trained with PPO (Proximal Policy Optimisation) via Stable-Baselines3. 

Cart damping is modelled, but joint friction, motor dynamics, and other real-world effects are not. It's an idealised, somewhat simplistic physics simulation.

Two policies work together: one learns to swing the pendulum up from hanging, and a second learns to balance it once near upright. An interactive pygame GUI lets you watch the trained controller run live and switches between the two policies automatically based on the pendulum's angle.

A force applied to the cart is the only control input. 

Swing-up — starting from hanging straight down, pump energy into the system to bring all three links up near vertical.
Balance — once near upright, make small, precise corrections to keep the pendulum stable against gravity and disturbance.

Rather than train one policy to do both, this project trains two separate PPO policies — a swing-up specialist and a balance specialist — and switches between them at runtime based on how close all three link angles are to upright.

Requires Python and following packages, stable-baselines3, gymnasium, numpy, pygame

train.py	Training script — trains either the swing-up or balance policy via PPO
triple_pendulum_env.py	Gymnasium environment defining the pendulum's observation/action space, reward, and reset logic for both modes
dynamics.py	Physics: computes accelerations from the triple-pendulum equations of motion
config.py	Physical constants (masses, lengths, damping, gravity) and reward/curriculum hyperparameters
tripple_pendulum_controller_rl_ppo.py	Interactive pygame GUI — loads both trained policies and runs them live, auto-switching between swing-up and balance
ppo_triple_pendulum_swingup.zip / .pkl	Trained swing-up policy and its VecNormalize observation stats
ppo_triple_pendulum_balance.zip / .pkl	Trained balance policy and its VecNormalize observation stats
Reward Structure

The reward function, termination logic, and success criteria are identical for both swing-up and balance training — only the reset distribution differs between the two modes (see triple_pendulum_env.py). Each step, the reward components are:

Height bonus — positive reward for each link's height (weighted cos(θ) per link); the core "get upright" signal.
Alignment reward — bonus for the three links staying straight relative to each other, gated on only once the bottom link is more than ~50° from hanging down, so it doesn't interfere with the initial swing.
Stillness reward — bonus for low angular velocity, ramped in only once the height bonus is already near maximum, rewarding settling down rather than oscillating near the top.
Penalties — cart position, control effort, cart velocity, a hard "anti-blender" penalty if any link's angular velocity exceeds a limit, and a hard energy penalty if total pendulum energy exceeds a target multiple.

Termination happens only if the cart goes off-track. A strict success condition (tight tolerances on all three angles, all three angular velocities, cart position, and cart velocity, sustained for a required number of consecutive steps) grants a one-time success bonus.
Because the reward is mode-agnostic, the difference between swing-up and balance behaviour comes entirely from where each policy starts training — hanging down with a wide initial-state distribution vs. near-upright with a narrow one — not from a different reward signal.

Training
Both policies are trained with the same script, train.py, using a --balance-mode flag to switch environment reset behavior, and a curriculum that gradually widens the initial-state distribution over training.
Swing-up (from scratch, hanging → upright). 2M-step curriculum ramp + 3M more steps at full difficulty = 5M total:

bash
python train.py \
    --timesteps 5000000 \
    --curriculum-start 0.05 \
    --curriculum-end 3.14159 \
    --curriculum-timesteps 2000000 \
    --learning-rate 3e-4 \
    --disable-adaptive-entropy \
    --run-name swingup \
    --save-path ppo_triple_pendulum_swingup

Balance (near-upright stabilization). 1M-step curriculum ramp to 0.25 radians + 4M more steps at full difficulty = 5M total:

bash
python train.py \
    --timesteps 5000000 \
    --curriculum-start 0.01 \
    --curriculum-end 0.25 \
    --curriculum-timesteps 1000000 \
    --learning-rate 5e-5 \
    --balance-mode \
    --run-name balance \
    --save-path ppo_triple_pendulum_balance

Each run writes checkpoints, best-model saves, and TensorBoard logs

Once both models are trained (or using the included pretrained .zip/.pkl files), launch the interactive GUI:

Controls:

Key	Action
SPACE	Start / pause simulation
R	Reset (hanging down)
1	Reset (near upright)
U	Toggle controller on/off
UP / DOWN	Adjust simulation speed
Mouse	Drag pendulum bobs (while paused)

The controller automatically uses the swing-up policy whenever any link is more than 0.25 rad from upright, and switches to the balance policy once all three links are within that threshold.

License

MIT 

"""
Gymnasium environment: triple inverted pendulum on a cart
Phase-Aware, Clean Swing-up reward.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dynamics import accelerations
from config import *  


class TripleInvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, balance_mode=False):
        super().__init__()
        self.balance_mode = balance_mode

        # ------------------------------------------------------------
        # Physical parameters (from config.py)
        # ------------------------------------------------------------
        self.dt = DT
        self.max_force = MAX_FORCE
        self.M = CART_MASS
        self.m1 = M1
        self.m2 = M2
        self.m3 = M3
        self.l1 = L1
        self.l2 = L2
        self.l3 = L3
        self.b = DAMPING
        self.g = GRAVITY

        self.max_steps = MAX_STEPS
        self.x_threshold = X_THRESHOLD

        # ------------------------------------------------------------
        # Curriculum variables
        # ------------------------------------------------------------
        self.init_noise = INIT_NOISE_DEFAULT
        self.max_init_noise = MAX_INIT_NOISE_DEFAULT
        self.velocity_noise_ratio = VELOCITY_NOISE_RATIO

        # ------------------------------------------------------------
        # Reward Parameters (Phase-Aware)
        # ------------------------------------------------------------
        self.height_weight_1 = SWINGUP_HEIGHT_WEIGHT_1
        self.height_weight_2 = SWINGUP_HEIGHT_WEIGHT_2
        self.height_weight_3 = SWINGUP_HEIGHT_WEIGHT_3
        
        self.position_penalty_coef = SWINGUP_POSITION_PENALTY_COEF
        self.effort_penalty_coef = SWINGUP_EFFORT_PENALTY_COEF
        self.off_track_penalty = SWINGUP_OFF_TRACK_PENALTY
        self.cart_velocity_penalty_coef = SWINGUP_CART_VELOCITY_PENALTY_COEF
        self.max_omega = SWINGUP_MAX_OMEGA
        self.max_omega_penalty = SWINGUP_MAX_OMEGA_PENALTY
        
        # Stabilization gating
        self.stillness_weight = SWINGUP_STILLNESS_WEIGHT
        self.stab_h_start = SWINGUP_STAB_HEIGHT_START
        self.stab_h_end = SWINGUP_STAB_HEIGHT_END
        self.energy_limit_mult = SWINGUP_ENERGY_LIMIT_MULT
        self.energy_excess_penalty = SWINGUP_ENERGY_EXCESS_PENALTY
        
        # Strict Success Criteria
        self.success_angle_tol = SUCCESS_ANGLE_TOL
        self.success_omega_tol = SUCCESS_OMEGA_TOL
        self.success_x_tol = SUCCESS_X_TOL
        self.success_xd_tol = SUCCESS_XD_TOL
        self.success_steps_req = SUCCESS_STEPS_REQ
        self.success_bonus = SUCCESS_BONUS
        
        # Link Alignment Reward 
        self.alignment_reward_weight = ALIGNMENT_REWARD_WEIGHT
        self.alignment_threshold = ALIGNMENT_THRESHOLD_RAD
        self.coupling_activation_threshold = ALIGNMENT_ACTIVATION_THRESHOLD_RAD

        # ------------------------------------------------------------
        # Gym spaces
        # ------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
             np.inf, np.inf, np.inf, 
             self.x_threshold * 2, np.inf],
            dtype=np.float32,
        )
        low = -high

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------
        self.state = None
        self.steps = 0
        self.balance_counter = 0

    # ================================================================
    # Curriculum
    # ================================================================
    def set_difficulty(self, init_noise):
        """Curriculum hook: widen the initial-condition distribution."""
        self.init_noise = float(np.clip(init_noise, 0.0, self.max_init_noise))

    # ================================================================
    # Dynamics
    # ================================================================
    def _deriv(self, state, F):
        x, th1, th2, th3, xd, th1d, th2d, th3d = state

        xdd, th1dd, th2dd, th3dd = accelerations(
            x, th1, th2, th3, xd, th1d, th2d, th3d, F,
            self.M, self.m1, self.m2, self.m3, 
            self.l1, self.l2, self.l3, self.g, self.b,
        )
        return np.array([xd, th1d, th2d, th3d, xdd, th1dd, th2dd, th3dd])

    def _rk4_step(self, state, F):
        dt = self.dt
        k1 = self._deriv(state, F)
        k2 = self._deriv(state + 0.5 * dt * k1, F)
        k3 = self._deriv(state + 0.5 * dt * k2, F)
        k4 = self._deriv(state + dt * k3, F)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ================================================================
    # Observation & Helpers
    # ================================================================
    def _get_obs(self):
        x, th1, th2, th3, xd, th1d, th2d, th3d = self.state
        return np.array(
            [math.sin(th1), math.cos(th1), math.sin(th2), math.cos(th2), 
             math.sin(th3), math.cos(th3), th1d, th2d, th3d, x, xd],
            dtype=np.float32,
        )

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi] for strict success checking."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # ================================================================
    # Reset
    # ================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
    
        if self.balance_mode:
            # Angle noise ramps via curriculum (0.01 -> 0.25).
            # Velocity noise stays at a fixed, meaningful level throughout —
            # the controller must handle real starting velocity from the very
            # first step, not just once the angle curriculum has widened.
            x0 = self.np_random.uniform(-BALANCE_INIT_X_NOISE, BALANCE_INIT_X_NOISE)
            th1_0 = self.np_random.uniform(-self.init_noise, self.init_noise)
            th2_0 = self.np_random.uniform(-self.init_noise, self.init_noise)
            th3_0 = self.np_random.uniform(-self.init_noise, self.init_noise)
            velocities = self.np_random.uniform(-BALANCE_INIT_VEL_NOISE, BALANCE_INIT_VEL_NOISE, size=(4,))
        else:
            # existing swing-up reset, unchanged
            velocity_noise = self.init_noise * self.velocity_noise_ratio
            x0 = self.np_random.uniform(-0.05, 0.05)
            th1_0 = self.np_random.uniform(-self.init_noise, self.init_noise)
            th2_0 = self.np_random.uniform(-self.init_noise, self.init_noise)
            th3_0 = self.np_random.uniform(-self.init_noise, self.init_noise)
            velocities = self.np_random.uniform(-velocity_noise, velocity_noise, size=(4,))
    
        self.state = np.concatenate([[x0, th1_0, th2_0, th3_0], velocities])
        self.steps = 0
        self.balance_counter = 0
        return self._get_obs(), {}


    # ================================================================
    # Step
    # ================================================================
    def step(self, action):
        # ------------------------------------------------------------
        # Apply bounded action & Integrate
        # ------------------------------------------------------------
        F = float(np.clip(action[0], -1.0, 1.0)) * self.max_force
        self.state = self._rk4_step(self.state, F)
        self.state = np.clip(self.state, -1000.0, 1000.0) # Safety clamp
        self.steps += 1

        x, th1, th2, th3, xd, th1d, th2d, th3d = self.state

        # ------------------------------------------------------------
        # Pre-calculate Trig
        # ------------------------------------------------------------
        c1, s1 = math.cos(th1), math.sin(th1)
        c2, s2 = math.cos(th2), math.sin(th2)
        c3, s3 = math.cos(th3), math.sin(th3)

        # ------------------------------------------------------------
        # Termination: ONLY off-track (Success overrides this below)
        # ------------------------------------------------------------
        off_track = abs(x) > self.x_threshold
        terminated = bool(off_track)
        truncated = self.steps >= self.max_steps
        
        # ------------------------------------------------------------
        # Physics Diagnostics (lab-frame + pendulum-relative)
        # ------------------------------------------------------------
        vx1 = xd + self.l1 * th1d * c1
        vy1 = -self.l1 * th1d * s1
        vx2 = vx1 + self.l2 * th2d * c2
        vy2 = vy1 - self.l2 * th2d * s2
        vx3 = vx2 + self.l3 * th3d * c3
        vy3 = vy2 - self.l3 * th3d * s3
        
        kinetic_energy = (
            0.5 * self.M * xd**2
            + 0.5 * self.m1 * (vx1**2 + vy1**2)
            + 0.5 * self.m2 * (vx2**2 + vy2**2)
            + 0.5 * self.m3 * (vx3**2 + vy3**2)
        )
        
        potential_energy = (
            self.m1 * self.g * self.l1 * c1
            + self.m2 * self.g * (self.l1 * c1 + self.l2 * c2)
            + self.m3 * self.g * (self.l1 * c1 + self.l2 * c2 + self.l3 * c3)
        )
        
        total_energy = kinetic_energy + potential_energy
        
        # pendulum-only (cart-relative) — used for the energy hard limit
        vx1_rel = self.l1 * th1d * c1
        vy1_rel = vy1
        vx2_rel = vx1_rel + self.l2 * th2d * c2
        vy2_rel = vy1_rel - self.l2 * th2d * s2
        vx3_rel = vx2_rel + self.l3 * th3d * c3
        vy3_rel = vy2_rel - self.l3 * th3d * s3
        
        pendulum_kinetic_energy = (
            0.5 * self.m1 * (vx1_rel**2 + vy1_rel**2)
            + 0.5 * self.m2 * (vx2_rel**2 + vy2_rel**2)
            + 0.5 * self.m3 * (vx3_rel**2 + vy3_rel**2)
        )
        
        pendulum_energy = pendulum_kinetic_energy + potential_energy
        
        # ------------------------------------------------------------
        # HARD SPEED LIMIT (Anti-Blender)
        # ------------------------------------------------------------
        omega_limit_penalty = 0.0
        if (abs(th1d) > self.max_omega or
            abs(th2d) > self.max_omega or
            abs(th3d) > self.max_omega):
            omega_excess = max(0.0, max(abs(th1d), abs(th2d), abs(th3d)) - self.max_omega)
            omega_limit_penalty = self.max_omega_penalty * omega_excess**2
        
        # ------------------------------------------------------------
        # HARD ENERGY LIMIT
        # ------------------------------------------------------------
        energy_excess = max(0.0, pendulum_energy - E_TARGET * self.energy_limit_mult)
        energy_excess = min(energy_excess, 20.0)
        energy_limit_penalty = self.energy_excess_penalty * energy_excess**2
        # ------------------------------------------------------------
        # A. Swing-up Reward (Height + Mild Cart/Effort)
        # ------------------------------------------------------------
        height_bonus = (self.height_weight_1 * c1 + 
                        self.height_weight_2 * c2 + 
                        self.height_weight_3 * c3)

        position_penalty = self.position_penalty_coef * x ** 2
        effort_penalty = self.effort_penalty_coef * (F / self.max_force) ** 2
        cart_velocity_penalty = self.cart_velocity_penalty_coef * xd**2  

        # ------------------------------------------------------------
        # B. Near-Top Stabilization (Implicit Phase Transition)
        # ------------------------------------------------------------
        # Smooth linear ramp from 0.0 to 1.0 as height goes from 3.0 to 4.5
        near_top = max(0.0, (height_bonus - self.stab_h_start) / (self.stab_h_end - self.stab_h_start))
        
        velocity_sumsq = th1d**2 + th2d**2 + th3d**2
        stillness_reward = near_top * self.stillness_weight / (1.0 + velocity_sumsq/10)

        # ------------------------------------------------------------
        # Link Alignment Reward (Smooth 10-degree window)
        # ------------------------------------------------------------
        # Calculate how far apart the links are
        delta_12 = abs(self._wrap_angle(th1 - th2))
        delta_23 = abs(self._wrap_angle(th2 - th3))
        
        # Linear ramp: 0.0 reward at 10 degrees, max reward at 0 degrees
        # (e.g., if threshold is 0.1745 and delta is 0.087, score is 0.5)
        alignment_12 = max(0.0, (self.alignment_threshold - delta_12) / self.alignment_threshold)
        alignment_23 = max(0.0, (self.alignment_threshold - delta_23) / self.alignment_threshold)
        
        # GATE IT: Activate if bottom pendulum is > 50 degrees away from hanging down
        # math.pi is straight down. We wrap it to handle angle wrapping safely.
        angle_from_down = abs(self._wrap_angle(th1 - math.pi))
        
        # Smooth linear ramp over 10 degrees so the neural network doesn't hit a hard math wall
        activation_margin = math.radians(10.0) 
        coupling_gate = min(1.0, max(0.0, (angle_from_down - self.coupling_activation_threshold) / activation_margin))
        alignment_reward = coupling_gate * self.alignment_reward_weight * (alignment_12 + alignment_23) 
        
        # ------------------------------------------------------------
        # Main Reward Assembly
        # ------------------------------------------------------------
        reward = (
            height_bonus 
            + alignment_reward  
            + stillness_reward 
            - position_penalty 
            - effort_penalty
            - cart_velocity_penalty
            - omega_limit_penalty     
            - energy_limit_penalty    
        )

        # ------------------------------------------------------------
        # D. Strict Success Condition
        # ------------------------------------------------------------
        is_balanced = (
            abs(self._wrap_angle(th1)) < self.success_angle_tol and
            abs(self._wrap_angle(th2)) < self.success_angle_tol and
            abs(self._wrap_angle(th3)) < self.success_angle_tol and
            abs(th1d) < self.success_omega_tol and
            abs(th2d) < self.success_omega_tol and
            abs(th3d) < self.success_omega_tol and
            abs(x) < self.success_x_tol and
            abs(xd) < self.success_xd_tol
        )

        if is_balanced:
            self.balance_counter += 1
            if self.balance_counter >= self.success_steps_req:
                reward += self.success_bonus

        else:
            self.balance_counter = 0

        if off_track:
            reward -= self.off_track_penalty
            
        # ------------------------------------------------------------
        # Info Dictionary
        # ------------------------------------------------------------
        info = {
            "th1": float(th1), "th2": float(th2), "th3": float(th3),
            "th1_dot": float(th1d), "th2_dot": float(th2d), "th3_dot": float(th3d),
            "cart_position": float(x), "cart_velocity": float(xd),
            "force": float(F),
            "kinetic_energy": float(kinetic_energy),
            "potential_energy": float(potential_energy),
            "total_energy": float(total_energy),
            "pendulum_energy": float(pendulum_energy),
            "height_bonus": float(height_bonus),
            "alignment_reward": float(alignment_reward), 
            "stillness_reward": float(stillness_reward),
            "position_penalty": float(position_penalty),
            "effort_penalty": float(effort_penalty),
            "cart_velocity_penalty": float(cart_velocity_penalty),
            "omega_limit_penalty": float(omega_limit_penalty),
            "energy_limit_penalty": float(energy_limit_penalty),

        }

        return self._get_obs(), float(reward), terminated, truncated, info  
    def render(self):
        pass
import numpy as np
from gymnasium import spaces

class PIController:
    def __init__(self, kp_range, ki_range, biases, dt, rng, action_space: spaces.Box):
        self.Kp = np.array([rng.uniform(*r) for r in kp_range])
        self.Ki = np.array([rng.uniform(*r) for r in ki_range])
        self.biases = np.array(biases)
        self.integral_errors = np.zeros_like(self.biases)
        self.dt = dt
        self.action_space = action_space

    def compute(self, setpoints, measurements):
        errors = setpoints - measurements
        raw_actions = self.biases + self.Kp * errors + self.Ki * self.integral_errors
        clipped_actions = np.clip(raw_actions, self.action_space.low, self.action_space.high)
        anti_windup_mask = np.isclose(raw_actions, clipped_actions, atol=1e-8)
        self.integral_errors += errors * self.dt * anti_windup_mask
        return clipped_actions, errors

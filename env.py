import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp
import tqdm


class PolyCSTR(gym.Env):
    def __init__(self):
        super().__init__()

        # Physical constants and parameters
        self.gas_constant = 8.314              # J/mol/K
        self.density = 879                     # kg/m3
        self.heat_capacity = 2010              # J/kg/K
        self.heat_transfer = 6000              # J/s/K

        # Reactor configuration
        self.volume = 50.0                     # L
        self.flow_rate = 0.05                  # L/s
        self.dilution_rate = self.flow_rate / self.volume  # 1/s
        self.time_step = 10.0                  # s

        # Feed bounds
        self.initiator_min = 0.0
        self.initiator_max = 0.75
        self.coolant_temp_min = 280.0          # K
        self.coolant_temp_max = 400.0          # K

        # Action space: [initiator_feed, coolant_temperature]
        self.action_space = spaces.Box(
            low=np.array([self.initiator_min, self.coolant_temp_min], dtype=np.float32),
            high=np.array([self.initiator_max, self.coolant_temp_max], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: [A, B, I, R, P, T]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 250], dtype=np.float32),
            high=np.array([np.inf, np.inf, 0.75, np.inf, np.inf, 500], dtype=np.float32),
            dtype=np.float32
        )

        # Initial state placeholders
        self.state = None
        self.current_time = 0.0

        # Feed concentrations (initialized in reset)
        self.feed_A = None
        self.feed_B = None

        # PI control parameters
        self.setpoint_polymer = 0.6
        self.setpoint_temperature = 350.0

        self.kp_polymer = 0.1 * np.random.uniform(0.2, 5)
        self.ki_polymer = 0.0005 * np.random.uniform(0.2, 5)
        self.kp_temperature = 0.2 * np.random.uniform(0.2, 5)
        self.ki_temperature = 0.002 * np.random.uniform(0.2, 5)

        # Error integrals
        self.integral_error_polymer = 0.0
        self.integral_error_temperature = 0.0

        # Control limits
        self.initiator_min, self.initiator_max = 0.0, 0.75
        self.coolant_min, self.coolant_max = 280.0, 400.0
    
    def generate_rollouts_with_pi_control(self, n_episodes=1):
        rollout_records = []

        for ep in tqdm.trange(n_episodes, desc="Generating rollouts"):
            obs, _ = self.reset()
            done = False
            t = 0.0

            while not done:
                obs, _, done, _, info = self.step_with_pi_control()

                if t % 120 == 0:  # Record every 2 minutes
                    rollout_records.append({
                        "episode": ep,
                        "time": t / 60,  # convert to minutes
                        "polymer": obs[4],
                        "temperature": obs[5],
                        "initiator_action": info["action"][0],
                        "coolant_action": info["action"][1],
                        "error_polymer": info["error_polymer"],
                        "error_temperature": info["error_temperature"],
                        "integral_error_polymer": info["integral_error_polymer"],
                        "integral_error_temperature": info["integral_error_temperature"]
                    })

                t += self.time_step

        return pd.DataFrame(rollout_records)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.feed_A = 2.0 * self.np_random.uniform(0.75, 1.25)
        self.feed_B = 1.0 * self.np_random.uniform(0.75, 1.25)
        initial_polymer = self.np_random.uniform(0.0, 0.4)

        self.state = np.array([
            self.feed_A,       # Monomer A
            self.feed_B,       # Monomer B
            0.3,               # Initiator
            0.0,               # Radicals
            initial_polymer,   # Polymer
            350.0              # Temperature (K)
        ])

        self.current_time = 0.0
        
        self.integral_error_polymer = 0.0
        self.integral_error_temperature = 0.0
        self.Kp_P = 0.1   * self.np_random.uniform(0.2, 5)
        self.Ki_P = 0.0005 * self.np_random.uniform(0.2, 5)
        self.Kp_T = 0.2   * self.np_random.uniform(0.2, 5)
        self.Ki_T = 0.002 * self.np_random.uniform(0.2, 5)
        self.initiator_bias = 0.30
        self.coolant_bias = 350.0

        return self.state.astype(np.float32), {}

    def step(self, action):
        initiator_feed = np.clip(action[0], self.initiator_min, self.initiator_max)
        coolant_temperature = np.clip(action[1], self.coolant_temp_min, self.coolant_temp_max)

        sol = solve_ivp(
            lambda t, y: self._reactor_ode(t, y, initiator_feed, coolant_temperature),
            [0, self.time_step],
            self.state,
            method="BDF",
            atol=1e-8,
            rtol=1e-6
        )

        self.state = sol.y[:, -1]
        self.current_time += self.time_step
        terminated = self.current_time >= 60 * 60 * 5  # 5 hours

        info = self._get_info()
        info["action"] = action

        obs = self.state.astype(np.float32)
        reward = 0.0
        truncated = False

        return obs, reward, terminated, truncated, info

    def _get_info(self):
        return {
            "error_polymer": self.setpoint_polymer - self.state[4],
            "error_temperature": self.setpoint_temperature - self.state[5],
            "integral_error_polymer": self.integral_error_polymer,
            "integral_error_temperature": self.integral_error_temperature
        }

    def step_with_pi_control(self, setpoints={"polymer": 0.6, "temperature": 350.0}):
        polymer_target = setpoints["polymer"]
        temperature_target = setpoints["temperature"]

        polymer_measured = self.state[4]
        temperature_measured = self.state[5]

        # Compute errors
        error_polymer = polymer_target - polymer_measured
        error_temperature = temperature_target - temperature_measured

        # Compute raw control actions
        initiator_raw = self.initiator_bias + self.Kp_P * error_polymer + self.Ki_P * self.integral_error_polymer
        coolant_raw = self.coolant_bias + self.Kp_T * error_temperature + self.Ki_T * self.integral_error_temperature

        # Apply clipping
        initiator_clipped = np.clip(initiator_raw, self.initiator_min, self.initiator_max)
        coolant_clipped = np.clip(coolant_raw, self.coolant_min, self.coolant_max)

        # Anti-windup: only integrate if not clipped
        if initiator_clipped == initiator_raw:
            self.integral_error_polymer += error_polymer * self.time_step
        if coolant_clipped == coolant_raw:
            self.integral_error_temperature += error_temperature * self.time_step

        # Step using the clipped control actions
        return self.step([initiator_clipped, coolant_clipped])

    def _rate_constant(self, A, E, T):
        return A * np.exp(-E / (self.gas_constant * T))

    def _reactor_ode(self, t, state, initiator_feed, coolant_temperature):
        A, B, I, R, P, T = state

        # Reaction rate constants
        k_decomp = self._rate_constant(4.5e15, 1.25e5, T)
        k_prop_A = self._rate_constant(3.207e7, 2.42e4, T)
        k_prop_B = self._rate_constant(1.233e8, 2.42e4, T)
        k_term   = self._rate_constant(2.103e7, 8.6e4, T)

        # Reaction rates
        rate_decomp = k_decomp * I
        rate_prop_A = k_prop_A * R * A
        rate_prop_B = k_prop_B * R * B
        rate_term = k_term * R**2

        # Material balances
        dA = self.dilution_rate * (self.feed_A - A) - rate_prop_A
        dB = self.dilution_rate * (self.feed_B - B) - rate_prop_B
        dI = self.dilution_rate * (initiator_feed - I) - rate_decomp
        dR = 2 * rate_decomp - rate_prop_A - rate_prop_B - 2 * rate_term
        dP = rate_prop_A + rate_prop_B + rate_term - self.dilution_rate * P

        # Energy balance
        heat_rxn_A = -540000.0  # J/mol
        heat_rxn_B = -860000.0  # J/mol

        total_heat_rxn = (heat_rxn_A * rate_prop_A + heat_rxn_B * rate_prop_B) * self.volume
        heat_removed = self.heat_transfer * (T - coolant_temperature)
        heat_feed = self.dilution_rate * self.heat_capacity * self.density * (coolant_temperature - T) * self.volume / 60
        total_capacity = self.density * self.heat_capacity * self.volume / 1000  # J/K

        dT = (total_heat_rxn - heat_removed + heat_feed) / total_capacity

        return [dA, dB, dI, dR, dP, dT]

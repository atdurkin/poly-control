import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp
import tqdm

from controller import PIController


class PolyCSTR(gym.Env):
    def __init__(self, time_step=1800.0):
        super().__init__()
 
        # Physical constants and parameters
        self.gas_constant = 8.314              # J/mol/K
        self.density = 1_000                   # kg/m3
        self.heat_capacity = 2_000             # J/kg/K
        self.heat_transfer = 500               # J/s/K
 
        # Reactor configuration
        self.volume = 1.0                      # m3 (1000 liters)
        self.solvent = 80                      # kg/h (assumed constant)
        self.time_step = time_step             # s
        self.max_sim_time = 60 * 60 * 100      # 100 hours in seconds
        self.heat_rxn = 100e3                  # J/mol
        self.feed_T = 350.0                    # K
        self.feed_A = 100                      # kg/h
 
        # Action space: [initiator_feed [kg/h], coolant_temperature [K]]
        self.action_space = spaces.Box(
            low=np.array([0.0, 300.0], dtype=np.float32),
            high=np.array([2.5, 350.0], dtype=np.float32),
            dtype=np.float32
        )
 
        # Observation space: [A [kg/m3], I [kg/m3], R [mol/m3], P [kg/m3], T [K]]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 250], dtype=np.float32),
            high=np.array([1e3, 1e3, 1e4, 1e3, 500], dtype=np.float32),
            dtype=np.float32
        )
 
        # noisy observations
        self.noise_std = np.array([0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
 
        # set setpoints
        self.setpoints = np.array([100.0, 350.0])  # [polymer, temperature]
 
        self.reset()
   
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
 
        initial_A = 440 * self.np_random.uniform(0.75, 1.25)  # kg/m3
        initial_polymer = 0 * self.setpoints[0] * self.np_random.uniform(0.0, 1.0)  # kg/m3
        initial_temperature = self.setpoints[1] + self.np_random.uniform(-5.0, 5.0)  # K

        self.current_time = 0.0
 
        self.controller = PIController(
            kp_range=[(2e-4, 5e-3), (8e-3, 0.2)],
            ki_range=[(6e-8, 1.5e-6), (2e-6, 5e-5)],
            biases=[1.0, 345.0],
            dt=self.time_step,
            rng=self.np_random,
            action_space=self.action_space
        )

        self.state = np.array(
            [
                initial_A,                              # Monomer A
                0.0,                                    # Initiator
                0.0,                                    # Radicals
                initial_polymer,                        # Polymer
                initial_temperature,                    # Temperature (K)
                self.controller.biases[0],              # Initiator feed (kg/h)
                self.controller.biases[1]               # Coolant temperature (K)
            ]
        )
 
        info = self._get_info()
        obs = self._get_obs()
 
        return obs, info
 
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        sol = solve_ivp(
            lambda t, y: self._reactor_ode(t, y, action[0], action[1]),
            [0, self.time_step],
            self.state[:-2],
            method="BDF",
            atol=1e-8,
            rtol=1e-6
        )
        self.state[:-2] = sol.y[:, -1]      # Update reactor state variables
        self.state[-2] = action[0]          # Update initiator feed
        self.state[-1] = action[1]          # Update coolant temperature
 
        info = self._get_info()
        info["action"] = action
        obs = self._get_obs()
        reward = - (info["error_polymer"] ** 2 + info["error_temperature"] ** 2)
        terminated = False
        self.current_time += self.time_step
        truncated = self.current_time >= self.max_sim_time
        return obs, reward, terminated, truncated, info

    def step_with_pi_control(self):
        measurements = self.state[3:5]
        actions, errors = self.controller.compute(self.setpoints, measurements)
        return self.step(actions)

    def generate_rollouts_with_pi_control(self, n_episodes=1, setpoints: dict = {"polymer": 0.6, "temperature": 350.0}):
        rollout_records = []
 
        for ep in tqdm.trange(n_episodes, desc="Generating rollouts"):
            obs, info = self.reset()
            done = False
            t = 0.0
 
            while not done:
                state = self.state
                rollout_records.append({
                    "episode": ep,
                    "time": t / 60,  # convert to minutes
                    "A [kg/m3]": state[0],
                    "initiator [kg/m3]": state[1],
                    "radicals [mol/m3]": state[2],
                    "polymer [kg/m3]": state[3],
                    "temperature [K]": state[4],
                    "initiator feed [kg/h]": state[5],
                    "coolant temp [K]": state[6],
                    "error_polymer [kg/m3]": info["error_polymer"],
                    "error_temperature [K]": info["error_temperature"],
                    "integral_error_polymer [kg s/m3]": info["integral_error_polymer"],
                    "integral_error_temperature [K s]": info["integral_error_temperature"]
                })
 
                _, _, terminated, truncated, info = self.step_with_pi_control()
                done = terminated or truncated
                t += self.time_step
 
        return pd.DataFrame(rollout_records)
 
    def _get_info(self):
        return {
            "error_polymer": self.setpoints[0] - self.state[3],
            "error_temperature": self.setpoints[1] - self.state[4],
            "integral_error_polymer": self.controller.integral_errors[0],
            "integral_error_temperature": self.controller.integral_errors[1],
        }
 
    def _get_obs(self):
        obs = np.array(
            [
                self.state[0],          # A [kg/m3]
                self.state[3],          # polymer [kg/m3]
                self.state[4],          # temperature [K]
                self.setpoints[0] - self.state[3],  # error_polymer [kg/m3]
                self.setpoints[1] - self.state[4],  # error_temperature [K]
                self.state[5],          # initiator feed [kg/h]
                self.state[6],          # coolant temperature [K]
            ]
        )
        noise = self.np_random.normal(0, self.noise_std, size=self.state.shape)
        obs += noise
        return obs.astype(np.float32)
 
    def _rate_constant(self, A, E, T):
        return A * np.exp(-E / (self.gas_constant * T))
 
    def _reactor_ode(self, t, state, initiator_feed, coolant_temperature):
        A, I, R, P, T = state
 
        # Mole weights
        MW_A = 100 / 1e3                # kg/mol
        MW_I = 165 / 1e3                # kg/mol
 
        # Activation energies for each reaction
        E_decomp = 125 * 1e3            # J/mol
        E_prop_A = 25 * 1e3             # J/mol
        E_term = 15 * 1e3               # J/mol
 
        # Pre-exponential factors
        A_decomp = 1e9                  # 1/s
        A_prop_A = 4e4                  # m3/mol s
        A_term = 1e6                    # m3/mol s
 
        # Reaction rate factors
        k_decomp = self._rate_constant(A_decomp, E_decomp, T)   # [1/s]
        k_prop_A = self._rate_constant(A_prop_A, E_prop_A, T)   # [m3/mol s]
        k_term = self._rate_constant(A_term, E_term, T)         # [m3/mol s]
 
        # Reaction rates [mol/m3 s]
        rate_decomp = k_decomp * I / MW_I           # [mol/m3 s] = [1/s] [kg/m3] [mol/kg]
        rate_prop_A = k_prop_A * R * A / MW_A       # [mol/m3 s] = [m3/mol s] [mol/m3] [kg/m3] [mol/kg]
        rate_term = k_term * R**2                   # [mol/m3 s] = [m3/mol s] [mol/m3]^2
 
        # Material balances
        dA = self.feed_A / 3600 / self.volume - self._dilution_rate * A - rate_prop_A * MW_A           # [kg/m3 s] = [kg/h] [h/s] [1/m3] - [1/s] [kg/m3] - [mol/m3 s] [kg/mol]
        dI = initiator_feed / 3600 / self.volume - self._dilution_rate * I - rate_decomp * MW_I         # [kg/m3 s] = [kg/h] [h/s] [1/m3] - [1/s] [kg/m3] - [mol/m3 s] [kg/mol]
        dR = 2 * rate_decomp - 2 * rate_term                                                          # [mol/m3 s] = [mol/m3 s] - [mol/m3 s]
        dP = rate_prop_A * MW_A - self._dilution_rate * P                                              # [kg/m3 s] = [mol/m3 s] [kg/mol] - [1/s] [kg/m3]
 
        # Energy balance
        heat_rxn = self.heat_rxn * rate_prop_A * self.volume                                            # [J/s] = [J/mol] [mol/m3 s] [m3]
        heat_removed = self.heat_transfer * (T - coolant_temperature)                                   # [J/s] = [J/s K] [K]
        heat_flowed = self._flow_rate * self.heat_capacity * self.density * (self.feed_T - T)            # [J/s] = [m3/s] [J/kg K] [kg/m3] [K]
        inertia = self.density * self.heat_capacity * self.volume                                       # [J/K] = [kg/m3] [J/kg K] [m3]
        dT = (heat_rxn - heat_removed + heat_flowed) / inertia                                          # [K/s] = [J/s] / [J/K]
 
        return [dA, dI, dR, dP, dT]
 
    @property
    def _flow_rate(self):
        total_mass_flow = self.state[5] + self.feed_A + self.solvent  # kg/h
        total_volume_flow_rate = total_mass_flow / self.density / 3600  # m3/s (assume constant density)
        return total_volume_flow_rate
 
    @property
    def _residence_time(self):
        return self.volume / self._flow_rate
 
    @property
    def _dilution_rate(self):
        return self._flow_rate / self.volume
 
 
if __name__ == "__main__":
 
    from utils import plot_rollouts, plot_reward_distributions
 
    env = PolyCSTR()
    df = env.generate_rollouts_with_pi_control(n_episodes=100)
    df.to_csv("rollouts.csv", index=False)
    plot_rollouts(df, env.setpoints)
 
    df["reward"] = -(df["error_polymer [kg/m3]"] ** 2 + df["error_temperature [K]"] ** 2)
    rewards = df.groupby("episode")["reward"].sum()
    rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min()) * 100
    plot_reward_distributions({'Data': rewards})

from env import PolyCSTR
from utils import plot_rollouts


env = PolyCSTR()
df = env.generate_rollouts_with_pi_control(n_episodes=120)
df.to_csv("rollouts.csv", index=False)
plot_rollouts(df)

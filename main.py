from env import PolyCSTR
from utils import plot_rollouts


env = PolyCSTR()
df = env.generate_rollouts_with_pi_control(n_episodes=20)
plot_rollouts(df)

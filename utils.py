import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class StandardScaler:
    def __init__(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-8  # prevent division by zero

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class MinMaxScaler:
    def __init__(self, data, feature_range=(-1.0, 1.0)):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)
        self.scale = feature_range[1] - feature_range[0]
        self.min_range = feature_range[0]

    def transform(self, data):
        return self.min_range + self.scale * (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return self.min + (data - self.min_range) * (self.max - self.min) / (self.scale + 1e-8)


def plot_rollouts(rollout_df, setpoints):
 
    # Create a figure with a custom layout
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 6)  # Define a 3x6 grid for flexibility
 
    # Top row (2 subplots, centered)
    ax1 = fig.add_subplot(gs[0, 1:3])  # First subplot spans columns 1-2
    ax2 = fig.add_subplot(gs[0, 3:5])  # Second subplot spans columns 3-4
 
    # Middle row (3 subplots)
    ax3 = fig.add_subplot(gs[1, :2])  # First subplot spans columns 0-1
    ax4 = fig.add_subplot(gs[1, 2:4])  # Second subplot spans columns 2-3
    ax5 = fig.add_subplot(gs[1, 4:])  # Third subplot spans columns 4-5
 
    # Bottom row (2 subplots, centered)
    ax6 = fig.add_subplot(gs[2, 1:3])  # First subplot spans columns 1-2
    ax7 = fig.add_subplot(gs[2, 3:5])  # Second subplot spans columns 3-4
 
    for ep_id, ep_data in rollout_df.groupby("episode"):
        ax1.plot(ep_data["time"]/60, ep_data["polymer [kg/m3]"], alpha=0.7)
        ax2.plot(ep_data["time"]/60, ep_data["temperature [K]"], alpha=0.7)
 
        ax3.plot(ep_data["time"]/60, ep_data["A [kg/m3]"], alpha=0.7)
        ax4.plot(ep_data["time"]/60, ep_data["initiator [kg/m3]"], alpha=0.7)
        ax5.plot(ep_data["time"]/60, ep_data["radicals [mol/m3]"], alpha=0.7)
       
        ax6.plot(ep_data["time"]/60, ep_data["initiator feed [kg/h]"], alpha=0.7)
        ax7.plot(ep_data["time"]/60, ep_data["coolant temp [K]"], alpha=0.7)
 
    ax1.axhline(y=setpoints[0], color='k', linestyle='--')
    ax2.axhline(y=setpoints[1], color='k', linestyle='--')
 
    ax1.set_title("Polymer Concentration")
    ax1.set_ylabel("kg/m3")
    ax1.grid()
 
    ax2.set_title("Reactor Temperature")
    ax2.set_ylabel("K")
    ax2.grid()
 
    ax3.set_title("Monomer A Concentration")
    ax3.set_ylabel("kg/m3")
    ax3.grid()
 
    ax4.set_title("Initiator Concentration")
    ax4.set_ylabel("kg/m3")
    ax4.grid()
 
    ax5.set_title("Radicals Concentration")
    ax5.set_ylabel("mol/m3")
    ax5.grid()
 
    ax6.set_title("Initiator Feed")
    ax6.set_ylabel("kg/h")
    ax6.grid()
 
    ax7.set_title("Coolant Temperature")
    ax7.set_ylabel("K")
    ax7.grid()
 
    ax1.set_xlabel("time [h]")
    ax2.set_xlabel("time [h]")
    ax3.set_xlabel("time [h]")
    ax4.set_xlabel("time [h]")
    ax5.set_xlabel("time [h]")
    ax6.set_xlabel("time [h]")
    ax7.set_xlabel("time [h]")
 
    plt.tight_layout()
    plt.show()


def plot_reward_distributions(data):
    # data is a dictionary with keys of xtick labels and values as lists of rewards

    keys = list(data.keys())
    rewards = list(data.values())

    # boxplot of the total rewards
    plt.figure(figsize=(10, 6))
    plt.boxplot(rewards,
                vert=True,
                patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'),
                showfliers=False,)
    plt.xticks(range(1, len(keys) + 1), keys)
    plt.ylabel('Total episode rewards')
    plt.grid()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt


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


def plot_rollouts(rollout_df):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for ep_id, ep_data in rollout_df.groupby("episode"):
        axs[0, 0].plot(ep_data["time"], ep_data["polymer"], alpha=0.7)
        axs[0, 1].plot(ep_data["time"], ep_data["temperature"], alpha=0.7)
        axs[1, 0].plot(ep_data["time"], ep_data["initiator_action"], alpha=0.7)
        axs[1, 1].plot(ep_data["time"], ep_data["coolant_action"], alpha=0.7)

    axs[0, 0].set_title("Polymer Concentration")
    axs[0, 0].set_ylabel("mol/L")
    axs[0, 0].grid()

    axs[0, 1].set_title("Reactor Temperature")
    axs[0, 1].set_ylabel("K")
    axs[0, 1].grid()

    axs[1, 0].set_title("Initiator Feed")
    axs[1, 0].set_ylabel("mol/L")
    axs[1, 0].set_xlabel("Time (min)")
    axs[1, 0].grid()

    axs[1, 1].set_title("Coolant Temperature")
    axs[1, 1].set_ylabel("K")
    axs[1, 1].set_xlabel("Time (min)")
    axs[1, 1].grid()

    plt.tight_layout()
    plt.show()

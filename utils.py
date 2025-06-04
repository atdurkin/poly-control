import matplotlib.pyplot as plt

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

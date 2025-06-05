import pandas as pd
import torch

from utils import StandardScaler, MinMaxScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_labels = [
    "polymer",
    "temperature",
    "error_polymer",
    "error_temperature",
    "initiator_action",  # absolute actions in the states
    "coolant_action",
]
action_labels = [
    "initiator_action",
    "coolant_action"
]


class PolyCSTRDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path="rollouts.csv", 
        state_labels=state_labels, 
        action_labels=action_labels
    ):
        self.data = pd.read_csv(data_path)
        self.state_labels = state_labels
        self.action_labels = action_labels
        self.state_dim = len(state_labels)
        self.action_dim = len(action_labels)

        self.state_scaler = StandardScaler(self.data[state_labels].values)
        self.states = torch.from_numpy(self.state_scaler.transform(self.data[state_labels].values)).float().to(device)
        
        self._get_dones()
        self._get_next_states()
        next_actions = self._get_next_actions()

        actions = next_actions - self.data[action_labels]
        self.action_scaler = MinMaxScaler(actions.values)
        self.actions = torch.from_numpy(self.action_scaler.transform(actions.values)).float().to(device)

        self._get_rewards()

        self.current_idx = 0
        self.len = self.__len__()
    
    def sample(self, batch_size, shuffle=True):
        if self.current_idx == 0 and shuffle:
            self.perm_indices = torch.randperm(self.len)
        if self.current_idx + batch_size > self.len:
            self.current_idx = 0
            if shuffle:
                self.perm_indices = torch.randperm(self.len)
        indices = self.perm_indices[self.current_idx:self.current_idx + batch_size]
        data = {
            "s": self.states[indices],
            "a": self.actions[indices],
            "s_": self.next_states[indices],
            "r": self.rewards[indices],
            "done": self.done[indices]
        }
        self.current_idx += batch_size
        return data

    def _get_rewards(self):
        self.data["reward"] = - self.data["error_polymer"] ** 2 + \
            - self.data["error_temperature"] ** 2
        self.rewards = torch.from_numpy(self.data["reward"].values.reshape(-1, 1)).float().to(device)    
 
    def _get_dones(self):
        self.data["done"] = 0
        last_of_episode = self.data["episode"].diff(-1) != 0
        self.data.loc[last_of_episode, "done"] = 1
        self.done = torch.from_numpy(self.data["done"].values.reshape(-1, 1)).to(device)
    
    def _get_next_states(self):
        next_states = self.data[self.state_labels].shift(-1)
        dones = self.data["done"] == 1
        next_states.loc[dones, :] = self.data.loc[dones, self.state_labels].copy()
        self.next_states = torch.from_numpy(next_states.values).float().to(device)

    def _get_next_actions(self):
        next_actions = self.data[self.action_labels].shift(-1)
        dones = self.data["done"] == 1
        next_actions.loc[dones, :] = self.data.loc[dones, self.action_labels].copy()
        return next_actions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        use_idx = idx % self.len
        row = self.data.iloc[use_idx]
        state = torch.tensor(row[self.state_labels].values, dtype=torch.float32).to(device)
        action = torch.tensor(row[self.action_labels].values, dtype=torch.float32).to(device)
        return state, action

if __name__ == "__main__":

    dataset = PolyCSTRDataset()
    data = dataset.sample(batch_size=256)

    print(dataset.len)
    
    for key, val in data.items():
        print(key, type(val), val.shape)

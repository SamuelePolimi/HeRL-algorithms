import numpy as np
import torch
from typing import Callable, Union

from herl.rl_interface import Offline, Critic, PolicyGradient, RLTaskDescriptor, Actor, RLAgent, RLParametricModel
from herl.dataset import Dataset
from herl.actor import NeuralNetwork, NeuralNetworkPolicy


class NeuralActor(Offline, Critic, Actor):

    def __init__(self, task_descriptor: RLTaskDescriptor,
                 dataset: Dataset,
                 policy: Union[RLAgent, RLParametricModel],
                 neural_network_constructor: Callable[[], NeuralNetwork], minibatch_size=500):
        name = "nfqi"
        Offline.__init__(self, name, task_descriptor, dataset)
        Actor.__init__(self, name, policy)
        self._minibatch_size = minibatch_size
        self._q_network = neural_network_constructor()
        self._q_target = neural_network_constructor()
        self._synchronize()

        self.optimizer = torch.optim.Adam(self._q_network.parameters(), lr=1E-4)

    def _synchronize(self):
        self._q_target.set_parameters(self._q_network.get_parameters())

    def get_loss(self, s, a, r, s_n, t):
        target = torch.from_numpy(r + self.task_descriptor.gamma * (1-t) * self._q_target(s_n, self.policy(s_n)))
        return torch.mean(
            (target - self._q_network(torch.from_numpy(s), torch.from_numpy(a), differentiable=True)).pow(2))

    def get_direct_loss(self, s, a, r, s_n, t):
        reward = torch.from_numpy(r)
        q_next = self._q_network(torch.from_numpy(s_n), torch.from_numpy(self.policy(s_n)), differentiable=True)

        return torch.mean(
            (reward + torch.from_numpy(self.task_descriptor.gamma * (1 - t)) * q_next
             - self._q_network(torch.from_numpy(s), torch.from_numpy(a), differentiable=True)).pow(2))

    def update(self, n_step=2000, n_iterations=50):
        progress = self.get_progress_bar("learn", n_step*n_iterations)
        for _ in range(n_iterations):
            for _ in range(n_step):
                progress.notify()
                data = self.dataset.get_minibatch(self._minibatch_size)
                loss = self.get_loss(data["state"], data["action"],
                                            data["reward"], data["next_state"], data["terminal"])
                self.optimizer.zero_grad()
                loss.backward()
                # print(loss)
                self.optimizer.step()
            self._synchronize()

    def get_Q(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self._q_network(state, action)

    def get_V(self, state: np.ndarray) -> np.ndarray:
        return self._q_network(state, self.policy(state))

    def get_return(self) -> np.ndarray:
        state = self.task_descriptor.initial_state_distribution.sample()
        return self.get_V(np.array([state]))



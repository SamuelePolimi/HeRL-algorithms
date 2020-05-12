import numpy as np
import torch
import matplotlib.pyplot as plt

from herl.classic_envs import Pendulum2D
from herl.rl_interface import DeterministicState, RLTask, StochasticState
from herl.actor import NeuralNetworkPolicy, NeuralNetwork, UniformPolicy
from herl.solver import RLCollector
from herl.rl_visualizer import ValueFunctionVisualizer, StateCloudVisualizer, ValueRowVisualizer
from herl.rl_analysis import MCAnalyzer

from src.algorithms.dpg import NeuralActor


def q_network_constructor():
    return NeuralNetwork([env.state_dim, env.action_dim], [500, 1], [torch.relu])


stochastic_state = StochasticState(lambda: np.random.uniform(np.array([-np.pi, -2.]), np.array([np.pi, 2.])))
env = Pendulum2D()
task = RLTask(env, DeterministicState(np.array([np.pi, 0.])), gamma=0.95, max_episode_length=200)
collection_task = RLTask(env, stochastic_state, max_episode_length=200)

dataset = task.get_empty_dataset(n_max_row=10000).train_ds
uniform_policy = UniformPolicy(np.array([-2.]), np.array([2.]))
collector = RLCollector(dataset, collection_task, uniform_policy)
collector.collect_rollouts(500)

policy = NeuralNetworkPolicy(env.get_descriptor(), [50], [torch.relu], lambda x: 2.*torch.tanh(x))
nfqi = NeuralActor(task.get_descriptor(), dataset, policy, q_network_constructor)
nfqi.unmute()

nfqi.update(n_iterations=50)

mc_analyzer = MCAnalyzer(task, policy)
value_row = ValueRowVisualizer()
value_row.compute(env.get_descriptor(), [mc_analyzer, nfqi], [[50, 50], [200, 200]])

fig, axs = plt.subplots(1, 2)

value_row.visualize(axs)
value_row.visualize_decorations(axs)
plt.show()


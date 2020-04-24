import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from itertools import product, chain
import torch
from agents.nnfactory.sac import Policy
import numpy as np


def plot_nn(input, output, activations):
    input = np.ravel(input)
    output = np.ravel(output)

    G = nx.DiGraph()

    neurons = []
    colors = []

    for layer_id, layer_activation in enumerate(activations):
        layer_neurons = []
        layer_colors = []
        for neuron_id, activation in enumerate(layer_activation):
            node_id = f"{layer_id} {neuron_id}"
            layer_neurons.append(node_id)
            layer_colors.append(activation)

        neurons.append(layer_neurons)
        colors.append(layer_colors)

    for layerA, layerB in zip(neurons[:-1], neurons[1:]):
        for neuronA, neuronB in product(layerA, layerB):
            G.add_edge(neuronA, neuronB)

    input_neurons = [f"I {node_id}" for node_id in range(len(input))]
    output_neurons = [f"O {node_id}" for node_id in range(len(output))]

    # add input layer
    for input_id, neuron_id in product(input_neurons, neurons[0]):
        G.add_edge(input_id, neuron_id)

    # add output layer
    for neuron_id, output_id in product(neurons[-1], output_neurons):
        G.add_edge(neuron_id, output_id)

    # input_colors = list(chain.from_iterable(state.tolist()))
    # output_colors = action
    nodelist = list(chain.from_iterable(neurons))
    colors = list(chain.from_iterable(colors))

    fig = plt.figure(figsize=(22, 12))
    # fig = plt.figure()
    pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
    node_distance = pos["0 0"][1] - pos["0 1"][1]

    nx.draw_networkx_edges(G, pos=pos, node_size=int(node_distance * 0.95),
                           alpha=0.2)
    # draw input
    nx.draw_networkx_nodes(G, nodelist=input_neurons,
                           node_color=input,
                           # node_size=int(node_distance * 0.95),
                           pos=pos,
                           cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
    # draw output
    nx.draw_networkx_nodes(G, nodelist=output_neurons,
                           node_color=output,
                           # node_size=int(node_distance * 0.95),
                           pos=pos, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)

    # draw activations
    nc = nx.draw_networkx_nodes(G, nodelist=nodelist, node_color=colors,
                                # node_size=int(node_distance * 0.95),
                                pos=pos, cmap=plt.cm.Reds)
    plt.colorbar(nc)

    return fig


if __name__ == "__main__":

    policy_file = "../results/20200421-104015/models/32500824_1.000/policy.pt"

    policy = Policy((21,), (7,), 32, 8)
    weights = policy.parameters()
    policy.load_state_dict(torch.load(policy_file))

    state = torch.zeros((1, 21))

    for _ in range(1):
        action, std = policy(state)
        activations = policy.get_activations(state)

        activations = np.array(
            [a.view(-1).detach().numpy() for a in activations])
        print(activations.shape)
        state = state.detach().numpy()[0]
        action = action.detach().numpy()

        plot_nn(state, action, activations)
        state = torch.rand((1, 21)) * 2 - 1

    plt.show()

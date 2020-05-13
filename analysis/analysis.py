import torch
import h5py
from multiprocessing import cpu_count

""" 
BIG ASK: How do we find similar structures between two neural networks (each of which approximating a policy) 

NOW:
1. generate experiences (action and/or state) for two nets 
2. compute similarity matrices between those experiences. Use Cos/L2 any other metric with weighted option
3. Filter the most similar experiences and use those experiences to do operations in the structure of the two nets
4. These operations can output just the path of the highest activation or some graph structure representing most 
important attributions to the resulting action.

IDEA:
von Neumann Entropy ? 
Shannon Entropy ?
both entropy measures to quantify information content encoded in our networks


Isomorphism (identify bijection between two graphs which preserves adjacency)                       - https://dl.acm.org/doi/pdf/10.1145/321556.321562 https://arxiv.org/abs/1802.08509
minimum cost transformations (with the cost referring to the operations to equal the two graphs)    - https://sci-hub.tw/10.1109/34.682179
maximum common subgraph (find largest isomorphic subgraphs between to the two graphs)               - https://sci-hub.tw/10.1016/s0167-8655(01)00017-4
statistical similariities                                                                           - https://sci-hub.tw/10.1103/revmodphys.74.47 https://sci-hub.tw/10.1086/210318 
iterative methods?                                 

"""



if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    from experience_generator import generate_exerience
    from experience_similarity import compute_weighted_similarity_matrix
    from agents import get_agent
    from environments.orchestrator import Orchestrator
    import os.path as osp
    import json

    agent_modelA = "20200421-104015/models/32500824_1.000"

    with open("../results/20200421-104015/config.json") as f:
        configA = json.load(f)

    agent_configA = configA["agent_config"]

    agent_modelB = "20200421-104015/models/32500824_1.000"

    with open("../results/20200421-104015/config.json") as f:
        configB = json.load(f)

    agent_configB = configB["agent_config"]
    #
    # agent_configB = {
    #     "algorithm": "sac", "soft_q_lr": 0.001,
    #     "policy_lr": 0.001, "alpha": 1,
    #     "alpha_lr": 0.0005,
    #     "weight_decay": 0.0001, "batch_size": 128,
    #     "gamma": 0.95, "auto_entropy": True,
    #     "memory_size": 100000, "tau": 0.0025,
    #     "hidden_dim": 5,
    #     "hidden_layers": 1,
    #     "seed": 192,
    #     "pretrained_models": "20200324-124838/models/15500432_0.880"
    # }

    env_config = {
        "nb_envs": 1,
        "base_pkg": "robot-task-rl",
        "render": False,
        "task_config": {"name": "reach",
                        "dof": 3,
                        "only_positive": False,
                        "sparse_reward": False,
                        "max_steps": 25
                        },
        "robot_config": {
            "name": "panda",
            "dof": 3,
            "sim_time": .1,
            "scale": .1
        }
    }

    nb_envs = env_config["nb_envs"]
    env_orchestrator = Orchestrator(env_config, nb_envs)

    agentA = get_agent(agent_configA, env_orchestrator.observation_space,
                      env_orchestrator.action_space)
    agentB = get_agent(agent_configB, env_orchestrator.observation_space,
                      env_orchestrator.action_space)

    models_folderA = osp.join(osp.dirname(__file__), "../results",
                             agent_modelA)

    models_folderB = osp.join(osp.dirname(__file__), "../results",
                             agent_modelB)

    import os
    print(os.listdir(models_folderA))

    agentA.load(models_folderA)
    agentB.load(models_folderB)

    fileA = 'experienceA.hdf5'
    fileB = 'experienceA.hdf5'

    # generate_exerience(agentA, env_orchestrator, 1, ["states", "actions"],
    #                    fileA)
    # experiences_h5_file_A = h5py.File(fileA, 'r')
    # experienceA = {k: torch.Tensor(v[:]) for k, v in
    #                experiences_h5_file_A.items()}
    #
    # print(experienceA)

    base_experience = {
        "actions": torch.Tensor([[1,0,0,0,0,0,0]])
    }

    generate_exerience(agentB, env_orchestrator, 5000, ["states", "actions"],
                       fileB)
    experiences_h5_file_B = h5py.File(fileB, 'r')

    weights = {
        'actions': {'l2': 0, 'cos': 1},
        # 'states': {'l2': 1, 'cos':0}

    }

    experienceB = {k:torch.Tensor(v[:]) for k,v in experiences_h5_file_B.items()}

    similarity_matrix = compute_weighted_similarity_matrix(base_experience, experienceB, weights)

    import numpy as np

    similarity_matrix[np.tril_indices(len(similarity_matrix))] = -np.inf

    print(similarity_matrix[similarity_matrix > 0.99])
    plt.hist(similarity_matrix[similarity_matrix > -np.inf])
    plt.show()

    exit()

    # topk returns k largest elements of input along given dimension with indices. Indices are from flattened tensor
    values, indices = similarity_matrix.view(-1).topk(100)

    #
    indices_2d = torch.empty((*indices.shape, 2))

    print(similarity_matrix)
    print(indices.shape, indices)
    print(indices_2d.shape, indices_2d)
    print(similarity_matrix.shape)


    indicesA = indices / similarity_matrix.shape[-1]
    indicesB = indices % similarity_matrix.shape[-1]

    print(indices)
    print(indicesA)
    print(indicesB)

    print(experienceA["states"])
    statesA = experienceA["states"][indicesA]
    statesB = experienceB["states"][indicesB]

    print(statesA)
    print(statesB)

    activationsA = agentA.get_activations(statesA)
    activationsB = agentB.get_activations(statesB)

    print()
    print(activationsA)

    import umap

    for layer_id, (activations_layerA, activations_layerB) in enumerate(zip(activationsA, activationsB)):
        print(layer_id)
        print(activations_layerA.shape)
        print(activations_layerB.shape)

        activations = torch.cat((activations_layerA, activations_layerB)).cpu().detach().numpy()
        print(activations.shape)

        reducer = umap.UMAP()

        embedding = reducer.fit_transform(activations)

        embeddingA = embedding[:len(activations_layerA)]
        embeddingB = embedding[-len(activations_layerB):]

        plt.figure()
        plt.scatter(embeddingA[:, 0], embeddingA[:, 1],
                    alpha=0.3)

        plt.scatter(embeddingB[:, 0], embeddingB[:, 1],
                    alpha=0.3)

        plt.legend()
        #

    plt.show()

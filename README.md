# KAROLOS
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

__KAROLOS (Open-Source Robot-Task Learning Simulation) is an open-source simulation and reinforcement learning suite.__

<p align="center">
<img src="docs/images/logo.png" width="500">
</p>

KAROLOS was developed with a focus on:

- __scalability__: As reinforcement learning algorithms require significant amounts of experience,
KAROLOS enables the parallelization of environments.
This way, you spend less time on data collection and more time on training and prototyping.

-  __modularization__: More and more research in reinforcement learning is looking into the transfer of agents from one environment to another.
KAROLOS was developed to quickly generate environments with different robot-task combinations.

## Installation

First, clone the repository

```
git clone https://github.com/tmdt-buw/karolos
cd karolos
```

Install the dependencies using [Anaconda](https://docs.anaconda.com/anaconda/install/). It might be necessary to specify the pytorch/cuda version that fits your system. Make sure that `pytorch<=1.13.1`. 

``` bash
conda env create -f environment.yml
conda activate karolos
```

## Getting Started

We recommend taking a look at the training examples, which you can run like this

``` bash
python examples/train_sac_panda_reach.py
```

Of course you will want to run your custom experiments. Running an experiment is done by specifying the training hyperparameters, initializing the experiment and then running it

``` python
from karolos.experiment import Experiment

training_config = {...}

experiment = Experiment(training_config)
    
experiment.run(results_dir="results/my_first_experiment")
```

You can monitor the progress of your experiment in real-time with a tensorboard

``` bash
tensorboard --logdir results
```

## Customizing KAROLOS to your needs

KAROLOS was developed with the goal to be modular and customizable. If you require a different environment, agent, or replay buffer, you can achieve this with the following steps:

1. Locate the component you want to add/change in the folder structure. E.g. if you want to add a new environment, go to `environment/`, if you want to add a new task for the provided `EnvironmentRobotTask`, go to `environment/robots`.
2. Add a new file which contains the new component class. This class should inherit from the base class found in the folder, i.e. `environment/environment.py`, or `environment/robots.robot.py`.
3. Register your component in the `__init__.py` found in the folder. This is necessary so your component will be found when constructing the experiment.
4. Use your component by using the `name` tags in the config.
``` json
{
"total_timesteps": 5_000_000,
"test_interval": 500_000,
"number_tests": 100,

"agent_config": {
    "name": "my_new_agent",
    "custom_agent_param": ...,

    "replay_buffer": {
      "name": "my_new_replay_buffer",
      "custom_replay_buffer_param": ...,
    }
},
"env_config": {
    "name": "my_new_env",
    "custom_env_param": ...,
```

## Contribute to KAROLOS

We welcome you to contribute to this project!

1. You can help us by giving us constructive feedback by opening a new issue if you encountered any bugs or if you have ideas for improvement.

2. We are always grateful for support with the development of KAROLOS.
If you are interested, check out the issues to see what features need to be worked on.
Once you have found an issue which you want to tackle, make sure that nobody else is working on it.
For extensive issues make sure to start a discussion to clarify the scope and design choices of your solution.
Communication is key!

## License

This project is published under the MIT license.

## Citation

Please cite **Karolos** if you use this framework in your publications:
```bibtex
@misc{karolos,
  title = {Karolos: An Open-Source Reinforcement Learning Framework for Robot-Task Environments},
  author = {Bitter, Christian and Thun, Timo and Meisen, Tobias},
  
  publisher = {arXiv},
  year = {2022},
  
  url = {https://arxiv.org/abs/2212.00906},
  doi = {10.48550/ARXIV.2212.00906},
}
```

## Trivia

The name is inspired by the karolus monogram (used in the logo), which was the signature of Charles the Great. 

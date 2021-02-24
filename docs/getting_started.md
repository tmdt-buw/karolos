# Getting Started

## First steps

Run an experiment by launching a trainer

```
python trainer/trainer.py
```

You can monitor the progress of your experiment in real-time with a tensorboard

```
tensorboard --logdir results
```

## Experiment Configuration

An experiment is parametrized by a configuration dictionary.


``` json
{
    "total_timesteps": 1_000_000,   # experiment is run until this amount of experience samples is collected
    "test_interval": 10_000,        # samples after which the agent performance is tested
    "number_tests": 100,            # amount of episodes to be run for a test
    "her_ratio": 0.,                # amount of data to be generate with hindsight experience replay relative to amount of collected samples. The generated data is added on top of the collected data.
    
    "number_processes": 1,          # amount of parallel processes in which environments are launched
    "number_threads": 1,            # amount of environments launched per process

    "agent_config": {...},          # agent configuration, see seperate subchapter
    
    "env_config": {...}             # environment configuration, see seperate subchapter
}
```

The agent configuration `agent_config` is specific to the agent you want to use. Please refer to the chapter [Agents](agents.md) for specifics.

Likewise, the environment configuration `agent_config` is specific to the environment you want to use. Please refer to the chapter [Environments](environments.md) for specifics.

This framework uses Pytorch applied to Pybullet simulation environments
composed of a robot and a task which can be arbitrarily combined.
Run the minimal configuration with an algorithm and task/robot combination of your choice.
Soft-Actor-Critic with reach task on pandas robot should converge after about 5 million steps to a successful policy.
See architecture doc for further description of the components and contributing 
if you'd like to contribute a feature, robot, task or new agent.

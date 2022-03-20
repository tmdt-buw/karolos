def get_task(task_config, bullet_client):
    """
    get the task from the implemented tasks
    :param task_config: config of task
    :param bullet_client: pybullet client
    :return: task
    """
    task_name = task_config.pop("name")

    if task_name == 'reach':
        from .task_reach import TaskReach
        task = TaskReach(bullet_client, **task_config)
    elif task_name == 'pick_place':
        from .task_pick_place import TaskPickPlace
        task = TaskPickPlace(bullet_client, **task_config)
    elif task_name == 'throw':
        from .task_throw import TaskThrow
        task = TaskThrow(bullet_client, **task_config)
    else:
        raise NotImplementedError(f"Unknown task {task_name}")

    return task

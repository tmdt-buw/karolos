def get_task(task_config, bullet_client):
    task_name = task_config.pop("name")

    if task_name == 'reach':
        from .task_reach import TaskReach
        task = TaskReach(bullet_client, **task_config)
    elif task_name == 'pick_place':
        from .task_pick_place import TaskPickPlace
        task = TaskPickPlace(bullet_client, **task_config)
    else:
        raise NotImplementedError(f"Unknown task {task_name}")

    return task

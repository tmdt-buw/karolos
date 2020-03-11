from .reach import Reach
from .push import Push

def get_task(task_config, bullet_client):
    task_name = task_config.pop("name")

    if task_name == 'reach':
        task = Reach(bullet_client, **task_config)
    elif task_name == 'push':
        task = Push(bullet_client, **task_config)
    else:
        raise ValueError()

    return task
from .reach import Reach
from .pick_place import Pick_Place

def get_task(task_config, bullet_client):
    task_name = task_config.pop("name")

    if task_name == 'reach':
        task = Reach(bullet_client, **task_config)
    elif task_name == 'pick_place':
        task = Pick_Place(bullet_client, **task_config)
    else:
        raise ValueError()

    return task
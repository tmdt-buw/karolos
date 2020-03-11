from .reach import Reach
from .move_box import MoveBox

def get_task(task_config, bullet_client):
    task_name = task_config.pop("name")

    if task_name == 'reach':
        task = Reach(bullet_client, **task_config)
    elif task_name == 'move_box':
        task = MoveBox(bullet_client, **task_config)
        bullet_client.loadURDF("plane.urdf")  # for gravity
    else:
        raise ValueError()

    return task
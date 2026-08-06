import os
import json
from datetime import datetime


def create_run_directory(base="checkpoints"):
    """
    Creates unique directory for experiment run.
    """

    run_name = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = os.path.join(
        base,
        run_name
    )

    os.makedirs(
        run_dir,
        exist_ok=True
    )

    return run_dir



def save_json(data, path):
    """
    Save dictionary as json.
    """

    with open(
        path,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )
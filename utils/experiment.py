import os
import json
from datetime import datetime
from dataclasses import asdict


def create_run_directory(
    base="results",
    experiment="stdp",
):
    """
    Creates a unique directory for an experiment run.
    """

    run_name = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = os.path.join(
        base,
        experiment,
        run_name,
    )

    os.makedirs(
        run_dir,
        exist_ok=True,
    )

    return run_dir


def save_config(config, path):
    """
    Save experiment configuration as JSON.
    """

    with open(path, "w") as f:
        json.dump(
            asdict(config),
            f,
            indent=4,
        )


def save_json(data, path):
    """
    Save dictionary as JSON.
    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=4,
        )
"""
Default configuration for SHD experiments.
"""

from dataclasses import dataclass


@dataclass
class Config:

    # -----------------------
    # Dataset
    # -----------------------

    dataset_root: str = "./data"

    time_steps: int = 200

    n_input: int = 700

    n_classes: int = 20


    # -----------------------
    # Network
    # -----------------------

    hidden1: int = 200

    hidden2: int = 100

    input_gain: float = 15.0


    # -----------------------
    # LIF neuron
    # -----------------------

    tau_m: float = 20.0

    v_rest: float = -65.0

    v_reset: float = -65.0

    v_threshold: float = -50.0

    tau_adaptation: float = 100.0

    adaptation_strength: float = 2.0


    # -----------------------
    # STDP
    # -----------------------

    a_plus: float = 0.001

    a_minus: float = 0.0012

    tau_plus: float = 20.0

    tau_minus: float = 20.0

    w_min: float = 0.0

    w_max: float = 0.5

    # колькасць эпох папярэдняга STDP-навучання
    stdp_pretrain_epochs: int = 5

    
    # -----------------------
    # Triplet STDP
    # -----------------------

    triplet: bool = False

    a3_plus: float = 0.0005

    a3_minus: float = 0.0006

    tau_pre_slow: float = 100.0

    tau_post_slow: float = 100.0
    
    # -----------------------
    # Homeostasis
    # -----------------------

    target_rate: float = 0.05

    tau_homeostasis: float = 5000.0

    homeostasis_strength: float = 0.02


    # -----------------------
    # Supervised training
    # -----------------------

    batch_size: int = 32

    learning_rate: float = 1e-3

    epochs: int = 30 # supervised

    weight_decay: float = 1e-5


    # -----------------------
    # Device
    # -----------------------

    device: str = "cuda"


    # -----------------------
    # Misc
    # -----------------------

    seed: int = 42

    save_path: str = "checkpoints/model.pt"
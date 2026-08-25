"""
Controlled Pair STDP vs Triplet STDP experiment.

Purpose
-------
Direct comparison of Pair STDP and Triplet STDP under identical
experimental conditions.

Both networks receive:
    - exactly the same SHD batch
    - exactly the same initial weights
    - exactly the same neuron parameters
    - exactly the same homeostasis parameters
    - exactly the same input gain
    - exactly the same number of batches

The ONLY difference is the plasticity rule:

Pair:
    a2_plus  = pair LTP
    a2_minus = pair LTD
    a3_plus  = 0
    a3_minus = 0

Triplet:
    same pair terms
    +
    non-zero triplet terms

This experiment does NOT modify the original Pair STDP results.
"""

import os
import json
import random
from datetime import datetime

import numpy as np
import torch

from datasets.shd import SHDDataset
from network.hierarchical_snn import HierarchicalSNN
from network.triplet.triplet_snn import TripletHierarchicalSNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIME_STEPS = 200
N_INPUT = 700

N_HIDDEN1 = 200
N_HIDDEN2 = 100

N_CLASSES = 20

BATCH_SIZE = 32

# Number of batches used in controlled experiment.
# Keep small for the first validation run.
BATCH_LIMIT = 255

INPUT_GAIN = 15.0


# ============================================================
# NEURON PARAMETERS
# ============================================================

NEURON_PARAMS = {
    "tau_m": 20.0,
    "v_rest": -65.0,
    "v_reset": -65.0,
    "v_threshold": -50.0,
}


# ============================================================
# HOMEOSTASIS PARAMETERS
#
# IMPORTANT:
# Homeostasis implementation expects "strength",
# NOT "homeostasis_strength".
# ============================================================

HOMEOSTASIS_PARAMS = {
    "target_rate": 0.05,
    "tau_homeostasis": 5000.0,
    "strength": 0.02,
}


# ============================================================
# PAIR STDP PARAMETERS
# ============================================================

PAIR_PARAMS = {
    "a_plus": 0.001,
    "a_minus": 0.0012,
    "tau_plus": 20.0,
    "tau_minus": 20.0,
    "w_min": 0.0,
    "w_max": 0.5,
}


# ============================================================
# TRIPLET STDP PARAMETERS
#
# Pair terms are intentionally identical to Pair STDP.
#
# Triplet terms are additional.
# ============================================================

TRIPLET_PARAMS = {
    "a2_plus": 0.001,
    "a2_minus": 0.0012,

    "a3_plus": 0.001,
    "a3_minus": 0.001,

    "tau_plus": 20.0,
    "tau_minus": 20.0,

    "tau_x": 100.0,
    "tau_y": 100.0,

    "w_min": 0.0,
    "w_max": 0.5,
}


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    """
    Set all relevant random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# RESULT DIRECTORY
# ============================================================

def create_results_directory():
    """
    Create unique result directory.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = os.path.join(
        "results",
        "controlled_pair_triplet",
        timestamp,
    )

    os.makedirs(
        path,
        exist_ok=True,
    )

    return path


# ============================================================
# DATA LOADING
# ============================================================

def load_batch():
    """
    Load exactly one deterministic batch from SHD.

    Dataset ordering is fixed by the seed and the first samples
    are used directly, avoiding DataLoader shuffling.
    """

    dataset = SHDDataset(
        root="./data",
        train=True,
        time_steps=TIME_STEPS,
        n_input=N_INPUT,
    )

    samples = []
    labels = []

    for index in range(BATCH_SIZE):

        spikes, label = dataset[index]

        samples.append(spikes)
        labels.append(label)

    batch = torch.stack(samples)
    labels = torch.stack(labels)

    return batch, labels


# ============================================================
# MODEL CREATION
# ============================================================

def create_pair_model():
    """
    Create original Pair-STDP network.
    """

    model = HierarchicalSNN(
        n_input=N_INPUT,
        n_hidden1=N_HIDDEN1,
        n_hidden2=N_HIDDEN2,
        n_classes=N_CLASSES,

        neuron_params=NEURON_PARAMS,

        stdp_params1=PAIR_PARAMS,
        stdp_params2=PAIR_PARAMS,

        homeostasis_params=HOMEOSTASIS_PARAMS,

        input_gain=INPUT_GAIN,

        use_classifier=False,
    )

    return model


def create_triplet_model():
    """
    Create Triplet-STDP network.
    """

    model = TripletHierarchicalSNN(
        n_input=N_INPUT,
        n_hidden1=N_HIDDEN1,
        n_hidden2=N_HIDDEN2,
        n_classes=N_CLASSES,

        neuron_params=NEURON_PARAMS,

        triplet_params1=TRIPLET_PARAMS,
        triplet_params2=TRIPLET_PARAMS,

        homeostasis_params=HOMEOSTASIS_PARAMS,

        input_gain=INPUT_GAIN,

        use_classifier=False,
    )

    return model


# ============================================================
# WEIGHT EXTRACTION
# ============================================================

def get_weights(model):
    """
    Return copies of both layer weight matrices.
    """

    return {
        "layer1": model.layer1.weights.detach().clone(),
        "layer2": model.layer2.weights.detach().clone(),
    }


# ============================================================
# SET WEIGHTS
# ============================================================

def set_weights(model, weights):
    """
    Copy externally supplied weights into the model.
    """

    with torch.no_grad():

        model.layer1.weights.copy_(
            weights["layer1"]
        )

        model.layer2.weights.copy_(
            weights["layer2"]
        )


# ============================================================
# WEIGHT STATISTICS
# ============================================================

def weight_statistics(initial, final):
    """
    Calculate weight-change statistics.
    """

    delta = final - initial

    total = delta.numel()

    increased = (
        (delta > 1e-12)
        .sum()
        .item()
        / total
        * 100.0
    )

    decreased = (
        (delta < -1e-12)
        .sum()
        .item()
        / total
        * 100.0
    )

    near_zero = (
        (delta.abs() <= 1e-12)
        .sum()
        .item()
        / total
        * 100.0
    )

    return {
        "initial_mean": initial.mean().item(),
        "final_mean": final.mean().item(),

        "delta_mean": delta.mean().item(),
        "delta_std": delta.std().item(),
        "delta_min": delta.min().item(),
        "delta_max": delta.max().item(),
        "delta_median": delta.median().item(),

        "increased_percent": increased,
        "decreased_percent": decreased,
        "near_zero_percent": near_zero,

        "initial_min": initial.min().item(),
        "initial_max": initial.max().item(),

        "final_min": final.min().item(),
        "final_max": final.max().item(),
    }


# ============================================================
# DIAGNOSTICS
# ============================================================

def get_pair_diagnostics(model):
    """
    Pair layer diagnostics.

    The original Pair STDP implementation does not expose
    LTP/LTD matrices, therefore only weight changes are reported.
    """

    return {
        "layer1": None,
        "layer2": None,
    }


def get_triplet_diagnostics(model):
    """
    Extract Triplet STDP diagnostics.
    """

    result = {}

    for name, layer in [
        ("layer1", model.layer1),
        ("layer2", model.layer2),
    ]:

        if layer.last_ltp is None:
            result[name] = None
            continue

        ltp = layer.last_ltp.detach()

        ltd = layer.last_ltd.detach()

        delta = layer.last_delta_w.detach()

        result[name] = {
            "ltp_mean": ltp.mean().item(),
            "ltp_total": ltp.sum().item(),

            "ltd_mean": ltd.mean().item(),
            "ltd_total": ltd.sum().item(),

            "delta_mean": delta.mean().item(),
            "delta_total": delta.sum().item(),

            "delta_min": delta.min().item(),
            "delta_max": delta.max().item(),

            "increased_percent": (
                (delta > 1e-12).float().mean().item()
                * 100.0
            ),

            "decreased_percent": (
                (delta < -1e-12).float().mean().item()
                * 100.0
            ),
        }

    return result


# ============================================================
# JSON SERIALIZATION
# ============================================================

def save_json(path, data):
    """
    Save dictionary as JSON.
    """

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CONTROLLED PAIR VS TRIPLET STDP")
    print("=" * 60)

    print(
        f"Device      : {DEVICE}"
    )

    print(
        f"Seed        : {SEED}"
    )

    print(
        f"Time steps  : {TIME_STEPS}"
    )

    print(
        f"Batch size  : {BATCH_SIZE}"
    )

    print(
        f"Batch limit : {BATCH_LIMIT}"
    )

    print(
        f"Input       : {N_INPUT}"
    )

    print(
        f"Hidden 1    : {N_HIDDEN1}"
    )

    print(
        f"Hidden 2    : {N_HIDDEN2}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_dir = create_results_directory()

    print()
    print(
        f"Results directory: {results_dir}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("Loading SHD batch...")

    batch, labels = load_batch()

    batch = batch.to(DEVICE)
    labels = labels.to(DEVICE)

    print(
        f"Input shape : {tuple(batch.shape)}"
    )

    print(
        f"Labels shape: {tuple(labels.shape)}"
    )

    # --------------------------------------------------------
    # Convert:
    #
    # [batch,time,input]
    #
    # ->
    #
    # [time,batch,input]
    # --------------------------------------------------------

    network_input = batch.permute(
        1,
        0,
        2,
    ).contiguous()

    print(
        f"Network input: {tuple(network_input.shape)}"
    )

    # --------------------------------------------------------
    # Create Pair model
    # --------------------------------------------------------

    print()
    print("Creating Pair network...")

    set_seed(SEED)

    pair_model = create_pair_model()

    pair_model = pair_model.to(DEVICE)

    # --------------------------------------------------------
    # Save Pair initial weights
    # --------------------------------------------------------

    pair_initial = get_weights(
        pair_model
    )

    # --------------------------------------------------------
    # Create Triplet model
    # --------------------------------------------------------

    print(
        "Creating Triplet network..."
    )

    set_seed(SEED)

    triplet_model = create_triplet_model()

    triplet_model = triplet_model.to(DEVICE)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We DO NOT rely only on random seed.
    #
    # We explicitly copy Pair initial weights into Triplet.
    #
    # Therefore the starting point is mathematically identical.
    # --------------------------------------------------------

    set_weights(
        triplet_model,
        pair_initial,
    )

    triplet_initial = get_weights(
        triplet_model
    )

    # --------------------------------------------------------
    # Verify identical initial weights
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("INITIAL WEIGHT EQUALITY CHECK")
    print("=" * 60)

    for layer in [
        "layer1",
        "layer2",
    ]:

        difference = (
            triplet_initial[layer]
            -
            pair_initial[layer]
        ).abs().max().item()

        print(
            f"{layer}: max difference = "
            f"{difference:.12e}"
        )

        if difference > 1e-12:

            raise RuntimeError(
                f"Initial weights are NOT identical "
                f"for {layer}!"
            )

    print(
        "✓ Pair and Triplet start from identical weights."
    )

    # --------------------------------------------------------
    # Initial weight files
    # --------------------------------------------------------

    torch.save(
        pair_initial["layer1"].cpu(),
        os.path.join(
            results_dir,
            "initial_weights_layer1.pt",
        ),
    )

    torch.save(
        pair_initial["layer2"].cpu(),
        os.path.join(
            results_dir,
            "initial_weights_layer2.pt",
        ),
    )

    # --------------------------------------------------------
    # Initial statistics
    # --------------------------------------------------------

    initial_stats = {}

    for layer in [
        "layer1",
        "layer2",
    ]:

        initial_stats[layer] = {
            "mean": pair_initial[layer].mean().item(),
            "std": pair_initial[layer].std().item(),
            "min": pair_initial[layer].min().item(),
            "max": pair_initial[layer].max().item(),
        }

    save_json(
        os.path.join(
            results_dir,
            "initial_weight_statistics.json",
        ),
        initial_stats,
    )

    # ========================================================
    # TRAIN PAIR
    # ========================================================

    print()
    print("=" * 60)
    print("RUNNING PAIR STDP")
    print("=" * 60)

    pair_rates = []

    for batch_number in range(
        BATCH_LIMIT
    ):

        print(
            f"Pair batch "
            f"{batch_number + 1}/{BATCH_LIMIT}"
        )

        with torch.no_grad():

            spikes1, spikes2 = pair_model(
                network_input,
                apply_stdp=True,
                return_activity=True,
            )

        rate1 = spikes1.mean().item()

        rate2 = spikes2.mean().item()

        pair_rates.append(
            {
                "layer1": rate1,
                "layer2": rate2,
            }
        )

    # --------------------------------------------------------
    # Pair final weights
    # --------------------------------------------------------

    pair_final = get_weights(
        pair_model
    )

    # ========================================================
    # TRAIN TRIPLET
    # ========================================================

    print()
    print("=" * 60)
    print("RUNNING TRIPLET STDP")
    print("=" * 60)

    triplet_rates = []

    for batch_number in range(
        BATCH_LIMIT
    ):

        print(
            f"Triplet batch "
            f"{batch_number + 1}/{BATCH_LIMIT}"
        )

        with torch.no_grad():

            spikes1, spikes2 = triplet_model(
                network_input,
                apply_stdp=True,
                return_activity=True,
            )

        rate1 = spikes1.mean().item()

        rate2 = spikes2.mean().item()

        triplet_rates.append(
            {
                "layer1": rate1,
                "layer2": rate2,
            }
        )

    # --------------------------------------------------------
    # Triplet final weights
    # --------------------------------------------------------

    triplet_final = get_weights(
        triplet_model
    )

    # --------------------------------------------------------
    # Save final matrices
    # --------------------------------------------------------

    for layer in [
        "layer1",
        "layer2",
    ]:

        torch.save(
            pair_final[layer].cpu(),
            os.path.join(
                results_dir,
                f"pair_final_weights_{layer}.pt",
            ),
        )

        torch.save(
            triplet_final[layer].cpu(),
            os.path.join(
                results_dir,
                f"triplet_final_weights_{layer}.pt",
            ),
        )

        torch.save(
            (
                pair_final[layer]
                -
                pair_initial[layer]
            ).cpu(),
            os.path.join(
                results_dir,
                f"pair_delta_weights_{layer}.pt",
            ),
        )

        torch.save(
            (
                triplet_final[layer]
                -
                triplet_initial[layer]
            ).cpu(),
            os.path.join(
                results_dir,
                f"triplet_delta_weights_{layer}.pt",
            ),
        )

    # ========================================================
    # NUMERICAL COMPARISON
    # ========================================================

    print()
    print("=" * 60)
    print("CONTROLLED NUMERICAL COMPARISON")
    print("=" * 60)

    comparison = {}

    for layer in [
        "layer1",
        "layer2",
    ]:

        pair_stats = weight_statistics(
            pair_initial[layer],
            pair_final[layer],
        )

        triplet_stats = weight_statistics(
            triplet_initial[layer],
            triplet_final[layer],
        )

        delta_difference = (
            triplet_stats["delta_mean"]
            -
            pair_stats["delta_mean"]
        )

        comparison[layer] = {
            "pair": pair_stats,
            "triplet": triplet_stats,
            "triplet_minus_pair_delta_mean":
                delta_difference,
        }

        print()
        print(
            f"{layer.upper()}"
        )

        print("-" * 60)

        print("PAIR")

        print(
            f"Initial mean : "
            f"{pair_stats['initial_mean']:.10f}"
        )

        print(
            f"Final mean   : "
            f"{pair_stats['final_mean']:.10f}"
        )

        print(
            f"Mean ΔW      : "
            f"{pair_stats['delta_mean']:+.10e}"
        )

        print(
            f"Increased    : "
            f"{pair_stats['increased_percent']:.4f}%"
        )

        print(
            f"Decreased    : "
            f"{pair_stats['decreased_percent']:.4f}%"
        )

        print()

        print("TRIPLET")

        print(
            f"Initial mean : "
            f"{triplet_stats['initial_mean']:.10f}"
        )

        print(
            f"Final mean   : "
            f"{triplet_stats['final_mean']:.10f}"
        )

        print(
            f"Mean ΔW      : "
            f"{triplet_stats['delta_mean']:+.10e}"
        )

        print(
            f"Increased    : "
            f"{triplet_stats['increased_percent']:.4f}%"
        )

        print(
            f"Decreased    : "
            f"{triplet_stats['decreased_percent']:.4f}%"
        )

        print()

        print(
            "TRIPLET - PAIR"
        )

        print(
            f"Δ mean difference : "
            f"{delta_difference:+.10e}"
        )

    # ========================================================
    # FIRING RATES
    # ========================================================

    pair_final_rate1 = (
        pair_rates[-1]["layer1"]
    )

    pair_final_rate2 = (
        pair_rates[-1]["layer2"]
    )

    triplet_final_rate1 = (
        triplet_rates[-1]["layer1"]
    )

    triplet_final_rate2 = (
        triplet_rates[-1]["layer2"]
    )

    print()
    print("=" * 60)
    print("FIRING RATES")
    print("=" * 60)

    print(
        f"Pair Layer 1    : "
        f"{pair_final_rate1 * 100:.4f}%"
    )

    print(
        f"Pair Layer 2    : "
        f"{pair_final_rate2 * 100:.4f}%"
    )

    print(
        f"Triplet Layer 1 : "
        f"{triplet_final_rate1 * 100:.4f}%"
    )

    print(
        f"Triplet Layer 2 : "
        f"{triplet_final_rate2 * 100:.4f}%"
    )

    # ========================================================
    # TRIPLET DIAGNOSTICS
    # ========================================================

    triplet_diagnostics = (
        get_triplet_diagnostics(
            triplet_model
        )
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "experiment": "controlled_pair_vs_triplet_stdp",

        "seed": SEED,

        "device": DEVICE,

        "time_steps": TIME_STEPS,

        "batch_size": BATCH_SIZE,

        "batch_limit": BATCH_LIMIT,

        "architecture": {
            "n_input": N_INPUT,
            "n_hidden1": N_HIDDEN1,
            "n_hidden2": N_HIDDEN2,
            "n_classes": N_CLASSES,
        },

        "input_gain": INPUT_GAIN,

        "pair_parameters": PAIR_PARAMS,

        "triplet_parameters": TRIPLET_PARAMS,

        "neuron_parameters": NEURON_PARAMS,

        "homeostasis_parameters":
            HOMEOSTASIS_PARAMS,

        "initial_weights_identical": True,

        "comparison": comparison,

        "pair_firing_rates": pair_rates,

        "triplet_firing_rates":
            triplet_rates,

        "triplet_diagnostics":
            triplet_diagnostics,
    }

    save_json(
        os.path.join(
            results_dir,
            "comparison_summary.json",
        ),
        summary,
    )

    # ========================================================
    # SAVE RATE HISTORY
    # ========================================================

    rate_history = {
        "pair": pair_rates,
        "triplet": triplet_rates,
    }

    save_json(
        os.path.join(
            results_dir,
            "firing_rate_history.json",
        ),
        rate_history,
    )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 60)
    print("CONTROLLED EXPERIMENT FINISHED")
    print("=" * 60)

    print()
    print(
        "Pair and Triplet used exactly the same "
        "initial weights."
    )

    print(
        "Only the plasticity rule differs."
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        results_dir
    )

    print()
    print(
        "Files:"
    )

    print(
        "  initial_weights_layer1.pt"
    )

    print(
        "  initial_weights_layer2.pt"
    )

    print(
        "  pair_final_weights_layer1.pt"
    )

    print(
        "  pair_final_weights_layer2.pt"
    )

    print(
        "  triplet_final_weights_layer1.pt"
    )

    print(
        "  triplet_final_weights_layer2.pt"
    )

    print(
        "  pair_delta_weights_layer1.pt"
    )

    print(
        "  pair_delta_weights_layer2.pt"
    )

    print(
        "  triplet_delta_weights_layer1.pt"
    )

    print(
        "  triplet_delta_weights_layer2.pt"
    )

    print(
        "  initial_weight_statistics.json"
    )

    print(
        "  firing_rate_history.json"
    )

    print(
        "  comparison_summary.json"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
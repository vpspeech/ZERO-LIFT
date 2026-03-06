#!/usr/bin/env python3
"""
get_domain_aware_lur.py

Compute Domain-Aware Layer Utilization Rate (LUR) for Wav2Vec2 using
pairs of audio files:
    (adult_speech.wav, pitch_formant_modified.wav)

The adult speech acts as the baseline and the modified speech is the input.
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from captum.attr import LayerIntegratedGradients
import random
import gc
from typing import List, Tuple

# reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################
# USER INPUT: LIST OF AUDIO PAIRS
# (adult_speech, pitch_formant_modified_speech)
############################################################

audio_pairs: List[Tuple[str, str]] = [
    ("adult_001.wav", "modified_001.wav"),
    ("adult_002.wav", "modified_002.wav"),
    ("adult_003.wav", "modified_003.wav"),
]

N_SAMPLES = len(audio_pairs)
MIN_AUDIO_LEN = 320

############################################################
# MODEL LOADING
############################################################

print("Loading Wav2Vec2 model and processor...")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h"
).to(device)

model.eval()

print("Model loaded.")

dictionary = processor.tokenizer.get_vocab()
id_to_char = {v: k for k, v in dictionary.items()}

blank_id = processor.tokenizer.pad_token_id
if blank_id is None:
    blank_id = dictionary.get("|", 0)


############################################################
# UTILITY FUNCTIONS
############################################################

def pad_to_min_length(tensor, min_len):
    if tensor.size(1) < min_len:
        return torch.nn.functional.pad(tensor, (0, min_len - tensor.size(1)))
    return tensor


def pad_to_length(tensor, length):
    if tensor.size(1) < length:
        return torch.nn.functional.pad(tensor, (0, length - tensor.size(1)))
    return tensor


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


############################################################
# ANALYSIS
############################################################

n_layers = len(model.wav2vec2.encoder.layers)

all_utilization_rates = []

#################################################################
###  ---- START LUR ATTRIBUTION ANALYSIS --------             ###
#################################################################

for sample_idx, (adult_path, modified_path) in enumerate(audio_pairs):

    print(f"\nProcessing sample {sample_idx+1}/{N_SAMPLES}")

    try:

        ############################################################
        # LOAD AUDIO PAIR
        ############################################################

        adult_waveform, sr1 = torchaudio.load(adult_path)
        mod_waveform, sr2 = torchaudio.load(modified_path)

        adult_waveform = adult_waveform.squeeze(0).numpy()
        mod_waveform = mod_waveform.squeeze(0).numpy()

        if sr1 != 16000 or sr2 != 16000:
            raise ValueError("Audio must be 16kHz")

        ############################################################
        # PROCESS INPUTS
        ############################################################

        inputs = processor(
            mod_waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )

        baseline_proc = processor(
            adult_waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(device)
        baseline = baseline_proc.input_values.to(device)

        max_len_before_padding = max(
            input_values.size(1),
            baseline.size(1)
        )

        input_values = pad_to_min_length(
            input_values,
            max(MIN_AUDIO_LEN, max_len_before_padding)
        )

        baseline = pad_to_min_length(
            baseline,
            max(MIN_AUDIO_LEN, max_len_before_padding)
        )

        max_len = max(input_values.size(1), baseline.size(1))

        input_values = pad_to_length(input_values, max_len)
        baseline = pad_to_length(baseline, max_len)

        ############################################################
        # MODEL FORWARD
        ############################################################

        with torch.no_grad():

            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)[0]

            unique_ids = torch.unique(predicted_ids)

        sample_utilization_rates = []

        ############################################################
        # LAYER-WISE ATTRIBUTION
        ############################################################

        for layer_idx in range(n_layers):

            layer = model.wav2vec2.encoder.layers[layer_idx]

            forward = lambda x: model(x).logits[0]

            lig = LayerIntegratedGradients(forward, layer)

            layer_attr_vectors = []

            for target_id in unique_ids:

                pred_indices = torch.where(
                    predicted_ids == target_id.item()
                )[0]

                if pred_indices.numel() == 0:
                    continue

                attribution = lig.attribute(
                    baselines=baseline,
                    inputs=input_values,
                    target=target_id.item(),
                    internal_batch_size=256,
                    n_steps=20,
                    attribute_to_layer_input=False
                )

                filtered = torch.stack(
                    [attribution[0, idx.item()] for idx in pred_indices],
                    dim=0
                )

                layer_attr_vectors.append(filtered)

            if not layer_attr_vectors:

                sample_utilization_rates.append(0)
                continue

            layer_attr = torch.cat(layer_attr_vectors, dim=0)

            neuron_means = layer_attr.mean(dim=0)

            mu = neuron_means.mean().item()
            sigma = neuron_means.std().item()

            outliers = (
                (neuron_means < mu - sigma) |
                (neuron_means > mu + sigma)
            ).sum().item()

            utilization = outliers / neuron_means.numel()

            sample_utilization_rates.append(utilization)

            del attribution, layer_attr, neuron_means

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_utilization_rates.append(sample_utilization_rates)

    except Exception as e:

        print(f"Error in sample {sample_idx+1}: {e}")

        gc.collect()

        continue


######################################################
### AGGREGATION + VISUALIZATION
######################################################

print("\nAggregating results...")

if not all_utilization_rates:

    print("No samples processed.")

else:

    utilization_array = np.array(all_utilization_rates)

    mean_utilization_per_layer = utilization_array.mean(axis=0)
    std_utilization_per_layer = utilization_array.std(axis=0)

    n_processed = utilization_array.shape[0]

    print(f"Processed {n_processed}/{N_SAMPLES} samples.")

    layers = range(n_layers)

    plt.figure(figsize=(12,6))

    plt.plot(
        layers,
        mean_utilization_per_layer,
        marker="o",
        label="Mean LUR"
    )

    plt.fill_between(
        layers,
        mean_utilization_per_layer - std_utilization_per_layer,
        mean_utilization_per_layer + std_utilization_per_layer,
        alpha=0.1,
        label="Std Dev"
    )

    plt.xlabel("Layer number")
    plt.ylabel("Layer Utilization Rate (LUR)")
    plt.title("Aggregated LUR across audio pairs")

    plt.grid(True)
    plt.legend()
    plt.show()

print("\nFinished batch analysis.")

import os
import json
import numpy as np
import subprocess
import torch
from deepcnn_baseline import run_quick_loso


data_dir = "faicecarsdata"
sources = ["pc", "vr", "face", "bin_faceblur_pc", "bin_faceblur_vr", "bin_facecar_pc", "bin_facecar_vr"]
model_name = "sherpamodel"  # "eegnet", "midcnn", or "deepcnn"
results_root = f"results_facecars_{model_name}"
 # Set class names (adjust as needed)
class_labels = [["Face", "Car", "Blurred Face", "Blurred Car"], #pc
                ["Face", "Car", "Blurred Face", "Blurred Car"], #vr
                ["PC", "VR"], #face
                ["Face", "Blurred Face"], #bin_faceblur_pc
                ["Face", "Blurred Face"], #bin_faceblur_vr
                ["Face", "Car"], #bin_facecar_pc
                ["Face", "Car"]] #bin_facecar_vr

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for i, source in enumerate(sources):
    print(f"\n=== Running for data source: {source} ===\n")
    tuning_path = f"{results_root}/{source}/tuning_subjects.json"
    labels = class_labels[i]

    print("Classification...")
    subprocess.run([
        "python3", "classification_model.py",
        "--data_dir", data_dir, 
        "--data_source", source,
        "--model", model_name,
        "--results_dir", results_root,
        "--class_labels", *labels
    ])

print("\n=== Generating summary across all sources ===\n")
subprocess.run(["python3", "summarize_results.py", "--results_dir", results_root, "--model", model_name])



data_dir = "faicedata"
# sources = ["pc", "vr", "face"]
sources = ["vr"]
model_name = "sherpamodel" 
results_root = f"results_sherpamodel_kfold"
 # Set class names (adjust as needed)
# class_labels = [["Face", "Blur", "Scram"], #pc
#                 ["Face", "Blur", "Scram"], #vr
#                 ["PC", "VR"]] #face
class_labels = [["Face", "Blur", "Scram"]]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for i, source in enumerate(sources):
    print(f"\n=== Running for data source: {source} ===\n")
    tuning_path = f"{results_root}/{source}/tuning_subjects.json"
    labels = class_labels[i]

    print("Classification...")
    subprocess.run([
        "python3", "classification_model.py",
        "--data_dir", data_dir, 
        "--data_source", source,
        "--model", model_name,
        "--results_dir", results_root,
        "--class_labels", *labels
    ])

print("\n=== Generating summary across all sources ===\n")
subprocess.run(["python3", "summarize_results.py", "--results_dir", results_root, "--model", model_name])
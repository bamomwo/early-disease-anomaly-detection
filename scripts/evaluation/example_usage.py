#!/usr/bin/env python
"""
Example usage of the updated evaluation script for both personalized and general models.
"""

import subprocess
import sys
import os

def run_evaluation_examples():
    """Run examples of both personalized and general model evaluation."""
    
    print("=== Personalized Model Evaluation ===")
    print("Evaluating a model trained on a single participant...")
    
    # Example 1: Personalized model evaluation
    personalized_cmd = [
        "python", "scripts/evaluation/evaluate_lstm_ae.py",
        "--model-type", "personalized",
        "--participant", "001",
        "--data-path", "data/normalized"
    ]
    
    print(f"Command: {' '.join(personalized_cmd)}")
    print("This will:")
    print("- Load model from: results/lstm_ae/pure/001/final_model_001.pth")
    print("- Load config from: config/lstm_config.json")
    print("- Save figures to: results/lstm_ae/pure/001/figs/")
    print("- Evaluate only participant 001")
    print()
    
    # Uncomment to run:
    # subprocess.run(personalized_cmd)
    
    print("=== General Model Evaluation ===")
    print("Evaluating a model trained on multiple participants...")
    
    # Example 2: General model evaluation
    general_cmd = [
        "python", "scripts/evaluation/evaluate_lstm_ae.py",
        "--model-type", "general",
        "--participants", "001", "002", "003",
        "--data-path", "data/normalized",
        "--model-dir", "results/lstm_ae/general"
    ]
    
    print(f"Command: {' '.join(general_cmd)}")
    print("This will:")
    print("- Load model from: results/lstm_ae/general/final_model_general.pth")
    print("- Load config from: config/lstm_config.json")
    print("- Save figures to: results/lstm_ae/general/figs/")
    print("- Evaluate participants: 001, 002, 003")
    print()
    
    # Uncomment to run:
    # subprocess.run(general_cmd)
    
    print("=== Advanced Examples ===")
    
    # Example 3: Custom paths
    custom_cmd = [
        "python", "scripts/evaluation/evaluate_lstm_ae.py",
        "--model-type", "general",
        "--participants", "001", "002", "003", "004",
        "--model-path", "custom/path/to/model.pth",
        "--figs-dir", "custom/path/to/figures",
        "--input-size", "45"
    ]
    
    print("Custom paths example:")
    print(f"Command: {' '.join(custom_cmd)}")
    print("This will:")
    print("- Use custom model path and figures directory")
    print("- Specify input size explicitly")
    print("- Evaluate 4 participants")
    print()
    
    # Example 4: Personalized with custom paths
    personalized_custom_cmd = [
        "python", "scripts/evaluation/evaluate_lstm_ae.py",
        "--model-type", "personalized",
        "--participant", "005",
        "--model-path", "results/custom_model.pth",
        "--figs-dir", "results/custom_figs"
    ]
    
    print("Personalized with custom paths:")
    print(f"Command: {' '.join(personalized_custom_cmd)}")
    print("This will:")
    print("- Use custom model path and figures directory")
    print("- Evaluate participant 005")
    print()

if __name__ == "__main__":
    run_evaluation_examples() 
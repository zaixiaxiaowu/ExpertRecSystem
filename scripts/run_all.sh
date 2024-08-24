#!/bin/bash

scripts=(
    "ExpertRecSystem/dataset/expert_vectors.py"
    "ExpertRecSystem/dataset/expert_analysis.py"
    "ExpertRecSystem/evaluation/recall.py"
)

for script in "${scripts[@]}"; do
    echo "Running $script..."
    if python "$script"; then
        echo "$script completed successfully."
    else
        echo "Error running $script. Exiting."
        exit 1
    fi
    echo
done

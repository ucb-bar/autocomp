#!/usr/bin/env python3
"""
Example usage of the create_ft_dataset.py script.
"""

from autocomp.learn.dataset_transformer import DatasetTransformer
import json

def main():
    # Example: Transform rerun results to fine-tuning dataset
    input_file = "rerun_results.json"  # or "rerun_results.pkl"
    output_file = "fine_tuning_dataset.json"
    
    # Create transformer
    transformer = DatasetTransformer(input_file, output_file)
    
    # Run transformation
    transformer.run()
    
    # Optionally, load and inspect a few examples
    with open(output_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"\nExample entries from the dataset:")
    print(f"Total entries: {len(dataset)}")
    
    # Show first entry structure
    if dataset:
        print(f"\nFirst entry structure:")
        first_entry = dataset[0]
        print(f"- reasoning_language: {first_entry['reasoning_language']}")
        print(f"- developer: {first_entry['developer']}")
        print(f"- user: {first_entry['user'][:100]}...")
        print(f"- final: {first_entry['final']}")
        print(f"- messages: {len(first_entry['messages'])} messages")
        print(f"- metadata keys: {list(first_entry['metadata'].keys())}")

if __name__ == "__main__":
    main()


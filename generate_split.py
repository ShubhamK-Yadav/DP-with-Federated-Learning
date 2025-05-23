import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

def create_split(data_path, num_clients, output_path="split.json", seed=42):
    random.seed(seed)
    
    # Get list of file names in data_path
    data_path = Path(data_path)
    files = sorted([f.name for f in data_path.iterdir() if f.is_file()])
    
    if not files:
        raise ValueError(f"No files found in directory: {data_path}")
    
    # Shuffle and split files across clients
    random.shuffle(files)
    client_splits = defaultdict(list)
    
    for idx, file in enumerate(files):
        client_id = f"site-{(idx % num_clients) + 1}"
        client_splits[client_id].append(file)

    # Write to JSON
    with open(output_path, "w") as f:
        json.dump(client_splits, f, indent=2)
    
    print(f"Data split into {num_clients} clients and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data among clients")
    parser.add_argument("--num_clients", type=int, required=True, help="Number of clients")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_path", type=str, default="split.json", help="Output JSON path")
    
    args = parser.parse_args()
    create_split(args.data_path, args.num_clients, args.output_path)


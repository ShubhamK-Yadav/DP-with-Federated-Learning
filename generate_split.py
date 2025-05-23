import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

def create_split(data_path, num_clients, output_path="split.json", seed=42):
    random.seed(seed)

    data_path = Path(data_path)
    files = sorted([f.name for f in data_path.iterdir() if f.is_file()])

    if not files:
        raise ValueError(f"No files found in directory: {data_path}")

    random.shuffle(files)

    client_splits = defaultdict(list)

    # Determine 60:40 client group sizes
    num_group1 = int(num_clients * 0.6)
    num_group2 = num_clients - num_group1

    if num_group1 == 0 or num_group2 == 0:
        raise ValueError("Number of clients is too small for a 60:40 split.")

    total_files = len(files)
    group1_files = int(total_files * 0.6)
    group2_files = total_files - group1_files

    group1_clients = [f"site-{i+1}" for i in range(num_group1)]
    group2_clients = [f"site-{i+1+num_group1}" for i in range(num_group2)]

    # Assign files
    group1_chunk = group1_files // num_group1
    group2_chunk = group2_files // num_group2

    start = 0
    for client in group1_clients:
        client_splits[client] = files[start:start+group1_chunk]
        start += group1_chunk

    for client in group2_clients:
        client_splits[client] = files[start:start+group2_chunk]
        start += group2_chunk

    # Distribute any remaining files
    remaining = files[start:]
    all_clients = group1_clients + group2_clients
    for idx, file in enumerate(remaining):
        client_splits[all_clients[idx % len(all_clients)]].append(file)

    with open(output_path, "w") as f:
        json.dump(client_splits, f, indent=2)

    print(f"Data split 60:40 into {num_clients} clients and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data among clients")
    parser.add_argument("--num_clients", type=int, required=True, help="Number of clients")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_path", type=str, default="split.json", help="Output JSON path")
    
    args = parser.parse_args()
    create_split(args.data_path, args.num_clients, args.output_path)


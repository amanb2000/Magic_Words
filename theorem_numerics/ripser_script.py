
# Take in command line args for:
# - input file path (.jsonl)
# - layer to analyze (int from 0 to 32)
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from ripser import ripser

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze connected components of a layer using Ripser')
    parser.add_argument('--input_file', type=str, help='Path to input file')
    parser.add_argument('--layer', type=int, help='Layer to analyze')
    parser.add_argument('--output_file', type=str, default=None, help='Path to output file. Do not include extension.')
    parser.add_argument('--all_layers', type=bool, help='Analyze all layers (overrides --layer)')

    args = parser.parse_args()

    # print out the args nicely
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    return args

def load_jsonl(file_path, layer_num):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    # Extract the final token representations
    final_token_reps = torch.tensor([item["final_token_rep"][layer_num] for item in data])

    return final_token_reps

def load_jsonl_all(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    # Extract the final token representations
    final_token_reps = [torch.tensor([item["final_token_rep"][layer_num] for item in data]) for layer_num in range(33)]

    return final_token_reps

def count_connected_components(dataset, epsilon):
    """
    Count the number of connected components in a dataset using epsilon-radius balls.

    Args:
        dataset (torch.Tensor or np.ndarray): Dataset of shape (n_points, d).
        epsilon (float): Radius for epsilon-balls to determine connectivity.

    Returns:
        int: Number of connected components.
    """
    # Ensure data is in numpy format
    if isinstance(dataset, torch.Tensor):
        data = dataset.numpy()
    elif isinstance(dataset, np.ndarray):
        data = dataset
    else:
        raise ValueError("Dataset must be a torch.Tensor or np.ndarray")

    # Compute persistence diagram
    result = ripser(data, maxdim=0, thresh=epsilon)
    diagrams = result['dgms']
    diagram_h0 = diagrams[0]

    # Count connected components properly
    # Consider every birth-death pair where death time is infinity or greater than birth plus a small threshold
    num_components = np.sum(diagram_h0[:, 1] == np.inf)
    return num_components, diagram_h0

def plot_diagram(diagram, output_file=None):
    x = diagram[:, 1]

    y = [len(x) - a for a in range(len(x))]

    plt.plot(x, y)
    plt.xlabel("epsilon")
    plt.ylabel("number of connected components")
    plt.title("Connected Components vs Epsilon")

    if output_file is not None:
        plt.savefig(output_file + ".png")
        print(f"Saved plot to {output_file}.png")
        # Clear the plot
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_args()

    if args.all_layers:
        data = load_jsonl_all(args.input_file)
        layer_i = 0
        for data_layer in data:
            # Count connected components
            epsilon = 200
            num_components = 2
            while num_components > 1:
                num_components, diagram_h0 = count_connected_components(data_layer, epsilon)
                epsilon *= 2

            plot_diagram(diagram_h0, output_file=args.output_file + f"_layer{layer_i}")

            # Save diagram to np file if output file is provided
            if args.output_file is not None:
                np.save(args.output_file + f"_layer{layer_i}.npy", diagram_h0)
                print(f"Saved diagram to {args.output_file}_layer{layer_i}.npy")
            layer_i += 1
    else:
        # Load data
        data = load_jsonl(args.input_file, args.layer)

        # Count connected components
        epsilon = 200
        num_components = 2
        while num_components > 1:
            num_components, diagram_h0 = count_connected_components(data, epsilon)
            epsilon *= 2

        plot_diagram(diagram_h0, output_file=args.output_file)
        
        # Save diagram to np file if output file is provided
        if args.output_file is not None:
            np.save(args.output_file + ".npy", diagram_h0)
            print(f"Saved diagram to {args.output_file}.npy")
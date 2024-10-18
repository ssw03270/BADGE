import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_data(output_dir):
    coords_path = os.path.join(output_dir, 'predicted_coords.npy')

    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Coordinates file not found at {coords_path}")

    predicted_coords = np.load(coords_path)

    return predicted_coords

def visualize(processed_coords, save_dir, sample_size=100):
    """
    Visualize the processed coordinates.

    Parameters:
    - processed_coords: list of numpy arrays, each containing the valid coordinates for a sample
    - save_dir: directory to save the plots
    - sample_size: number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(sample_size, len(processed_coords))

    for i in range(num_samples):
        coords_i = processed_coords[i]  # Shape: (num_coords, 2)

        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        if coords_i.shape[0] == 0:
            plt.close()
            continue  # Skip if there are no valid coordinates

        x = coords_i[:, 0]
        y = coords_i[:, 1]

        # Plot the building outline
        ax.plot(x, y, marker='o', linestyle='-')
        
        polygon = patches.Polygon(coords_i, closed=True, fill=True, edgecolor='k', facecolor='blue', alpha=0.5)
        ax.add_patch(polygon)

        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_aspect('equal')

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(save_dir, f'sample_{i+1}.png')
        plt.savefig(plot_path)
        plt.close()

    print(f"Visualization of {num_samples} samples completed. Plots saved to '{save_dir}'")

def main():
    parser = argparse.ArgumentParser(description='Visualize Inference Results.')
    parser.add_argument('--output_dir', type=str, default='inference_outputs', help='Directory where inference results are saved.')
    parser.add_argument('--save_dir', type=str, default='visualizations', help='Directory to save the visualization plots.')
    parser.add_argument('--sample_size', type=int, default=100, help='Number of individual samples to visualize.')
    parser.add_argument('--aggregate', action='store_true', help='Whether to create an aggregate visualization.')
    args = parser.parse_args()

    predicted_coords = load_data(args.output_dir)

    # Visualize individual samples
    visualize(predicted_coords, args.save_dir, sample_size=args.sample_size)

if __name__ == "__main__":
    main()

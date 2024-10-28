import os
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from PIL import Image

def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): Path of the directory to create.
    """
    os.makedirs(path, exist_ok=True)

def load_depth_image(filename):
    """Loads the depth image from the txt file."""
    with open(filename, 'r') as file:
        rows, cols = map(int, file.readline().split())
        depth_values = np.loadtxt(file).reshape((rows, cols))
    return depth_values, rows, cols

def depth_to_pointcloud(depth_image, focal_length, sensor_width):
    """Converts a depth image to a 3D point cloud using camera intrinsics."""
    rows, cols = depth_image.shape
    f_x = f_y = focal_length * (cols / sensor_width)
    c_x, c_y = cols / 2, rows / 2
    points = np.array([[ (u - c_x) * z / f_x, (v - c_y) * z / f_y, z]
                       for v in range(rows) for u in range(cols)
                       if (z := depth_image[v, u]) > 0])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def classify_point_cloud_dbscan(points, eps=0.05, min_samples=10):
    """Clusters points using DBSCAN and returns the cluster labels."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

def create_classification_image(depth_image, labels, rows, cols):
    """Creates a classification image based on DBSCAN results."""
    classification_image = np.zeros((rows, cols), dtype=np.uint8)
    valid_depth_points = depth_image.flatten() > 0
    classification_image_flat = classification_image.flatten()
    classification_image_flat[valid_depth_points] = labels + 1
    return classification_image_flat.reshape((rows, cols))

def save_classification_image(classification_image, save_path):
    """Saves the classification image as a PNG with a fixed size of 1260x704."""
    image = Image.fromarray(classification_image.astype(np.uint8))
    resized_image = image.resize((1260, 704), Image.LANCZOS)
    resized_image.save(save_path)

def paint_clusters(classification_image):
    """Paint the largest cluster in black and all others in white (grayscale, 1 channel)."""
    unique, counts = np.unique(classification_image, return_counts=True)

    if len(unique) == 0:  # No clusters found
        return np.zeros(classification_image.shape, dtype=np.uint8)

    largest_label = unique[counts.argmax()]  # Largest cluster
    grayscale_image = np.zeros(classification_image.shape, dtype=np.uint8)
    
    # Set the largest cluster to 0 (black) and others to 255 (white) then reverse it
    grayscale_image[classification_image != largest_label] = 255
    grayscale_image[:] = 255 - grayscale_image[:]

    return grayscale_image

def main(args: argparse.Namespace) -> None:
    """
    Main function to detect mirror with DBSCAN.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    root_path = Path(args.input_path).resolve()
    parent_name = root_path.parent.name
    output_root = root_path.parent / f"{parent_name}_nerfstudio"

    # Create new directory structure
    create_directory(output_root)
    
    save_dir = output_root / "dbscan_masks"
    create_directory(save_dir)

    # Read depth data
    depth_image_dir = root_path.parent / "colmap" / "depth_images"

    focal_length = 26  # mm
    sensor_width = 35  # mm

    for file_number in range(1, 75):
        print(f"Processing image {file_number:04d}")
        depth_file = f'{depth_image_dir}/{file_number:04d}_depth.txt'
        depth_image, rows, cols = load_depth_image(depth_file)

        pcd = depth_to_pointcloud(depth_image, focal_length, sensor_width)
        labels = classify_point_cloud_dbscan(np.asarray(pcd.points), eps=0.07, min_samples=10)

        classification_image = create_classification_image(depth_image, labels, rows, cols)
        colored_image = paint_clusters(classification_image)

        save_filename = f"{file_number:04d}.png"
        save_path = os.path.join(save_dir, save_filename)
        save_classification_image(colored_image, save_path)

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect mirror with DBSCAN.")
    parser.add_argument("--input_path", help="Path to the root directory of mask output")
    args = parser.parse_args()
    
    main(args)

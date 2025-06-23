import numpy as np
from scipy.spatial import cKDTree
import time
import open3d as o3d

def compute_centroids(points_a, points_b, fraction=0.15, method='mean'):
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    
    # Find the closest points of points_a in points_b
    distances_a, indices_a = tree_a.query(points_b)
    
    # Find the closest points of points_b in points_a
    distances_b, indices_b = tree_b.query(points_a)

    # Merge and sort the distances 
    all_distances = np.concatenate([distances_a, distances_b])
    sorted_indices = np.argsort(all_distances)
    
    num_selected = int(fraction * len(all_distances))
    selected_a = sorted_indices[:num_selected][sorted_indices[:num_selected] < len(points_b)]
    selected_b = sorted_indices[:num_selected][sorted_indices[:num_selected] >= len(points_b)] - len(points_b)
    
    # Compute centroids using the specified method
    if method == 'mean':
        centroid_a = np.mean(points_a[indices_a[selected_a]], axis=0)
        centroid_b = np.mean(points_b[indices_b[selected_b]], axis=0)
    elif method == 'median':
        centroid_a = np.median(points_a[indices_a[selected_a]], axis=0)
        centroid_b = np.median(points_b[indices_b[selected_b]], axis=0)
    else:
        raise ValueError("Method must be either 'mean' or 'median'")
    
    return centroid_a, centroid_b

def depth_to_point_cloud(depth_image, f=1000, W=1280, H=720):
    c_x, c_y = W / 2, H / 2
    point_cloud = []
    z_values = []

    non_zero_indices = np.transpose(np.nonzero(depth_image))

    for v, u in non_zero_indices:
        Z = depth_image[v, u]
        X = (u - c_x) * Z / f
        Y = (v - c_y) * Z / f
        point_cloud.append([X, Y, Z])
        z_values.append(Z)

    return np.array(point_cloud), np.array(z_values)



def visualize_point_clouds(cloud1, cloud2):
    # Convert numpy arrays to open3d format
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cloud1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2)

    # Set the colors (optional)
    pcd1.paint_uniform_color([1, 0, 0])  # red for first cloud
    pcd2.paint_uniform_color([0, 1, 0])  # green for second cloud

    # Visualize
    o3d.visualization.draw_geometries([pcd1, pcd2])

def normalize_array(arr):
    """
    Standardizes a 1D or 2D numpy array.

    Parameters:
        - arr (numpy.ndarray): The array to be normalized.

    Returns:
        - numpy.ndarray: The normalized array.
    """
    # If the array is 1D
    if len(arr.shape) == 1:
        mean = np.mean(arr)
        std = np.std(arr)
        normalized = (arr - mean) / std
    # If the array is 2D
    elif len(arr.shape) == 2:
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        normalized = (arr - mean) / std
    else:
        raise ValueError("Only 1D or 2D arrays are supported")

    return normalized


if __name__ == "__main__":
    cloud1 = np.random.rand(1000, 3)
    cloud2 = np.random.rand(1000, 3) + [1, 1, 1]  # shifted for distinct visualization

    visualize_point_clouds(cloud1, cloud2)

    points_teacher = np.random.rand(1000, 3)
    points_student = np.random.rand(1000, 3)

    start_time = time.time()
    centroid_teacher, centroid_student = compute_centroids(points_teacher, points_student)
    end_time = time.time()

    distance = np.linalg.norm(centroid_teacher - centroid_student)

    print("Teacher's Centroid:", centroid_teacher)
    print("Student's Centroid:", centroid_student)
    print("Distance Between Centroids:", distance)
    print("Computational Time:", end_time - start_time, "seconds")

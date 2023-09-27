import numpy as np
from stl import mesh
import open3d as o3


def get_points_from_triangles(triangles, flip_axis=-1):
    points_A = triangles[:, 0:3]
    points_B = triangles[:, 3:6]
    points_C = triangles[:, 6:9]

    if flip_axis > -1:
        points_A[:, flip_axis] = -points_A[:, flip_axis]
        points_B[:, flip_axis] = -points_B[:, flip_axis]
        points_C[:, flip_axis] = -points_C[:, flip_axis]

    return points_A, points_B, points_C


def stl_to_xyz_with_normals_vectorized(
    input_stl_file, output_xyz_file, sep=",", stride=1, flip_axis=-1
):
    stl_mesh = mesh.Mesh.from_file(input_stl_file)
    triangles = stl_mesh.vectors.reshape(-1, 9)  # Reshape to (N, 9) array
    triangles = triangles[::stride, :]

    # Calculate normals for all triangles
    points_A, points_B, points_C = get_points_from_triangles(triangles, flip_axis)
    edges1 = points_B - points_A
    edges2 = points_C - points_A
    normals = np.cross(edges1, edges2)
    if flip_axis > -1:
        normals = -normals
    norm_mags = np.linalg.norm(normals, axis=1)
    eps = 1e-8
    normals /= norm_mags[:, np.newaxis] + eps

    # Combine vertices and normals
    vertices_with_normals = np.hstack((points_A, normals))

    # Save to XYZ file
    np.savetxt(output_xyz_file, vertices_with_normals, delimiter=sep, fmt="%.8f")


def read_point_cloud_text(points_file, stride=1, flip_axis=-1):
    points = np.loadtxt(points_file, delimiter=",")[::stride, :3].copy()
    # transformation
    if flip_axis > -1:
        points[:, flip_axis] = -points[:, flip_axis]
        points[:, flip_axis + 3] = -points[:, flip_axis + 3]
    return points


def visualize_pointcloud(points_file, stride=1, flip_axis=-1, postprocess_fn=None):
    points = np.loadtxt(points_file, delimiter=",")[::stride, :3].copy()
    visualize_pointcloud_np(points, flip_axis, postprocess_fn)


def visualize_pointcloud_np(points, flip_axis=-1, postprocess_fn=None):
    pcd = o3.geometry.PointCloud()
    # transformation
    if flip_axis > -1:
        points[:, flip_axis] = -points[:, flip_axis]
    if postprocess_fn:
        points = postprocess_fn(points)
    # visualize
    pcd.points = o3.utility.Vector3dVector(points)
    o3.visualization.draw_plotly([pcd])

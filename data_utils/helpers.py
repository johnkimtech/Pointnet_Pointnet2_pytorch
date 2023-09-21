import numpy as np
from stl import mesh
import open3d as o3
import os
from IPython.display import display
from ipyfilechooser import FileChooser


def stl_to_xyz_with_normals_vectorized(input_stl_file, output_xyz_file, sep=',', stride=1):
    stl_mesh = mesh.Mesh.from_file(input_stl_file)
    triangles = stl_mesh.vectors.reshape(-1, 9)  # Reshape to (N, 9) array
    triangles = triangles[::stride,:]

    # Calculate normals for all triangles
    edges1 = triangles[:, 3:6] - triangles[:, 0:3]
    edges2 = triangles[:, 6:9] - triangles[:, 0:3]
    normals = np.cross(edges1, edges2)
    norm_mags = np.linalg.norm(normals, axis=1)
    eps = 1e-8
    normals /= (norm_mags[:, np.newaxis] + eps)

    # Combine vertices and normals
    vertices_with_normals = np.hstack((triangles[:, 0:3], normals))

    # Save to XYZ file
    np.savetxt(output_xyz_file, vertices_with_normals, delimiter=sep, fmt='%.8f')



def visualize_pointcloud(points_file, stride=1):
    points = np.loadtxt(points_file, delimiter=',')[::stride, :3].copy()
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(points)
    o3.visualization.draw_plotly([pcd])
import open3d as o3d

# Load XYZ file
point_cloud = o3d.io.read_point_cloud("../data/airborne_lidar2.xyz", format="xyz")

# Save as PLY
o3d.io.write_point_cloud("../data/airborne_lidar2.ply", point_cloud)

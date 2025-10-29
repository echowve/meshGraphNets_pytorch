import h5py
import numpy as np
import glob, os
import pyvista as pv

def vtk_sequence_to_h5(vtk_dir, output_path):
    vtk_files = sorted([f for f in os.listdir(vtk_dir) if f.endswith('.vtk')])
    
    # Extract case names from filenames
    case_names = sorted(set('_'.join(f.split('_')[:-1]) for f in vtk_files))
    
    trajectories = {}
    for case_name in case_names:
        files = sorted(glob.glob(os.path.join(vtk_dir, f"{case_name}_t*.vtk")))
        if len(files) == 0:
            continue
        meshes = [pv.read(f) for f in files]
        N = meshes[0].n_points
        
        # Sanity check: ensure all meshes have same number of points
        for m in meshes:
            assert m.n_points == N, f"Mesh {m} has different number of points!"

        pos = np.stack([m.points for m in meshes], axis=0).astype(np.float32)
        
        # Handle missing keys gracefully
        try:
            velocity = np.stack([m.point_data['velocity'] for m in meshes], axis=0).astype(np.float32)
        except KeyError:
            raise KeyError(f"'velocity' not found in point_data for case {case_name}")

        try:
            pressure = np.stack([m.point_data['total_pressure'] for m in meshes], axis=0).astype(np.float32)
        except KeyError:
            raise KeyError(f"'total_pressure' not found in point_data for case {case_name}")

        node_type = np.zeros((len(meshes), N, 1), dtype=np.int32)
        
        # For tetrahedral mesh: cells stored as (num_points_per_cell, v0, v1, v2, v3)
        cells = meshes[0].cells.reshape(-1, 5)[:, 1:].astype(np.int32)

        trajectories[case_name] = dict(
            pos=pos,
            node_type=node_type,
            velocity=velocity,
            pressure=pressure,
            cells=cells,
        )

    with h5py.File(output_path, "w") as f:
        for case_name, data in trajectories.items():
            grp = f.create_group(case_name)
            for k, v in data.items():
                grp.create_dataset(k, data=v)
    print(f"Saved {output_path}")


vtk_sequence_to_h5("/usagers3/nashe/cfd_dataset", "/usagers3/nashe/cfd_dataset/train.h5")
# vtk_sequence_to_h5("my_dataset/test", "my_dataset/test.h5")



# import pyvista as pv

# # Load the file
# mesh = pv.read(r"/usagers3/nashe/cfd_dataset/aorta_case01_t000.vtk") 

# # Print summary info
# print(mesh)

# # See available data fields
# print("Point Data:", mesh.point_data.keys())
# print("Cell Data:", mesh.cell_data.keys())

# mesh.plot()

# # Get the point coordinates
# points = mesh.points
# print("Points shape:", points.shape)
# print("First 5 points:\n", points[:5])

# # Get the cell connectivity
# cells = mesh.cells
# print("Cells (flattened):", cells[:20])

# # Get the offset array
# offset = mesh.offset
# print("Cell offsets:", offset[:10])

# # Get the cell types (VTK enum codes)
# cell_types = mesh.celltypes
# print("Cell types:", cell_types[:10])


# # UnstructuredGrid (0x7fdcc5ea3520)
# #   N Cells:    513411
# #   N Points:   147193
# #   X Bounds:   9.789e-03, 3.454e-02
# #   Y Bounds:   4.193e-02, 8.287e-02
# #   Z Bounds:   4.910e-02, 1.186e-01
# #   N Arrays:   6
# # Point Data: ['total_pressure', 'velocity', 'velocity_magnitude', 'x_velocity', 'y_velocity', 'z_velocity']
# # Cell Data: []
# # Points shape: (147193, 3)
# # First 5 points:
# #  [[0.02463894 0.05220252 0.08692422]
# #  [0.02442449 0.05212553 0.08725515]
# #  [0.02458854 0.05185655 0.08721188]
# #  [0.0245819  0.05217541 0.08688121]
# #  [0.02436897 0.05209755 0.08721577]]
# # Cells (flattened): [ 4 19 20 21 30  4 19 21 23 30  4 29 21 20 39  4 20 21 30 39]
# # Cell offsets: [ 0  4  8 12 16 20 24 28 32 36]
# # Cell types: [10 10 10 10 10 10 10 10 10 10]
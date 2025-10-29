import os
import shutil

# Original dataset folder
vtk_dir = "/usagers3/nashe/cfd_dataset"
# Base case prefix (the one you already have)
base_case = "aorta_case01"
# Number of dummy cases to create (excluding the original)
num_dummy_cases = 5

# Get all files for the original case
original_files = sorted(f for f in os.listdir(vtk_dir) if f.startswith(base_case) and f.endswith(".vtk"))

for i in range(2, 2 + num_dummy_cases):
    new_case_prefix = f"aorta_case{i:02d}"
    for f in original_files:
        # Create new filename by replacing base_case prefix
        new_filename = f.replace(base_case, new_case_prefix)
        src = os.path.join(vtk_dir, f)
        dst = os.path.join(vtk_dir, new_filename)
        shutil.copyfile(src, dst)
        print(f"Copied {src} -> {dst}")

print(f"{num_dummy_cases} dummy cases created successfully!")

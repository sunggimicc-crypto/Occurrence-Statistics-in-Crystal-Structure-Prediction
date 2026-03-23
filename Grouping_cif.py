from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, FrameworkComparator
import os
from ase.io import read
import glob
from tqdm import tqdm
import pandas as pd

max_num = 600

# Directory where your CIF files are stored
cif_directory = ['MLP_opt_structure/Ac2O3_mp-11107_0/']

for cif_dir in tqdm(cif_directory):
    # Load structures from CIF files
    structures = []
    filenames = []
    seen_num = 0
    for filename in os.listdir(cif_dir):
        if filename.endswith(".cif"):
            seen_num+=1
            file_path = os.path.join(cif_dir, filename)
            try:
                structure = Structure.from_file(file_path)
                structures.append(structure)
                filenames.append(filename)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
        if len(structures) == max_num:
            break
    if len(structures) == 0:
        continue
    # Initialize StructureMatcher
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=1, primitive_cell=True,scale=True, attempt_supercell=True, allow_subset=False, comparator=FrameworkComparator())
    
    # Group structures
    grouped_structures = matcher.group_structures(structures)

    # Extracting the five largest groups
    group_sizes = [(i, len(group)) for i, group in enumerate(grouped_structures)]
    largest_groups = sorted(group_sizes, key=lambda x: x[1], reverse=True)#[:3]

    # Print the results and count the number of structures in each group
    for i, group in enumerate(grouped_structures):
        group_filenames = [filenames[structures.index(struct)] for struct in group]
    print('Total data number / Top 3 group size :', seen_num, ' / ',largest_groups[0][1])

    # Prepare data for CSV
    csv_data = []
    for index, size in largest_groups:
        group = grouped_structures[index]
        group_filenames = [filenames[structures.index(struct)] for struct in group]
        for filename in group_filenames:
            csv_data.append({"Group Number": index + 1, "Filename": filename})


    # Create a DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    csv_file_path = cif_dir+'Groups_%d_%d.csv' %(seen_num, int(largest_groups[0][1]))  # Set your desired path
    df.to_csv(csv_file_path, index=False)
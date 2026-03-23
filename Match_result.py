from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, FrameworkComparator
import os
from ase.io import read
import glob
from tqdm import tqdm
import csv
    
target_data = glob.glob('binary_oxide/Ac2O3_mp-11107_0/*.cif')


# Prepare to collect data for CSV
for td in tqdm(target_data):
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True,scale=True, attempt_supercell=True, allow_subset=False, comparator=FrameworkComparator())
    filename = 'binary_oxide/Ac2O3_mp-11107_0/'+td   # Target ground truth
    structure = Structure.from_file(filename)
               
    composition = td.split('_')[0]
    
    ground_truth = glob.glob('binary_oxides_exp_cifs/%s*.cif' %(composition))
    for gt in ground_truth:
        reference_structure = Structure.from_file(gt)   
        # Compute the fit and get the similarity score
        if matcher.fit(reference_structure, structure):
            score = matcher.get_rms_dist(reference_structure, structure)[0]   
            results.append([td, gt, score])
            break
        else:
            score = 10000.0  # Use None or a specific value to indicate no match 

# Create a DataFrame from the results and save to CSV
uid = ['Filename','Ground_truth','Score']
with open('Score_exp.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(uid)
    writer.writerows(results)
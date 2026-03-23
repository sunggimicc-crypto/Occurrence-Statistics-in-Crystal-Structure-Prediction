from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import glob
from tqdm import tqdm

cif_dir = glob.glob('binary_oxides_exp_cifs/*.cif')

space_group_info, space_group_num = [], []
# Load the structure from a CIF file
for st_path in tqdm(cif_dir):
    structure = Structure.from_file(st_path)

    # Create a SpacegroupAnalyzer object
    analyzer = SpacegroupAnalyzer(structure)
    space_group_symbol = analyzer.get_space_group_symbol()
    
    # Get the space group number
    space_group_number = analyzer.get_space_group_number()  
    space_group_num.append(space_group_number)
    space_group_info.append(st_path+'\t'+str(space_group_number)+'\t'+space_group_symbol+'\n')
  
print(space_group_num)
open('binary_oxide_spacegroup.txt','w').writelines(space_group_info)
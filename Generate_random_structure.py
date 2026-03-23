import csv
from tqdm import tqdm
import math
from ase.geometry import cellpar_to_cell
from ase.io import write,read
import numpy as np
import re
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure, Composition
from pymatgen.core.periodic_table import Species
from scipy.spatial.distance import pdist, squareform
import os
import random
import glob

def is_integer(num):
    if isinstance(num, (int, float)):  # Ensure the input is numeric
        return num.is_integer() if isinstance(num, float) else True
    return False

def make_ionic(elem, oxi):
    if is_integer(oxi):
        oxi = int(oxi)
    if oxi == 0:
        return elem
    elif oxi > 0:
        return elem+str(oxi)+'+'
    else:
        return elem+str(-oxi)+'-'
        
def calculate_atomic_volume(atomic_radius):
    return (4/3) * math.pi * (atomic_radius ** 3)
    
def get_cell_length(compo,num_mul, v_occu):
    try:
        charge_dict = Composition(compo).oxi_state_guesses()[0]
    except:
        charge_dict = {ii:0 for ii in list(set(parse_formula(compo)))}
    return_dict = {}
    use_atomic, sum_atom = False, 0
    for i,v in charge_dict.items():
        atomic_radius = None
        try:
            atomic_radius = Species(make_ionic(i,v)).ionic_radius
        except:
            use_atomic = True
        if atomic_radius == None:
            use_atomic = True
        if not use_atomic:
            return_dict[i] = calculate_atomic_volume(atomic_radius)
    if use_atomic:
        for i,v in charge_dict.items():
            atomic_radius = Species(i).atomic_radius
            return_dict[i] = calculate_atomic_volume(atomic_radius)
            
    for ell in parse_formula(compo):
        sum_atom += return_dict[ell]
    sum_atom *= num_mul
    cell_l = (sum_atom * v_occu) ** (1 / 3)
    return cell_l
    
def generate_random_list(length=3):
    choices = [60, 90, 120]
    while True:
        random_list = [random.choice(choices) for _ in range(length)]
        if random_list not in [[60, 60, 120],[120, 60, 60],[60, 120, 60],[120, 120, 120],[60,60,60]]:
            return random_list
    
def parse_formula(formula):
    parse_list = re.findall(r'[A-Z][a-z]*|\d+', re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', formula))
    
    # Expand the formula
    expanded_formula = []
    for idx in range(int(len(parse_list)/2)):
        elem,num = parse_list[2*idx], int(parse_list[2*idx+1])
        expanded_formula.extend([elem] * num)    
    return expanded_formula

def format_floats(floats, decimal_places):
    return [f"{value:.{decimal_places}f}" for value in floats]

def dist_check(coords, lower_b=0.5, upper_b=10):
    re_ture = True
    dist_list = []
    for i,coord in enumerate(coords):
        for j,other in enumerate(coords):
            if j<=i:
                continue
            dist = distance_between_points(coord, other)
            dist_list.append(dist)
            if dist > upper_b or dist < lower_b:
                re_ture = False
    return re_ture, dist_list

def distance_between_points(p1, p2):
    x1, y1, z1 = float(p1[0]),float(p1[1]),float(p1[2])
    x2, y2, z2 = float(p2[0]),float(p2[1]),float(p2[2])
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def set_cell_angles_and_lengths(a, b, c, alpha, beta, gamma):
    # Convert angles from degrees to radians for computation
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # Compute the cell vectors
    cell = np.zeros((3, 3))
    cell[0] = [a, 0, 0]
    cell[1] = [b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0]
    cell[2] = [
        c * np.cos(beta_rad),
        c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad),
        c * np.sqrt(1 - np.cos(beta_rad)**2 - ((np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad))**2)
    ]

    return cell
    
def nearest_atom_distances(atoms):
    # Calculate all pairwise distances between atoms
    min_distances = []
    distances = atoms.get_all_distances(mic=True)
    for i, row in enumerate(distances):
        min_distances.append(min(row[:i].tolist()+row[i+1:].tolist()))

    # Find the minimum distance to any other atom for each atom
    return min_distances
    
def count_atoms_in_zones(atoms, cell_params):   
    # Create the unit cell from parameters
    a, b, c, alpha, beta, gamma = cell_params
    cell = cellpar_to_cell(cell_params)
    atoms.set_cell(cell)
    atoms.set_pbc([True, True, True])
    
    # Get the fractional coordinates of atoms
    fractional_positions = atoms.get_scaled_positions()
    
    # Initialize counts for each zone
    zone_counts = {f"zone_{1 + x + 2*y + 4*z}": 0 for x in (0, 1) for y in (0, 1) for z in (0, 1)}
    
    # Assign atoms to zones based on fractional coordinates
    for pos in fractional_positions:
        zone_index = 1 + (pos[0] > 0.5) + 2 * (pos[1] > 0.5) + 4 * (pos[2] > 0.5)
        zone_counts[f"zone_{zone_index}"] += 1

    return zone_counts

max_gen_num, gen_count = 10000, 0
compo_list = [dat.split('\\')[1].split('_')[0]  for dat in glob.glob('binary_oxides_exp_cifs/*cif')]
mpid_list = [dat.split('\\')[1].split('_')[1].split('.')[0]  for dat in glob.glob('binary_oxides_exp_cifs/*cif')]
v_occu_list = [1.5,2,3]

for cl_num, (compo,mpid) in tqdm(enumerate(tqdm(zip(compo_list,mpid_list)))):
    for get_idx in range(15):
        element_list = parse_formula(compo)
        cif_path = 'binary_oxides_exp_cifs/'+compo+'_'+mpid+'.cif'
        parser = CifParser(cif_path)
        structure = parser.get_structures()[0]
        num_mul = int(len(structure)/len(element_list))
        if num_mul < 1:
            num_mul = len(structure)/len(element_list)
            element_list = element_list[:int(num_mul*len(element_list))]
            num_mul = 1
        v_occu = v_occu_list[get_idx%3]
        cell_length = get_cell_length(compo,num_mul,v_occu)
        a,b,c = cell_length, cell_length, cell_length
        alpha,beta,gamma = generate_random_list()
        cell_params = [a,b,c,alpha,beta,gamma]
        new_element_list = []
        for _ in range(num_mul):
            new_element_list.extend(element_list)
        element_list = new_element_list
        
        for gen_count in range(max_gen_num):
            impossible_compo = False
            if gen_count == 50:
                break
            input_file_2 = [[0], []]
            coords = []
            for elem_idx, elem in enumerate(element_list):
                bad_pos, try_count = True, 0
                input_file_2[0] = [elem_idx+1]
                input_file_2.append([])
                while bad_pos:
                    input_file_2 = input_file_2[:-1]
                    try_count += 1

                    cell = cellpar_to_cell(cell_params)
                    fractional_coords = np.random.rand(3)
                    random_point = np.dot(fractional_coords, cell)

                    detail_list = [elem,str(random_point[0]),str(random_point[1]),str(random_point[2])]
                    new_line = '     '.join(detail_list)
                    input_file_2.append([new_line])
                    coords.append(random_point)
                    
                    with open("one.xyz", 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(input_file_2) 

                    # Read the CIF file
                    xyz_path = "one.xyz"
                    atoms = read(xyz_path)
                    atoms.set_cell(set_cell_angles_and_lengths(a, b, c, alpha, beta, gamma), scale_atoms=True)
                    write("one_cif.cif", atoms)                                       
                    atoms = read("one_cif.cif")
                    
                    if elem_idx == 0:
                        break
                    
                    near_dist = nearest_atom_distances(atoms)
                    min_dist, max_dist = min(near_dist), max(near_dist)
                    zone_counts = count_atoms_in_zones(atoms, [a,b,c,alpha,beta,gamma])
                    min_zone, max_zone = min(list(zone_counts.values())), max(list(zone_counts.values()))

                    os.makedirs("Generation_output/"+compo+'_'+mpid+'_'+str(cl_num)+"/", exist_ok=True)
                    if min_dist > 0.5 and max_dist < 3.5:
                        # Define a cubic unit cell and set periodic boundaries
                        cell_size = cell_length  # Adjust as needed
                        atoms.set_cell(set_cell_angles_and_lengths(a, b, c, alpha, beta, gamma), scale_atoms=True)
                        atoms.set_pbc((True, True, True))
                        bad_pos = False
                    if try_count > 3000:
                        impossible_compo = True
                        break
                
                if impossible_compo:
                   break
            if not impossible_compo:       
                write("Generation_output/"+compo+'_'+mpid+'_'+str(cl_num)+"/%s_%d_%d.cif" %(compo,get_idx,gen_count), atoms)
                gen_count += 1
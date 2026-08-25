# Occurrence Statistics in Crystal Structure Prediction

This repository contains the analysis codes and benchmark data for studying the relationship between **structural occurrence statistics** and **experimental realizability** in crystal structure prediction (CSP).

The central idea of this work is that frequently sampled structural basins on the potential energy surface may contain experimentally accessible structures. By combining structural occurrence with thermodynamic information, we analyze whether occurrence provides additional information beyond energy-based stability criteria.

---

## Overview

The workflow consists of:

1. Generation of random crystal structures for target compositions
2. Structural relaxation using machine-learning interatomic potentials (MLPs)
3. Identification of converged structures
4. Structural grouping into crystal structure basins
5. Calculation of occurrence statistics
6. Comparison with experimentally reported structures

The repository also includes benchmark results comparing the proposed occurrence-based analysis with other CSP approaches.

---

## Repository contents

```
.
├── Benchmark/
│   ├── generate_airss_structure_as_cif.py
│   ├── Result_natcomm_CSPML.csv
│   ├── Result_natcomm_Ours.csv
│   └── Result_natcomm_RSS.csv
│
├── binary_oxide.zip
│
├── binary_oxide_random_structure_generated/
│   └── Generated random structures and relaxed structures
│
├── analysis/
│   └── Check_converged_num.ipynb
│
└── README.md
```

---

# Analysis notebook

## Check_converged_num.ipynb

This notebook contains post-processing analyses for generated structures, including:

- convergence statistics of generated structures
- machine-learning potential relaxation success rate
- structural grouping statistics
- grouping confidence analysis
- comparison between generated structures and reference structures

The notebook expects the generated structure directories and associated CSV files produced during the CSP workflow.

---

# Structure generation and relaxation

For each chemical composition, random structures are generated and relaxed.

The analysis tracks:

- total generated structures
- successfully converged structures
- number of identified structural groups
- basin occurrence statistics

The occurrence of a structural basin is calculated from the fraction of generated structures assigned to that basin.

---

# Benchmark data

The `Benchmark` directory contains comparison results used for evaluating different structure-generation approaches.

Included benchmark results:

- RSS (random structure search)
- CSPML-based structure generation
- Proposed occurrence-based approach

---

# Data format

Generated structures are stored as CIF files.

Example:

```
binary_oxide_random_structure_generated/
    Fe2O3_mp-xxxx/
        structure_001.cif
        structure_002.cif
        ...
```

Associated CSV files contain structural grouping and analysis information.

---

# Requirements

The analysis notebooks require:

- Python >= 3.9
- numpy
- pandas
- matplotlib
- pymatgen
- tqdm

Install dependencies:

```bash
pip install numpy pandas matplotlib pymatgen tqdm
```

---

# Reproduction

A typical workflow is:

1. Generate random crystal structures
2. Relax structures using the selected ML potential
3. Group relaxed structures based on structural similarity
4. Calculate occurrence statistics
5. Analyze occurrence-energy-experiment relationships

The provided notebook can be used for evaluating convergence and structural statistics from generated datasets.

---

# Citation

If you use this repository, please cite the associated manuscript:

```
Occurrence Statistics in Crystal Structure Prediction
```

---

# License

Please refer to the repository license information before redistribution or reuse.

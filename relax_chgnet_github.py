#!/usr/bin/env python3

from __future__ import annotations

import os
import random
import sys
import traceback
import warnings
from glob import glob

from joblib import Parallel, delayed
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from chgnet.model import CHGNet, StructOptimizer


for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category)


BATCH_SIZE = 15000
MAX_SHUFFLE = 10
MAX_NUM_STRUCTURES = 3
FMAX = 0.05
MAX_STEPS = 1000

INPUT_PATTERN = "random_structures/*/*.cif"
OUTPUT_DIR = "CHGNet_relax"
LOG_DIR = "logs"


def initialize_log(logfile):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "w"):
        pass


def log(logfile, msg):
    with open(logfile, "a") as f:
        f.write(msg + "\n")


def hamming_distance(list1, list2):
    return sum(el1 != el2 for el1, el2 in zip(list1, list2))


def assign_elements(structure, max_shuffle=MAX_SHUFFLE, max_num_structures=MAX_NUM_STRUCTURES):
    lattice = structure.lattice
    coords = structure.frac_coords
    original_elements = [site.specie.symbol for site in structure]

    rng = random.Random(42)
    shuffled_lists = []

    for _ in range(max_shuffle):
        elements = original_elements.copy()
        rng.shuffle(elements)

        if elements not in shuffled_lists:
            shuffled_lists.append(elements)

    scored_lists = [
        (
            elements,
            sum(hamming_distance(elements, other) for other in shuffled_lists),
        )
        for elements in shuffled_lists
    ]

    scored_lists.sort(key=lambda x: x[1], reverse=True)

    return [
        Structure(lattice, elements, coords)
        for elements, _ in scored_lists[:max_num_structures]
    ]


def relax_structure(cif, relaxer, output_dir, logfile):
    relative_name = os.path.join(*cif.split(os.sep)[-3:])

    try:
        structure = Structure.from_file(cif)
        structures = assign_elements(structure)

        for index, structure in enumerate(structures):
            base_path = os.path.join(
                output_dir,
                relative_name.replace(".cif", f"_{index}"),
            )

            save_pkl = base_path + ".pkl"
            save_cif = base_path + ".cif"

            os.makedirs(os.path.dirname(save_pkl), exist_ok=True)

            if os.path.isfile(save_pkl):
                continue

            try:
                relax_results = relaxer.relax(
                    structure,
                    fmax=FMAX,
                    steps=MAX_STEPS,
                    verbose=False,
                    save_path=save_pkl,
                )

                final_structure = relax_results["final_structure"]

                suffix = "_converged.cif" if relaxer.converged else "_unconverged.cif"
                CifWriter(final_structure).write_file(
                    save_cif.replace(".cif", suffix)
                )

                log(
                    logfile,
                    f"{save_cif}\n"
                    "Complete\n"
                    f"Converged: {relaxer.converged}\n",
                )

            except Exception:
                log(
                    logfile,
                    f"{save_cif}\n"
                    f"{traceback.format_exc()}\n",
                )

    except Exception:
        log(
            logfile,
            f"{cif}\n"
            f"{traceback.format_exc()}\n",
        )


def main():
    batch_index = int(sys.argv[1])

    logfile = os.path.join(
        LOG_DIR,
        f"relax_batch_{batch_index}.log",
    )
    initialize_log(logfile)

    file_list = sorted(glob(INPUT_PATTERN))

    # Keep structures whose index is in [0, 8].
    file_list = [
        path
        for path in file_list
        if int(os.path.basename(path).split("_")[1]) in range(9)
    ]

    print("Total files:", len(file_list))

    start = BATCH_SIZE * (batch_index - 1)
    end = BATCH_SIZE * batch_index
    file_list = file_list[start:end]

    print("Batch size:", len(file_list))

    model = CHGNet.load()
    model.average_atom_feas = False
    relaxer = StructOptimizer(model=model)

    n_jobs = min(len(file_list), os.cpu_count() or 1)

    Parallel(
        n_jobs=n_jobs,
        verbose=1,
        batch_size=1,
    )(
        delayed(relax_structure)(
            cif,
            relaxer,
            OUTPUT_DIR,
            logfile,
        )
        for cif in file_list
    )


if __name__ == "__main__":
    main()

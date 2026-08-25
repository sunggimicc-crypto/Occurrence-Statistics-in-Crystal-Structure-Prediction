#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from pymatgen.core import Structure


def extract_if_zip(input_path: Path, workdir: Path) -> Path:
    """
    If input_path is a zip file, extract it and return the extracted directory.
    Otherwise, return input_path directly.
    """
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = workdir / input_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(extract_dir)

        return extract_dir

    return input_path


def find_cif_files(root: Path) -> List[Path]:
    """
    Recursively find CIF files.
    """
    return sorted(root.rglob("*.cif"))


def integer_composition_from_structure(
    structure: Structure,
    tol: float = 1e-6,
) -> Dict[str, int]:
    """
    Read the exact integer atom counts from the source CIF cell.

    Example:
        source cell Ac2O3 -> {"Ac": 2, "O": 3}
        source cell V4O10 -> {"V": 4, "O": 10}
    """
    comp_dict = structure.composition.get_el_amt_dict()

    int_comp = {}
    for elem, amount in comp_dict.items():
        rounded = round(amount)
        if abs(amount - rounded) > tol:
            raise ValueError(
                f"Non-integer occupancy/count detected for {elem}: {amount}. "
                "This script expects ordered structures with integer atom counts."
            )
        int_comp[elem] = int(rounded)

    return int_comp


def reduced_formula_and_fu(int_comp: Dict[str, int]) -> Tuple[str, int]:
    """
    Get reduced formula and formula-unit multiplier.

    Example:
        {"V": 4, "O": 10} -> ("O5V2", 2)
    """
    from math import gcd
    from functools import reduce

    counts = list(int_comp.values())
    fu = reduce(gcd, counts)

    reduced = {
        elem: count // fu
        for elem, count in int_comp.items()
    }

    def part(elem: str, count: int) -> str:
        return elem if count == 1 else f"{elem}{count}"

    reduced_formula = "".join(
        part(elem, reduced[elem])
        for elem in sorted(reduced.keys())
    )

    return reduced_formula, fu


def exact_formula_from_counts(int_comp: Dict[str, int]) -> str:
    """
    Make formula string from exact atom counts in the source cell.

    Example:
        {"Ac": 2, "O": 3} -> "Ac2O3"
    """
    def part(elem: str, count: int) -> str:
        return elem if count == 1 else f"{elem}{count}"

    return "".join(
        part(elem, int_comp[elem])
        for elem in sorted(int_comp.keys())
    )


def make_buildcell_input(
    int_comp: Dict[str, int],
    volume: float,
    minsep: float = 1.0,
    use_symmetry: bool = False,
) -> str:
    """
    Make AIRSS buildcell input.

    Species-specific atom counts are assigned using %NUM.
    #NATOM is intentionally omitted because %NUM already fixes the exact
    number of atoms for each species.
    """
    species = sorted(int_comp.keys())

    species_line = ",".join(
        f"{elem}%NUM={int_comp[elem]}"
        for elem in species
    )

    lines = [
        f"#SPECIES={species_line}",
        f"#VOLUME={volume:.6f}",
        f"#MINSEP={minsep:.6f}",
    ]

    if use_symmetry:
        lines.append("#SYMMOPS=1-48")
    else:
        lines.append("#SYMMOPS=1")

    lines.append("")
    return "\n".join(lines)


def run_buildcell(buildcell_input: str, buildcell_cmd: str) -> str:
    """
    Run AIRSS buildcell and return generated CASTEP .cell text.
    """
    try:
        result = subprocess.run(
            [buildcell_cmd],
            input=buildcell_input,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Could not find buildcell executable: {buildcell_cmd}. "
            "Please check the path or provide it using --buildcell-cmd."
        )
    except PermissionError:
        raise RuntimeError(
            f"Permission denied when trying to execute buildcell: {buildcell_cmd}. "
            "Check that this is the compiled executable file, not a directory, "
            "and run: chmod +x <buildcell_path>"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "buildcell failed.\n"
            f"STDOUT:\n{e.stdout}\n\n"
            f"STDERR:\n{e.stderr}\n\n"
            f"Input:\n{buildcell_input}"
        )

    return result.stdout


def safe_stem(path: Path) -> str:
    """
    Make a safe directory/file stem.
    """
    return path.stem.replace(" ", "_").replace("/", "_").replace("\\", "_")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate AIRSS random .cell structures for each source CIF using the same "
            "composition and same exact atom counts as the source CIF."
        )
    )

    parser.add_argument(
        "--input",
        default="mp_cifs/",
        help="Input CIF directory or zip file containing CIF files.",
    )
    parser.add_argument(
        "--outdir",
        default="AIRSS_gen/",
        help="Output directory.",
    )
    parser.add_argument(
        "--buildcell-cmd",
        default="./airss/src/buildcell/src/buildcell",
        help="Path to compiled AIRSS buildcell executable.",
    )
    parser.add_argument(
        "--nstruct-per-cif",
        type=int,
        default=100,
        help="Number of AIRSS random structures to generate per input CIF.",
    )
    parser.add_argument(
        "--minsep",
        type=float,
        default=1.0,
        help="Default minimum interatomic distance in Angstrom. Default: 1.0",
    )
    parser.add_argument(
        "--volume-scale",
        type=float,
        default=1.0,
        help="Scale source CIF volume by this factor. Default: 1.0",
    )
    parser.add_argument(
        "--symmetry",
        action="store_true",
        help="Use #SYMMOPS=1-48. Default is #SYMMOPS=1.",
    )
    parser.add_argument(
        "--max-cifs",
        type=int,
        default=None,
        help="Maximum number of CIF files to process. Default: all.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    buildcell_cmd = Path(args.buildcell_cmd)
    if not buildcell_cmd.exists():
        raise FileNotFoundError(
            f"buildcell executable does not exist: {buildcell_cmd}\n"
            "Please compile AIRSS buildcell first and provide the correct path."
        )

    if buildcell_cmd.is_dir():
        raise IsADirectoryError(
            f"buildcell path points to a directory, not an executable: {buildcell_cmd}\n"
            "You probably need something like ./airss/src/buildcell/src/buildcell"
        )

    metadata = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cif_root = extract_if_zip(input_path, tmpdir)

        cif_files = find_cif_files(cif_root)
        if args.max_cifs is not None:
            cif_files = cif_files[: args.max_cifs]

        print(f"Found {len(cif_files)} CIF files.")

        for cif_index, cif_path in enumerate(cif_files, start=1):
            print(f"\n[{cif_index}/{len(cif_files)}] Processing {cif_path}")

            try:
                structure = Structure.from_file(cif_path)
                int_comp = integer_composition_from_structure(structure)
            except Exception as e:
                print(f"Failed to read {cif_path}: {e}")
                continue

            reduced_formula, fu = reduced_formula_and_fu(int_comp)
            exact_formula = exact_formula_from_counts(int_comp)

            source_volume = float(structure.lattice.volume)
            target_volume = source_volume * args.volume_scale

            buildcell_input = make_buildcell_input(
                int_comp=int_comp,
                volume=target_volume,
                minsep=args.minsep,
                use_symmetry=args.symmetry,
            )

            this_outdir = outdir / f"{safe_stem(cif_path)}__{exact_formula}"
            cell_outdir = this_outdir / "cell"
            cell_outdir.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(cif_path, this_outdir / f"source_{cif_path.name}")
            except Exception:
                pass

            template_path = this_outdir / "buildcell.template"
            template_path.write_text(buildcell_input, encoding="utf-8")

            print(f"  reduced formula: {reduced_formula}")
            print(f"  formula units in source cell: {fu}")
            print(f"  exact atom counts: {int_comp}")
            print(f"  exact formula in source cell: {exact_formula}")
            print(f"  source volume: {source_volume:.3f} Å^3")
            print(f"  target volume: {target_volume:.3f} Å^3")
            print(f"  minsep: {args.minsep:.3f} Å")
            print(f"  symmetry: {'#SYMMOPS=1-48' if args.symmetry else '#SYMMOPS=1'}")
            print("  buildcell input:")
            print(buildcell_input)
            print(f"  output: {this_outdir}")

            n_requested = args.nstruct_per_cif
            n_cell_generated = 0
            n_generation_failed = 0

            for i in range(1, n_requested + 1):
                try:
                    cell_text = run_buildcell(
                        buildcell_input,
                        buildcell_cmd=str(buildcell_cmd),
                    )
                except Exception as e:
                    print(f"  buildcell failed at structure {i}: {e}")
                    n_generation_failed += 1
                    continue

                cell_path = cell_outdir / f"{safe_stem(cif_path)}_airss_{i:06d}.cell"
                cell_path.write_text(cell_text, encoding="utf-8")
                n_cell_generated += 1

                if i % 50 == 0 or i == n_requested:
                    print(
                        f"  progress {i}/{n_requested}: "
                        f"cell_generated={n_cell_generated}, "
                        f"failed={n_generation_failed}"
                    )

            metadata.append(
                {
                    "source_cif": str(cif_path),
                    "output_dir": str(this_outdir),
                    "cell_output_dir": str(cell_outdir),
                    "reduced_formula": reduced_formula,
                    "formula_units_in_source_cell": fu,
                    "exact_formula_in_source_cell": exact_formula,
                    "atom_counts": int_comp,
                    "source_volume_A3": source_volume,
                    "target_volume_A3": target_volume,
                    "minsep_A": args.minsep,
                    "symmetry": "1-48" if args.symmetry else "1",
                    "n_requested": n_requested,
                    "n_cell_generated": n_cell_generated,
                    "n_generation_failed": n_generation_failed,
                    "buildcell_template": str(template_path),
                    "buildcell_input": buildcell_input,
                }
            )

            print(f"  generated .cell files: {n_cell_generated}/{n_requested}")
            print(f"  generation failures: {n_generation_failed}/{n_requested}")

    metadata_path = outdir / "airss_generation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone. Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
import os
import shutil
from pathlib import Path

import numpy as np

from sasa_lammps.constants import (
    FN_ETOT,
    RADII_MAP,
    FN_TRAJ,
    FN_THERMOLOG,
    FN_SPEC,
    FN_IN_PRE,
    FN_IN_TEMPLATE
)
from sasa_lammps.resource_loader import resources


def check_files(path: str) -> None:
    """
    Check if all the relevant files are present to exec LAMMPS
    and copy LAMMPS input file templates.
    """
    fs = os.listdir(path)

    to_rm = [FN_ETOT, FN_TRAJ, FN_THERMOLOG, FN_SPEC]
    for f in to_rm:
        if f in fs:
            print(f"{f} file already exists. Will be overwritten...")
            os.remove(Path(path) / f)

    # copy the LAMMPS input template to the working dir
    shutil.copy(resources / FN_IN_PRE, Path(path) / FN_IN_PRE)
    shutil.copy(resources / FN_IN_TEMPLATE, Path(path) / FN_IN_TEMPLATE)

    return 0


def count_atoms_in_macromol(data_file: str) -> int:
    """Count number of atoms in macro molecule file"""
    with open(data_file, "r") as f:
        for line in f:
            if "atoms" in line:
                atom_number = int(line.split()[0])
                break

    return atom_number

def count_atoms_in_mol(mol_file: str) -> int:
    """Count number of atoms in molecule file"""
    with open(mol_file, "r") as f:
        for line in f:
            if "atoms" in line:
                N = int(line.split()[0])
                break

    return N

# method faulty. Needs a path
def write_params_file(string: str, path: str, file_name: str) -> None:
    """Write string to a file. Needed to create the include files for LAMMPS to read"""
    with open(Path(path) / file_name, "w") as f:
        f.write(f"{string}\n")

    return 0


def read_last_two(path: str, file_name: str) -> list[float, float]:
    """Read last to lines of file, convert to float and return the read values"""
    with open(Path(path) / file_name, "r") as f:
        lines = f.readlines()[-2:]
    l1 = float(lines[0])
    l2 = float(lines[1])

    return l1, l2


def _get_vdw_radius(element):
    """
    Get VdW radius for element (in Angstroms).

    Uses standard VdW radii from authoritative literature sources.
    Values are hierarchically selected from:
    1. Bondi, A. (1964). "van der Waals Volumes and Radii".
       J. Phys. Chem. 68(3), 441-451. DOI: 10.1021/j100785a001
    2. Batsanov, S.S. (2001). "Van der Waals radii of elements".
       Inorg. Mater. 37(9), 871-885.
    3. Alvarez, S. (2013). "A cartography of the van der Waals territories".
       Dalton Trans. 42(24), 8617-8636. DOI: 10.1039/C3DT50599E

    Priority: Bondi > Batsanov > Alvarez for maximum compatibility.

    Parameters
    ----------
    element : str
        Element symbol (case-insensitive, digits removed)

    Returns
    -------
    float
        Van der Waals radius in Angstroms
    """
    # Clean element symbol (remove digits, make uppercase)
    clean_element = ''.join(c for c in element if c.isalpha()).capitalize()

    return RADII_MAP.get(clean_element, 1.70)  # Default to carbon radius


def parse_xyz_file(xyz_file_path):
    """
    Parse XYZ file to extract coordinates and atom types.

    Parameters
    ----------
    xyz_file_path : str
        Path to the XYZ file

    Returns
    -------
    coords : numpy.ndarray
        (N, 3) array of atomic coordinates in Angstroms
    radii : numpy.ndarray
        (N,) array of VdW radii for each atom
    """
    with open(xyz_file_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("Empty XYZ file")

    try:
        n_atoms = int(lines[0].strip())
    except (ValueError, IndexError) as e:
        raise ValueError(f"Could not read atom count from first line: {e}")

    if len(lines) < 2:
        raise ValueError("XYZ file must have at least 2 lines (atom count + comment)")

    coords = []
    elements = []

    for i in range(2, 2 + n_atoms):  # Skip header lines
        if i >= len(lines):
            raise ValueError(f"XYZ file claims {n_atoms} atoms but only has {i-2} atom lines")

        parts = lines[i].strip().split()
        if len(parts) < 4:
            raise ValueError(f"Line {i+1}: Expected at least 4 fields (element x y z), got {len(parts)}")

        element = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError as e:
            raise ValueError(f"Line {i+1}: Could not parse coordinates: {e}")

        elements.append(element)
        coords.append([x, y, z])

    coords = np.array(coords, dtype=np.float32)
    radii = np.array([_get_vdw_radius(elem) for elem in elements], dtype=np.float32)

    return coords, radii

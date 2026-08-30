"""
SASA computation module using C extension.

This module provides solvent accessible surface area calculations
using Monte Carlo sampling on atomic surfaces.
"""

import numpy as np
from pathlib import Path
import sasa_ext

from ovito.io import import_file
from ovito.data import NearestNeighborFinder

from sasa_lammps.constants import SAS_SEED, FN_SASA_XYZ
from sasa_lammps.utils import parse_xyz_file


def neighbor_finder(path, data_file, sasa_positions):
    """
    Compute informations on the nearest neighbors of every SAS point, i.e. which
    atom of the macromolecule is closest to the SAS point.

    Parameters
    ----------
    path : str
        Path to xyz_file and where to export files to
    data_file: str
        Name of the LAMMPS data file of the macromolecule
    sasa_positions : numpy.array
        Coordinates of the points on the SAS

    Returns
    -------
    neighbors: dict{"id", "pos", "res", "dist"}
        Dictionary of informations on nearest neighbors:
        neighbors["id"]: Particle ID of the nearest neighbor
        neighbors["pos"]: Position of the nearest neighbor
        neighbors["res"]: Molecule ID of the nearest neighbor, which can be used for residue identification
        neighbors["dist"]: Distance of the nearest neighbor

    """

    pipeline = import_file(Path(path) / data_file)
    data = pipeline.compute()

    finder = NearestNeighborFinder(1, data)  # 1st nearest neighbor
    neighbors = {"id": [], "pos": [], "res": [], "dist": []}
    for pos in sasa_positions:
        for neigh in finder.find_at(pos):
            neighbors["id"].append(data.particles["Particle Identifier"][neigh.index])
            neighbors["pos"].append(data.particles["Position"][neigh.index])
            neighbors["res"].append(data.particles["Molecule Identifier"][neigh.index])
            neighbors["dist"].append(neigh.distance)

    return neighbors

def rotate_probe(path, data_file, sasa_positions, neighbors):
    """
    Rotate the probe molecule on the SAS. Finds the nearest SAS point for each atom
    of the macromolecule. The rotation is then done with respect to the center-of-mass
    of the macromolecule.

    Parameters
    ----------
    path : str
        Path to xyz_file and where to export files to
    data_file: str
        Name of the LAMMPS data file of the macromolecule
    sasa_positions : numpy.array
        Coordinates of the points on the SAS
    neighbors : dict
        Dictionary of neighbor list informations.
        Output of the neighbor_finder() method

    Returns
    -------
    rot_export: numpy.ndarray
        Array of shape (N, 4) where N is the number of points on the SAS.
        rot_export[N, 0]: rotation angle
        rot_export[N, 1:]: x, y, z coordinates of the respective rotation vector

    """

    pipeline = import_file(Path(path) / data_file)

    # calculate the center-of-mass of the macromolecule
    data = pipeline.compute()
    data_positions = data.particles.positions
    data_masses = data.particles.masses
    com = np.sum([m * p for m, p in zip(data_masses, data_positions)], axis=0) / np.sum(
        data_masses
    )

    # get the indices of the particle container according to the IDs
    ordering = np.argsort(data.particles.identifiers)
    indices = ordering[
        np.searchsorted(data.particles.identifiers, neighbors["id"], sorter=ordering)
    ]

    # calculate rotation angles and vectors to parse them to LAMMPS
    rot_export = []
    for index, sasa_pos in zip(indices, sasa_positions):
        curr_id = neighbors["id"][index]
        com_vec = data_positions[curr_id] - com
        sas_vec = sasa_pos - data_positions[curr_id]

        angle = np.rad2deg(
            np.arccos(
                np.dot(com_vec, sas_vec)
                / (np.linalg.norm(com_vec) * np.linalg.norm(sas_vec))
            )
        )

        vector = np.cross(com_vec, sas_vec)
        rot_export.append(np.insert(vector, 0, angle))

    return rot_export

def compute_sasa_from_xyz(xyz_file_path, srad=1.4, samples=500, points=True):
    """
    Compute SASA from XYZ file using Monte Carlo sampling.

    Parameters
    ----------
    xyz_file_path : str
        Path to XYZ coordinate file
    srad : float
        Probe radius (solvent radius)
    samples : int
        Number of Monte Carlo sample points per atom
    points : bool
        Whether to return surface point coordinates

    Returns
    -------
    total_sasa : float
        Total solvent accessible surface area
    surface_points : numpy.ndarray or None
        (N, 3) array of surface point coordinates if points=True, else None
    """

    # Parse coordinates and radii from XYZ file
    coords, radii = parse_xyz_file(xyz_file_path)

    # Compute SASA using C extension
    # Use fixed seed for reproducible results
    total_sasa, surface_points = sasa_ext.compute_sasa(
        coords, radii,
        probe_radius=srad,
        n_samples=samples,
        seed=SAS_SEED
    )

    if points:
        return total_sasa, surface_points.tolist()  # Convert to list for compatibility
    else:
        return total_sasa, None

def create_sasa_xyz(path, xyz_file, srad, samples):
    """
    Create van der Waals surface points for molecular analysis.

    Parameters
    ----------
    path : str
        Path to xyz_file and where to export files to
    xyz_file : str
        Name of the xyz-file to use
    srad : float
        Probe radius: Effectively a scaling factor for the vdW radii
    samples : int
        Maximum points on the atomic vdW sphere to generate per atom

    Returns
    -------
    sasa_points : numpy.ndarray
        (N, 3) Array of coordinates on the SAS.
        N is loosely determined by 'samples' argument.
    """
    xyz_file_path = Path(path) / xyz_file

    # Compute SASA surface points
    _, sasa_points = compute_sasa_from_xyz(xyz_file_path, srad=srad, samples=samples, points=True)

    # Convert to numpy array
    sasa_points = np.array(sasa_points)

    # Export points in expected format
    export_points = np.array(sasa_points, dtype=str)
    export_points = np.insert(export_points, 0, "He", axis=1)

    header = f"{len(export_points)}\n "
    np.savetxt(
        Path(path) / FN_SASA_XYZ,
        export_points,
        header=header,
        comments="",
        fmt="%s",
    )

    return sasa_points

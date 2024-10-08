"""
Package to execute instances of LAMMPS to perform probe analysis of the
solvent-accessible-surface-area (SASA).

...Instead of calling LAMMPS repeatedly one could also write a loop in the 
LAMMPS input instead. This would probably avoid the huge overhead of the LAMMPS
initialization, which probably takes the most amount of computational time right now. 
"""

import os
import subprocess
import numpy as np
import tqdm
import signal

from multiprocessing import Pool

from sasa_lammps.gro2lammps import * 

from sasa_lammps.helper import (
    _check_files,
    _count_atoms_in_mol,
    _count_atoms_in_macromol,
    _write_params_file,
    _read_last_two,
)
from sasa_lammps.conversion import (
    _rotate_probe,
    _convert_data_file,
    _create_sasa_xyz,
    _neighbor_finder,
)

from sasa_lammps.postprocessing import *

from sasa_lammps.constants import *

class Sasa:
    def __init__(
        self,
        gro_file,
        data_file,
        mol_file,
        ff_str,
        dump_str,
        lammps_exe,
        n_procs,
        srad,
        samples,
        path,
    ):
        self.gro_file = gro_file
        self.data_file = data_file
        self.mol_file = mol_file
        self.ff_str = ff_str
        self.dump_str = dump_str
        self.lammps_exe = lammps_exe
        self.n_procs = n_procs
        self.srad = srad
        self.samples = samples
        self.path = path
        self.parent_pid = os.getpid()   # pid of the parent process: needed to kill in case of an exception

    def _process(self):
        # generate lammps data file
        gro2lammps(self.path, ELEM_LIBRARY).convert(self.gro_file, self.data_file)

        # remove existing files and copy input templates...
        _check_files(self.path)

        # write ff_str and dump_str to files for LAMMPS to read in
        _write_params_file(self.ff_str, FF_PARAMS)
        _write_params_file(self.dump_str, DUMP_COM)

        # get the energies for the isolated macro- and probe molecule, respectively
        e_mol, e_prob = self._pre_calc()

        # convert data file
        self.xyz_file = _convert_data_file(self.path, self.data_file)

        # create sasa position file
        self.sasa_positions = _create_sasa_xyz(
            self.path, self.xyz_file, self.srad, self.samples
        )
        n_probes = len(self.sasa_positions)  # need this only to get the total num of iterations

        # build neigbor list
        self.neighbors = _neighbor_finder(
            self.path, self.data_file, self.sasa_positions
        )

        # execute
        self._exec_lammps_iterations(n_probes, e_mol, e_prob)

        # postprocessing
        ## atom analysis
        neighbor = neighbor_analysis(self.path, SPEC, self.gro_file)
        result = atom_analysis(self.path, SPEC, neighbor)
        atom_analysis_plot(self.path, neighbor, result)

        ## residue analysis
        residuelist(self.path, self.gro_file)
        result = residue_analysis(self.path, SPEC, RESIDUELIST)
        residue_analysis_plot(self.path, result)

        return 0

    def _pre_calc(self) -> tuple[float, float]:
        """
        Do two pre-runs in LAMMPS: One of the isolated macromolecule and one of the isolated probe molecule

        Returns
        -------
        e_mol : float
            Energy of the macro molecule in kcal/mole
        e_prob : float
            Energy of the probe molecule in kcal/mole

        """

        self._run_lmp(
            IN_PRE,
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        )

        e_mol, e_prob = _read_last_two(self.path, "etot")

        return e_mol, e_prob

    def _exec_lammps_iterations(self, n_probes: int, e_mol: float, e_prob: float) -> None:
        """Execute LAMMPS singlepoints on SASA coordinates using a N-atomic probe"""

        # Count atoms in macro molecule
        atom_number = _count_atoms_in_macromol(os.path.join(self.path, self.data_file))

        # create final output file header and write to spec.xyz
        header = f"{n_probes}\natom\tx\ty\tz\tres\tetot/eV\teint/eV\n"
        with open(os.path.join(self.path, SPEC), "w") as f:
            f.write(header)

        # rotate the probe molecule for n-atomic probes (n > 1)
        if _count_atoms_in_mol(os.path.join(self.path, self.mol_file)) > 1:
            self.rotations = _rotate_probe(
                self.path, self.data_file, self.sasa_positions, self.neighbors
            )
        else:
            self.rotations = np.zeros((n_probes, 4))
            # add some direction otherwise LAMMPS raises an error because of the zero vector
            self.rotations[:, 1] += 1.000

        # create iterable for multiprocessing.Pool.starmap()
        iters = [self.sasa_positions, self.neighbors["res"], self.rotations]
        run_iterable = [
            [IN_TEMPLATE, pos, rot, atom_number, res, e_mol, e_prob]
            for pos, res, rot in zip(*iters)
        ]

        # modify the SIGINT handler to exit Pool gently. For more infos see:
        # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        with Pool(processes=self.n_procs) as pool:
            signal.signal(signal.SIGINT, original_sigint_handler)
            try:
                pool.starmap(
                    self._run_lmp, tqdm.tqdm(run_iterable, total=n_probes), chunksize=1
                )
            except KeyboardInterrupt:
                pool.terminate()
            else:
                pool.close()
            # data procesing and output of results handled by LAMMPS

        return 0

    def _run_lmp(
        self,
        in_file: str,
        pos: list[float, float, float],
        rot: list[float, float, float, float],
        atom_number = 0,
        res = 0,
        e_mol = 0.0,
        e_prob = 0.0,
    ) -> None:
        """
        Run LAMMPS by running a subprocess. May not be the most elegant way,
        because it cannot handle LAMMPS errors and is dependent on OS etc...
        Also as of now assumes LAMMPS to be build in MPI mode, but executes
        on one MPI task only... But who uses LAMMPS in serial, anyway...

        Parameters
        ----------
        in_file : str
            Name of the LAMMPS input file
        pos : list
            x, y, z position list of the SAS positions
        rot : list
            List of rotation data:
            rot[0]: Rotation angle
            rot[1]: X-component of rotation vector
            rot[2]: Y-component of rotation vector
            rot[3]: Z-component of rotation vector
        atom_number: int
            The number of atoms in the macromolecule
        res : int
            Residue ID
        e_mol : float
            Energy of the isolated macromolecule in kcal/mole
        e_prob : float
            Energy of the isolated probemolecule in kcal/mole

        Returns
        -------
        None

        """

        cmd = f"""
            mpirun -np 1 {self.lammps_exe} -in {in_file} \
            -var DataFile {self.data_file} -var MolFile {self.mol_file} \
            -var sasaX {pos[0]:.3f} -var sasaY {pos[1]:.3f} \
            -var sasaZ {pos[2]:.3f} -var rotAng {rot[0]:.3f} \
            -var rotVecX {rot[1]:.3f} -var rotVecY {rot[2]:.3f} \
            -var rotVecZ {rot[3]:.3f} \
            -var atom_number {atom_number:d} \
            -var res {res} -var emol {e_mol:.3f} \
            -var eprob {e_prob:.3f} -var conv {KCAL_TO_EV} \ 
        """

        try:
            subprocess.run(
                cmd, shell=True, env=os.environ, check=True, capture_output=True
            )
        except subprocess.CalledProcessError as exc:
            print(exc.stdout.decode())
            os.kill(self.parent_pid, signal.SIGINT)     # not the best method maybe. but its sufficient...
        finally:
            pass

        return 0


def sasa(
    gro_file,
    data_file,
    mol_file,
    ff_str,
    dump_str,
    lammps_exe,
    n_procs=1,
    srad=1.4,
    samples=100,
    path=".",
):
    """
    Run the SASA (solvet accasible surface analysis) on a given macromolecule 
    using a given probe molecule.
    The package was designed to start from a gromacs file of the macromolecule.
    For good simulation practices the macromolecule should be pre-equilibrated in water.
    Care must be taken for N-atomic probe molecules: The package does not identify
    a plane or something in the probe molecule. It just makes sure that at every
    interaction site the probe faces the macromolecule with the same orientation.
    However, the orientation itself is purely determined by the configuration
    given in the mol_file.

    Parameters
    ----------
    gro_file : str
        Name of the gromacs file of the macromolecule
    data_file : str
        Name of the LAMMPS data file of the macromolecule
    mol_file : str
        Name of the LAMMPS mol file to use as probe of the SAS (solvent acessible surface)
    ff_str : str
        Force field parameters to provide to LAMMPS. See examples directory
        https://docs.lammps.org/pair_style.html
        https://docs.lammps.org/pair_coeff.html
        Care must be taken because currently the 'unit real' in the in.template basically restricts to only use pair_style reaxff.
    dump_str : str
        Dump command to provide to LAMMPS. See examples directory
        https://docs.lammps.org/dump.html
    lammps_exe : str
        Full path to the LAMMPS executable
    n_procs : int
        Number of LAMMPS instances to run in parallel (Default: 1)
    srad : float
        Probe radius: Effectively a scaling factor for the vdW radii
        (Default: 1.4, which is the most commonly used because its approx. the
        radius of water)
    samples : int
        Maximum points on the atomic vdW sphere to generate per atom (Default: 100)
    path : str
        Execution path (Default: .)

    Returns
    -------
    None


    """

    S = Sasa(
        gro_file,
        data_file,
        mol_file,
        ff_str,
        dump_str,
        lammps_exe,
        n_procs,
        srad,
        samples,
        path,
    )
    S._process()

    return 0


def main():
    """main function mainly for testing"""
    return 0


if __name__ == "__main__":
    main()

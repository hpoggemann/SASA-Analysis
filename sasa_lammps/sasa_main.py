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

from sasa_lammps.helper import (
    _check_files,
    _count_atoms_in_mol,
    _write_params_file,
    _read_last_two,
)
from sasa_lammps.conversion import (
    _rotate_probe,
    _convert_data_file,
    _create_sasa_xyz,
    _neighbor_finder,
)

KCAL_TO_EV = 0.04336  # convert kcal/mole to eV


class Sasa:
    def __init__(
        self,
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
        # convert data file
        self.xyz_file = _convert_data_file(self.path, self.data_file)

        # create sasa position file
        self.sasa_positions = _create_sasa_xyz(
            self.path, self.xyz_file, self.srad, self.samples
        )

        # build neigbor list
        self.neighbors = _neighbor_finder(
            self.path, self.data_file, self.sasa_positions
        )

        # execute
        self._exec_lammps_iterations()

        return 0

    def _exec_lammps_iterations(self) -> None:
        """Execute LAMMPS singlepoints on SASA coordinates using a N-atomic probe"""

        # remove existing files and copy input templates...
        _check_files(self.path)

        # write ff_str and dump_str to files for LAMMPS to read in
        _write_params_file(self.ff_str, "ff_params.dat")
        _write_params_file(self.dump_str, "dump_com.dat")

        # get the energies for the isolated macro- and probe molecule, respectively
        e_mol, e_prob = self._pre_calc()
        n_probes = len(
            self.sasa_positions
        )  # need this only to get the total num of iterations

        # create final output file header and write to spec.xyz
        header = f"{n_probes}\natom\tx\ty\tz\tres\tetot [eV]\teint [eV]\n"
        with open(os.path.join(self.path, "spec.xyz"), "w") as f:
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
            ["in.template", pos, rot, res, e_mol, e_prob]
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
            "in.pre",
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            0,
            0.0,
            0.0,
        )

        e_mol, e_prob = _read_last_two(self.path, "etot")

        return e_mol, e_prob

    def _run_lmp(
        self,
        in_file: str,
        pos: float,
        rot: list[float, float, float, float],
        res: int,
        e_mol: float,
        e_prob: float,
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
        iterat : int
            Number of iteration
        max_iterat : int
            Max Number of iterations
        pos : list
            x, y, z position list of the SAS positions
        rot : list
            List of rotation data:
            rot[0]: Rotation angle
            rot[1]: X-component of rotation vector
            rot[2]: Y-component of rotation vector
            rot[3]: Z-component of rotation vector
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
            -var rotVecZ {rot[3]:.3f} -var res {res} -var emol {e_mol:.3f} \
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
    Run the SASA analysis on a given macromolecule using a given probe molecule.
    Care must be taken for N-atomic probe molecules: The package does not identify
    a plane or something in the probe molecule. It just makes sure that at every
    interaction site the probe faces the macromolecule with the same orientation.
    However, the orientation itself is purely determined by the configuration
    given in the mol_file.

    Parameters
    ----------
    data_file : str
        Name of the LAMMPS data file of the macromolecule
    mol_file : str
        Name of the LAMMPS mol file to use as probe of the SAS
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

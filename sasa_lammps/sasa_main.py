"""
Package to execute instances of LAMMPS to perform probe analysis of the
solvent-accessible-surface-area (SASA).

...Instead of calling LAMMPS repeatedly one could also write a loop in the 
LAMMPS input instead. This would probably avoid the huge overhead of the LAMMPS
initialization, which probably takes the most amount of computational time right now. 
"""

import os
import subprocess
from pathlib import Path
import numpy as np
import tqdm
import signal

import multiprocessing

from sasa_lammps.sasa_core import (
    create_sasa_xyz,
    rotate_probe,
    neighbor_finder,
)
from sasa_lammps.lammps_manager import get_lammps_executable
from sasa_lammps.utils import (
    check_files,
    count_atoms_in_mol,
    count_atoms_in_macromol,
    write_params_file,
    read_last_two,
)
from sasa_lammps.conversion import (
    Converter,
    LammpsToXyzStrategy,
    GromacsToLammpsStrategy
)
from sasa_lammps.postprocessing import (
    residue_analysis,
    neighbor_analysis,
    residue_analysis_plot,
    residuelist,
    atom_analysis,
    atom_analysis_plot
)
from sasa_lammps.constants import (
    FN_ETOT,
    FN_FF_PARAMS,
    FN_DUMP_COM,
    FN_SPEC,
    FN_RESIDUE_LIST,
    FN_IN_PRE,
    FN_IN_TEMPLATE,
    DATA_FILE,
    KCAL_TO_EV
)

class Sasa:
    """
    Class to perform a computation of the SASA, as well as hold the information of its results.
    Several files need to be provided in the construction of this object. Then, a computation can be performed with
    ```
    sasa.compute(*args)
    ```
    and a subsequent post-processing with
    ´``
    sasa.postprocess(*args)
    ```
    """
    def __init__(
        self,
        gro_file: str,
        mol_file: str,
        ff_str: str,
        dump_str: str,
        lammps_exe: str = None,
    ):
        """Constructor for the `Sasa` object.

        Parameters
        ----------
        gro_file : str
            Name of the gromacs file of the macromolecule
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
        lammps_exe : str,
            Path to the LAMMPS executable to be used. If None, an executable is downloaded. 
        """
        self.gro_file = gro_file
        self.mol_file = mol_file
        self.ff_str = ff_str
        self.dump_str = dump_str
        self.lammps_exe = lammps_exe if lammps_exe is not None else get_lammps_executable()
        self.data_file = DATA_FILE
        
    def compute(
        self,
        n_procs: int = 1,
        srad: float = 1.4,
        samples: int = 100,
        path: str = ".", # os.getcwd() ?
    ) -> None:
        """
        Compute the solvet accessible surface analysis.

        Parameters
        ----------
        n_procs : int, optional
            Number of LAMMPS instances to run in parallel (Default: 1)
        srad : float, optional
            Probe radius: Effectively a scaling factor for the vdW radii
            (Default: 1.4, which is the most commonly used because its approx. the
            radius of water)
        samples : int, optional
            Maximum points on the atomic vdW sphere to generate per atom (Default: 100)
        path : str, optional
            Execution path (Default: .)

        Returns
        -------
        None
        """
        self.n_procs = n_procs
        self.srad = srad
        self.samples = samples
        self.path = path
        self.parent_pid = os.getpid()   # PID of the parent process: required in order to kill in case of an exception

        self._initialize()

        # here starts the computation
        self._exec_lammps_iterations(self.e_mol, self.e_prob)

        return 0
    
    def postprocess(self) -> None:
        """
        Perform postprocessing on the results.
        """
        # Todo:
        # add checks that all necessary files exist
        # provide methods to load result data, such that a postprocessing can be performed without running the computation

        # atom analysis
        neighbor = neighbor_analysis(self.path, FN_SPEC, self.gro_file)
        atom_analysis_result = atom_analysis(self.path, FN_SPEC, neighbor)
        atom_analysis_plot(self.path, neighbor, atom_analysis_result)

        # residue analysis
        residuelist(self.path, self.gro_file)
        residue_analysis_result = residue_analysis(self.path, FN_SPEC, FN_RESIDUE_LIST)
        residue_analysis_plot(self.path, residue_analysis_result)

        # one could add a config-method to define more details about the postprocessing
        # or at least allow arguments here to do so.
        # Also, the postprocessing could be refactored to be more general.

        return 0

    def _initialize(self) -> None:
        """
        Perform a number of initialization steps to prepare the actual computation.
        """
        # generate lammps data file
        Converter(GromacsToLammpsStrategy()).convert(self.path, self.gro_file)

        # remove existing files and copy input templates...
        check_files(self.path)

        # write ff_str and dump_str to files for LAMMPS to read in
        write_params_file(self.ff_str, self.path, FN_FF_PARAMS)
        write_params_file(self.dump_str, self.path, FN_DUMP_COM)

        # get the energies for the isolated macro- and probe molecule, respectively
        self.e_mol, self.e_prob = self._pre_calc()

        # convert data file
        self.xyz_file = Converter(LammpsToXyzStrategy()).convert(self.path, DATA_FILE)

        # create sasa position file
        self.sasa_positions = create_sasa_xyz(
            self.path, self.xyz_file, self.srad, self.samples
        )
        self.n_probes = len(self.sasa_positions)  # need this only to get the total num of iterations

        # build neigbor list
        self.neighbors = neighbor_finder(
            self.path, DATA_FILE, self.sasa_positions
        )

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
            FN_IN_PRE,
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        )

        e_mol, e_prob = read_last_two(self.path, FN_ETOT)

        return e_mol, e_prob

    def _exec_lammps_iterations(self, e_mol: float, e_prob: float) -> None:
        """
        Execute LAMMPS singlepoints on SASA coordinates using a N-atomic probe
        
        Parameters
        ----------
        e_mol : float
            Energy of the macro molecule in kcal/mole
        e_prob : float
            Energy of the probe molecule in kcal/mole
        
        Returns
        -------
        None
        """
        # Count atoms in macro molecule
        atom_number = count_atoms_in_macromol(Path(self.path) / DATA_FILE)

        # create final output file header and write to spec.xyz
        header = f"{self.n_probes}\natom\tx\ty\tz\tres\tetot/eV\teint/eV\n"
        with open(Path(self.path) / FN_SPEC, "w") as f:
            f.write(header)

        # rotate the probe molecule for n-atomic probes (n > 1)
        if count_atoms_in_mol(Path(self.path) / self.mol_file) > 1:
            self.rotations = rotate_probe(
                self.path, DATA_FILE, self.sasa_positions, self.neighbors
            )
        else:
            self.rotations = np.zeros((self.n_probes, 4))
            # add some direction otherwise LAMMPS raises an error because of the zero vector
            self.rotations[:, 1] += 1.000

        # create iterable for multiprocessing.Pool.starmap()
        iters = [self.sasa_positions, self.neighbors["res"], self.rotations]
        run_iterable = [
            [FN_IN_TEMPLATE, pos, rot, atom_number, res, e_mol, e_prob]
            for pos, res, rot in zip(*iters)
        ]

        # modify the SIGINT handler to exit Pool gently. For more infos see:
        # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        with multiprocessing.Pool(processes=self.n_procs) as pool:
            signal.signal(signal.SIGINT, original_sigint_handler)
            try:
                pool.starmap(
                    self._run_lmp, tqdm.tqdm(run_iterable, total=self.n_probes), chunksize=1
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
            -var DataFile {DATA_FILE} -var MolFile {self.mol_file} \
            -var sasaX {pos[0]:.3f} -var sasaY {pos[1]:.3f} \
            -var sasaZ {pos[2]:.3f} -var rotAng {rot[0]:.3f} \
            -var rotVecX {rot[1]:.3f} -var rotVecY {rot[2]:.3f} \
            -var rotVecZ {rot[3]:.3f} \
            -var atom_number {atom_number:d} \
            -var res {res} -var emol {e_mol:.3f} \
            -var eprob {e_prob:.3f} -var conv {KCAL_TO_EV}
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


def main():
    """main function mainly for testing"""
    return 0


if __name__ == "__main__":
    main()

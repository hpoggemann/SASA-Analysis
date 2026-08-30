# Overview

This package calculates the solvent-accesible surface analysis (SASA) on a given macro-molecule (protein). It places a given probe-molecule on the surface points to compute a 3D interaction energy surface. The potential energy is calculated with a ReaxFF potential that needs to be provided (see below) and uses [LAMMPS](https://www.lammps.org) as actual implementation for the energy calculations.
This package was written for the publication J. Phys. Chem. B 2025, 129, 44, 11374–11386 by Poggemann et al. (https://pubs.acs.org/doi/10.1021/acs.jpcb.5c03518). The original version of the package relied on the VMD molecular visualization program for the calculation of the solvent-accesible surface, this dependency was removed in a major updated and repaced by a custom implementation. 

# Build and Install

This package uses a custom C extension for SASA calculations and can be installed as a standard Python package using pip.

## Requirements

- Python 3.9 or higher
- A C compiler (gcc, clang)
- Python development headers

## Installation

It is recommended to use a python environment.
```bash
# Create a python environment (optional) or use an existing one
python -m venv ~/venv/sasa
# Activate your Python environment
source ~/venv/sasa/bin/activate
```

From this projects root directory install the package with all dependencies
```
pip install .
```

## Verification

After installation, you can either verify that the SASA extension is loaded correctly (tp not confuse python with the local `sasa_ext` directory, execute this from any other than the source-file directory):
```bash
cd && python -c "import sasa_ext; print('✓ SASA extension loaded successfully')"
```

or/and execute the tests in the `tests` directory. This requires the installation of the test-requirements and the install in editable mode:
```bash
pip install -e ".[test]"
pytest ./tests
```

# Usage

The package really only has one usable object `sasa_lammps.Sasa`:

```python
from sasa_lammps import Sasa
```

To see a list of usable methods, use `help(Sasa)` or see `docs/sasa_lammps.html`.

## Example

In the directory `example` you find an example on the protein lysozyme with either H or H2O2 as probe molecules. There are two files lysozyme_part.gro and lysozyme_full.gro in the folder. The first contains only a part of the protein to speed up the test (sampling 2691 SAS points), the latter contains the full protein (sampling 8130 SAS points). 
To succesfully convert the .gro file to a lammps data file you need a library file of all elements included in you system, element_library.txt. This files should contain the ElementType (H,C,O, ...), the gromacs_ParticleType (1, 6, 8, ...), the lammps_ParticleType (1,2,3, ...) and the mass (1.00, 12.0, 16.0 ...). 
You should doubble check first which gromacs_ParticleType number ovito gives the elements. It is mostly related to the atomic number, but for Mg, for example, gromacs_ParticleType is 0. So check that first! 
You can choose any lammps_ParticleType you want for each of the elements, but the numbers have to be consecutive (1,2,3, ...).
For the ReaxFF simulation with lammps the elements have to be listed in the right order in the "pair_coeff" command. Resulting in: ```pair_coeff     * * protein2013.ff H C O```. If you use the protein2013.ff force field and H has the lammps_ParticleType 1, C is 2 and O is 3. 
The following files need to be provided for LAMMPS:
- lysozyme_part.gro (.gro file)
- h.mol or h2o2.mol (.mol file)
- element_library.txt
- protein2013.ff

```python
from sasa_lammps import Sasa

# Import the gromacs file
gro_file = "lysozyme_part.gro"
# Import the molecule file (example also contains h2o2.mol)
mol_file = "h.mol"
# Path to your lammps executable (optional - will auto-download if not provided)
# lammps_exe =  "/opt/lammps-23Jun2022/src/lmp_mpi"
# Specify the force filed parameters, in this case reaxFF parameters*
ff_str = """
pair_style      reaxff NULL safezone 1.6 mincap 100 minhbonds 150
pair_coeff      * * protein2013.ff H C N O S 
fix             QEq all qeq/reax 1 0.0 10.0 1e-6 reaxff
"""
# Specify the output, if needed. Caution: This file gets very lage.
dump_str = """ """
"""
dump            traj all custom 1 traj.lmp id mol type element x y z vx vy vz q 
dump_modify     traj append yes element H C N O S 
"""
# Run sasa
sasa = Sasa(gro_file, mol_file, ff_str, dump_str, lammps_exe)
sasa.compute()
#sasa.postprocess() # post-process is currently broken. WIP
```

# Citations

1. H.-F. Poggemann et al., “Phenylalanine modification in plasma-driven biocatalysis revealed by solvent accessibility and reactive dynamics in combination with protein mass spectrometry,” The Journal of Physical Chemistry B, 2025, doi: 10.1021/acs.jpcb.5c03518. (https://pubs.acs.org/doi/10.1021/acs.jpcb.5c03518)
2. A. P. Thompson et al., “LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales,” Computer Physics Communications, vol. 271, p. 108171, Feb. 2022, doi: 10.1016/j.cpc.2021.108171. (https://doi.org/10.1016/j.cpc.2021.108171)
3. A. Stukowski, “Visualization and analysis of atomistic simulation data with OVITO–the Open Visualization Tool,” Modelling and Simulation in Materials Science and Engineering, vol. 18, no. 1, p. 15012, Dec. 2009, doi: 10.1088/0965-0393/18/1/015012. (https://iopscience.iop.org/article/10.1088/0965-0393/18/1/015012)


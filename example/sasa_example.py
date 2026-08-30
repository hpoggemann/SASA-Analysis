from sasa_lammps import Sasa

# Import the gromacs file
gro_file = "lysozyme_part.gro"
# Import the molecule file (example also contains h2o2.mol)
mol_file = "h.mol" 
# Path to you lammps executable
lammps_exe =  "/opt/lammps-23Jun2022/src/lmp_mpi"
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
sasa = Sasa(gro_file, mol_file, ff_str, dump_str)#, lammps_exe)
sasa.compute()
# sasa.postprocess() # currently broken. WIP


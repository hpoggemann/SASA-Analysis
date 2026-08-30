from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

from ovito.io import import_file, export_file
from ovito.pipeline import Pipeline
from ovito.modifiers import (
    ExpressionSelectionModifier,
    DeleteSelectedModifier,
    ComputePropertyModifier
)

from sasa_lammps.constants import DATA_FILE, FN_ELEM_LIBRARY


class ConverterStrategy(ABC):
    @abstractmethod
    def convert(self, path: str, in_file: str) -> str:
        """
        Converts the `in_file` in `path` and returns the exported file name.
        Parameters
        ----------
        path : str
            Path where in_file resised and where the output is exported to.
        in_file : str
            Name of the file to be converted
        Returns
        -------
        out_file : str
            Path to the converted, exported file.
        """
        pass


class GromacsToLammpsStrategy(ConverterStrategy):
    """Strategy implementation to convert Gromacs to LAMMPS files."""
    def convert(self, path: str, in_file: str) -> str:
        self.path = path
        self.di = pd.read_csv(Path(self.path) / FN_ELEM_LIBRARY, sep='\s+')

        pipeline = import_file(Path(self.path) / in_file)
        self._delete_solvent(in_file, pipeline)

        pipeline.modifiers.append(self._change_particleTypes)
        pipeline.modifiers.append(self._change_particleIDs)
        pipeline.modifiers.append(self._change_masses)

        # Store residue informations in the Molecule Identifier 
        pipeline.modifiers.append(
            ComputePropertyModifier(
                expressions = ('ResidueIdentifier',), 
                output_property = 'Molecule Identifier')
            )
        pipeline.compute()

        export_file(
            pipeline, 
            Path(self.path) / DATA_FILE, 
            "lammps/data",  
            atom_style="full")

        return DATA_FILE

    def _delete_solvent(self, in_file: str, pipeline: Pipeline) -> None:
        """
        Delete the solvent and ions from a dataset
        
        Parameters
        ----------
        in_file: str
            Path to the GROMACS data file
        pipeline: Pipeline
            Ovito pipeline object that acts on the dataset
        """
        # Find the place to cut
        with open(Path(self.path) /in_file, "r") as rf:
            for i, line in enumerate(rf):
                if 'SOL' in line or 'SOD' in line:
                    atmnr = i - 2
                    break
        # UPO_del_Solv - Expression selection:
        pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'ParticleIdentifier>%s'%str(atmnr)))
        #  UPO_del_Solv - Delete selected
        pipeline.modifiers.append(DeleteSelectedModifier())

    def _change_particleTypes(self, frame, data):
        """Change Particle Types"""
        types = data.particles_.particle_types_
        for gro_PT, lammps_PT in zip(self.di['gromacs_ParticleType'], self.di['lammps_ParticleType']):
            types[types == gro_PT ] = lammps_PT
    
    def _change_particleIDs(self, frame, data):
        """Change Particle IDs"""
        for i,item in enumerate(self.di['gromacs_ParticleType']):
            data.particles_.particle_types_.type_by_id_(item).id = i+1
    
    def _change_masses(self, frame, data):
        """Change Masses and Names"""
        for lammps_PT, et, mass in zip(self.di['lammps_ParticleType'], self.di['ElementType'], self.di['mass']):
            data.particles_.particle_types_.type_by_id_(lammps_PT).name = et
            data.particles_.particle_types_.type_by_id_(lammps_PT).mass = mass


class LammpsToXyzStrategy(ConverterStrategy):
    """Strategy implementation to convert LAMMPS to XYZ files."""
    def convert(self, path, in_file):
        pipeline = import_file(Path(path) / in_file)

        ### temporary solution
        xyz_file = f"{in_file.split('.')[-1]}.xyz"

        export_file(
            pipeline,
            Path(path) / xyz_file,
            "xyz",
            columns=[
                "Particle Type",
                "Position.X",
                "Position.Y",
                "Position.Z",
            ],
        )

        return xyz_file


class Converter:
    """Class to convert data formats according to the choosen `ConverterStrategy`."""
    def __init__(self, strategy: ConverterStrategy):
        self._strategy = strategy
    
    @property
    def strategy(self) -> ConverterStrategy:
        """Get the currently set strategy to convert."""
        return self._strategy
    
    def convert(self, path: str, in_file: str) -> str:
        """
        Converts the `in_file` in `path` and returns the path to exported file.
        Parameters
        ----------
        path : str
            Path where in_file resised and where the output is exported to.
        in_file : str
            Name of the file to be converted
        Returns
        -------
        out_file : str
            Path to the converted, exported file.
        """

        return self._strategy.convert(path, in_file)

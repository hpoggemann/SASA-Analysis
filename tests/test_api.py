"""
Integration tests of the exposed API.
"""

import os

import pytest
import shutil
import glob
import numpy as np
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

from sasa_lammps.sasa_main import Sasa
from sasa_lammps.constants import FN_ETOT, FN_SPEC, DATA_FILE


class TestMainAPI:
    """Test of main API methods. Uses a mock for the LAMMPS execution and therefore does not check for correct I/O."""
    ff_str = """
    pair_style      reaxff NULL safezone 1.6 mincap 100 minhbonds 150
    pair_coeff      * * protein2013.ff H C N O S 
    fix             QEq all qeq/reax 1 0.0 10.0 1e-6 reaxff
    """

    def _setup_mocks(self, mocker):
        """Helper to set up all necessary mocks for API tests."""
        mocks = {
            'create_sasa_xyz': mocker.patch('sasa_lammps.sasa_main.create_sasa_xyz', 
                                           return_value=np.array([[0.0, 0.0, 0.0]])),
            'neighbor_finder': mocker.patch('sasa_lammps.sasa_main.neighbor_finder',
                                            return_value={"id": [], "pos": [], "res": [], "dist": []}),
            'converter': mocker.patch('sasa_lammps.sasa_main.Converter'),
            'count_macromol': mocker.patch('sasa_lammps.sasa_main.count_atoms_in_macromol', 
                                           return_value=100),
            'count_mol': mocker.patch('sasa_lammps.sasa_main.count_atoms_in_mol', 
                                      return_value=1),
            'rotate_probe': mocker.patch('sasa_lammps.sasa_main.rotate_probe', 
                                         return_value=np.zeros((1, 4))),
        }
        mocks['converter'].return_value.convert.return_value = DATA_FILE
        return mocks

    def test_api_lysozyme_default(self, prepare_api_test_files, mock_for_api_test, write_file_for_api_test_mock, mocker):
        """Test of Sasa.compute() with lysozyme. """
        mock_for_api_test["mock_pool_instance"].starmap.side_effect = partial(
            write_file_for_api_test_mock,
            path = prepare_api_test_files,
            fn = FN_SPEC,
            return_type = ""
        )
        
        mock_for_api_test["mock_pre_calc"].side_effect = partial(
            write_file_for_api_test_mock,
            path=prepare_api_test_files,
            fn=FN_ETOT,
            return_type="tuple"
        )

        # Set up all mocks using helper
        self._setup_mocks(mocker)

        sasa = Sasa(
            "lysozyme_part.gro",
            "h.mol",
            self.ff_str,
            ""
        )
        
        sasa.compute(path=prepare_api_test_files)
        
        assert True # checks only that this runs without error


    def test_api_lysozyme_explicit_exe(self, prepare_api_test_files, mock_for_api_test, write_file_for_api_test_mock, mocker):
        """Test of Sasa.compute() with lysozyme and stating the LAMMPS exe explicitly."""
        mock_for_api_test["mock_pool_instance"].starmap.side_effect = partial(
            write_file_for_api_test_mock,
            path = prepare_api_test_files,
            fn = FN_SPEC,
            return_type = ""
        )
        
        mock_for_api_test["mock_pre_calc"].side_effect = partial(
            write_file_for_api_test_mock,
            path=prepare_api_test_files,
            fn=FN_ETOT,
            return_type="tuple"
        )

        # Set up all mocks using helper
        self._setup_mocks(mocker)

        sasa = Sasa(
            "lysozyme_part.gro",
            "h.mol",
            self.ff_str,
            "",
            "/some/path/to/a/hypothetical/lammps/exe"
        )
        
        sasa.compute(path=prepare_api_test_files)
        
        assert True # checks only that this runs without error

    def test_api_lysozyme_sub_path(self, prepare_api_test_files, mock_for_api_test, write_file_for_api_test_mock, mocker):
        """Test of Sasa.compute() with lysozmye from another directory than the cwd."""
        mock_for_api_test["mock_pool_instance"].starmap.side_effect = partial(
            write_file_for_api_test_mock,
            path = prepare_api_test_files,
            fn = FN_SPEC,
            return_type = ""
        )
        
        mock_for_api_test["mock_pre_calc"].side_effect = partial(
            write_file_for_api_test_mock,
            path=prepare_api_test_files,
            fn=FN_ETOT,
            return_type="tuple"
        )

        # create a sub-path and then move everything there
        sub_path = prepare_api_test_files / Path("sub-path")
        os.mkdir(sub_path)
        for f in glob.glob(str(prepare_api_test_files / "*")):
            if Path(f) != sub_path:
                shutil.move(f, sub_path / Path(f).name)

        # Set up all mocks using helper
        self._setup_mocks(mocker)

        sasa = Sasa(
            "lysozyme_part.gro",
            "h.mol",
            self.ff_str,
            ""
        )
        
        sasa.compute(path=sub_path)
        
        assert True # checks only that this runs without error

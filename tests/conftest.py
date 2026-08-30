"""
Pytest configuration and shared fixtures for SASA tests.
"""

from unittest.mock import MagicMock, patch
import pytest
import shutil
import numpy as np
import json
from pathlib import Path

# Mock ovito imports to avoid libGLX.so.0 dependency
import sys
sys.modules['ovito'] = MagicMock()
sys.modules['ovito.io'] = MagicMock()
sys.modules['ovito.data'] = MagicMock()
sys.modules['ovito.pipeline'] = MagicMock()
sys.modules['ovito.modifiers'] = MagicMock()
sys.modules['ovito.plugins'] = MagicMock()

import sasa_ext
from sasa_lammps.utils import read_last_two
from sasa_lammps.constants import FN_ETOT, FN_SPEC
from sasa_lammps.sasa_core import create_sasa_xyz, neighbor_finder
from sasa_lammps.conversion import Converter, LammpsToXyzStrategy, GromacsToLammpsStrategy


resource_path = Path(__file__).parent / "resources" 


@pytest.fixture
def reference_data():
    """Load VMD reference data if available."""
    ref_file = Path(__file__).parent / "resources" / "vmd_reference_data.json"
    if ref_file.exists():
        with open(ref_file, 'r') as f:
            return json.load(f)

    return {}

@pytest.fixture
def test_molecules():
    """Standard test molecules with known geometric properties."""
    return {
        'single_atom': {
            'coords': np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            'radii': np.array([1.5], dtype=np.float32),
            'area_factor': 4 * np.pi  # For probe_radius=0
        },
        'two_separated': {
            'coords': np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
            'radii': np.array([1.5, 1.5], dtype=np.float32),
            'behavior': 'independent'
        },
        'two_close': {
            'coords': np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
            'radii': np.array([1.5, 1.5], dtype=np.float32),
            'behavior': 'reduced'
        },
        'buried': {
            'coords': np.array([
                [0.0, 0.0, 0.0],   # Central atom
                [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
                [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]
            ], dtype=np.float32),
            'radii': np.array([1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float32),
            'behavior': 'central_buried'
        }
    }

@pytest.fixture
def xyz_files(tmp_path):
    """Standard XYZ files for testing."""
    files = {}

    # Basic test cases
    files['single_atom'] = create_xyz_file(tmp_path, "single.xyz", [('C', 0.0, 0.0, 0.0)])
    files['two_atoms'] = create_xyz_file(tmp_path, "two.xyz", [('C', 0.0, 0.0, 0.0), ('C', 5.0, 0.0, 0.0)])
    files['mixed_elements'] = create_xyz_file(tmp_path, "mixed.xyz", [
        ('C', 0.0, 0.0, 0.0), ('H', 1.0, 0.0, 0.0),
        ('N', 0.0, 1.0, 0.0), ('O', 0.0, 0.0, 1.0)
    ])

    # Performance test case
    np.random.seed(42)  # Reproducible
    large_atoms = [(np.random.choice(['C', 'N', 'O', 'H']),
                   *np.random.uniform(-10, 10, 3)) for _ in range(100)]
    files['large_molecule'] = create_xyz_file(tmp_path, "large.xyz", large_atoms)

    return files

@pytest.fixture
def vdw_radii():
    """Standard van der Waals radii for elements (in Angstroms)."""
    return {
        'H': 1.20,
        'C': 1.70,
        'N': 1.55,
        'O': 1.52,
        'S': 1.80,
        'P': 1.80,
        'F': 1.47,
        'Cl': 1.75,
        'He': 1.40,
        'Unknown': 1.70  # Default
    }

@pytest.fixture
def sasa_parameters():
    """Standard SASA parameter sets for testing."""
    return {
        'standard': {'probe_radius': 1.4, 'n_samples': 500, 'seed': 38572111},
        'small_probe': {'probe_radius': 1.0, 'n_samples': 500, 'seed': 38572111},
        'large_probe': {'probe_radius': 2.0, 'n_samples': 500, 'seed': 38572111},
        'few_samples': {'probe_radius': 1.4, 'n_samples': 100, 'seed': 38572111},
        'many_samples': {'probe_radius': 1.4, 'n_samples': 1000, 'seed': 38572111},
        'zero_probe': {'probe_radius': 0.0, 'n_samples': 500, 'seed': 38572111},
    }

@pytest.fixture
def tolerances():
    """Tolerance values for different types of test comparisons."""
    return {
        'area_relative': 0.03,       # 3% for total SASA area
        'area_absolute': 1.0,        # 1.0 absolute tolerance
        'point_count_relative': 0.09, # 9% for point counts
        'reproducibility': 1e-10,    # Exact for same seed
        'convergence': 0.02,         # 2% for sample convergence
        'geometric': 0.01            # 1% for geometric relationships
    }

### TestMainAPI
@pytest.fixture
def prepare_api_test_files(tmp_path: Path):
    """Creates a temporary test directory and deletes it after the test."""
    temp_path = tmp_path / "api_test_lysozyme"
    resource_path = Path(__file__).parent / "resources" 
    required_files = [
        "element_library.txt",
        "h.mol",
        "h2o2.mol",
        "lysozyme_full.gro",
        "lysozyme_part.gro",
        "protein2013.ff"
        ]
    
    temp_path.mkdir()
    for f in required_files:
        shutil.copyfile(resource_path / f, temp_path / f)
    # copy relevant files from ./resources
    
    yield temp_path

    # teardown: Delete the results
    shutil.rmtree(temp_path)

@pytest.fixture
def write_file_for_api_test_mock():
    """Writes files for API tests that in prod would be written by the mocked methods."""
    def _write_file(*args, **kwargs):
        path = kwargs["path"]
        fn = kwargs["fn"]
        return_type = kwargs["return_type"]
        shutil.copyfile(resource_path / fn, path / fn)
        if return_type == "tuple":
            return read_last_two(path, FN_ETOT)
        else:
            return 0
    return _write_file

@pytest.fixture
def mock_for_api_test():
    """Creates mock for multiprocessing.Pool and sasa._pre_calc()."""
    # simply mocking _run_lmp does not work. Raises a PicklingError as described here: https://github.com/python/cpython/issues/100090
    # I guess I cannot fix this at the moment

    with patch("sasa_lammps.sasa_main.multiprocessing.Pool") as mock_pool, \
        patch("sasa_lammps.sasa_main.Sasa._pre_calc") as mock_pre_calc:

        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance

        mock_pool_instance = mock_pool_instance
        mock_pre_calc = mock_pre_calc

        yield {"mock_pool_instance": mock_pool_instance, "mock_pre_calc": mock_pre_calc}

### helper
def create_xyz_file(tmp_path, filename, atoms):
    """Create an XYZ file from atom data.

    Args:
        tmp_path: pytest tmp_path fixture
        filename: name of file to create
        atoms: list of (element, x, y, z) tuples
    """
    filepath = tmp_path / filename
    content = f"{len(atoms)}\nTest molecule\n"
    for elem, x, y, z in atoms:
        content += f"{elem} {x:.6f} {y:.6f} {z:.6f}\n"
    filepath.write_text(content)
    return filepath

def load_xyz_file(filepath, vdw_radii=None):
    """Load coordinates and radii from XYZ file.

    Args:
        filepath: path to XYZ file
        vdw_radii: dict of element -> radius mapping

    Returns:
        (coords, radii) as numpy arrays
    """
    if vdw_radii is None:
        vdw_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
            'S': 1.80, 'P': 1.80, 'F': 1.47, 'Cl': 1.75,
            'He': 1.40
        }

    with open(filepath, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    coords = []
    radii = []

    for i in range(2, 2 + n_atoms):
        parts = lines[i].strip().split()
        element = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

        coords.append([x, y, z])
        radii.append(vdw_radii.get(element, 1.70))

    return np.array(coords, dtype=np.float32), np.array(radii, dtype=np.float32)

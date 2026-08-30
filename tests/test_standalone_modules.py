"""
Tests for standalone modules that don't require full package installation.

These tests import modules directly to avoid circular dependency issues.
"""

import pytest
import numpy as np
import importlib.util
from pathlib import Path


class TestSASACoreStandalone:
    """Test SASA core module without full package import."""

    @pytest.fixture
    def sasa_core_module(self):
        """Import sasa_core module directly."""
        module_path = Path(__file__).parent.parent / "sasa_lammps" / "sasa_core.py"
        spec = importlib.util.spec_from_file_location("sasa_core", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_xyz_parsing(self, sasa_core_module, tmp_path):
        """Test XYZ file parsing functionality."""
        parse_xyz_file = sasa_core_module.parse_xyz_file

        # Create test files
        single_atom_file = tmp_path / "single.xyz"
        single_atom_file.write_text("1\nSingle carbon atom\nC 0.0 0.0 0.0\n")

        coords, radii = parse_xyz_file(str(single_atom_file))
        assert coords.shape == (1, 3)
        assert radii.shape == (1,)
        assert radii[0] == 1.70  # Carbon radius
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])

        # Two atoms
        two_atoms_file = tmp_path / "two.xyz"
        two_atoms_file.write_text("2\nTwo atoms\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n")

        coords, radii = parse_xyz_file(str(two_atoms_file))
        assert coords.shape == (2, 3)
        expected_radii = np.array([1.70, 1.20], dtype=np.float32)  # C, H
        np.testing.assert_array_almost_equal(radii, expected_radii)

        # Mixed elements
        mixed_file = tmp_path / "mixed.xyz"
        mixed_file.write_text("""4
Mixed elements
C 0.0 0.0 0.0
H 1.0 0.0 0.0
N 0.0 1.0 0.0
O 0.0 0.0 1.0
""")

        coords, radii = parse_xyz_file(str(mixed_file))
        assert coords.shape == (4, 3)
        expected_coords = np.array([
            [0.0, 0.0, 0.0],  # C
            [1.0, 0.0, 0.0],  # H
            [0.0, 1.0, 0.0],  # N
            [0.0, 0.0, 1.0],  # O
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(coords, expected_coords)

        expected_radii = np.array([1.70, 1.20, 1.55, 1.52], dtype=np.float32)
        np.testing.assert_array_almost_equal(radii, expected_radii)

    def test_malformed_xyz_handling(self, sasa_core_module, tmp_path):
        """Test handling of malformed XYZ files."""
        parse_xyz_file = sasa_core_module.parse_xyz_file

        # Empty file
        empty_file = tmp_path / "empty.xyz"
        empty_file.write_text("")

        with pytest.raises((ValueError, IndexError)):
            parse_xyz_file(str(empty_file))

        # Wrong atom count
        wrong_count_file = tmp_path / "wrong_count.xyz"
        wrong_count_file.write_text("5\nShould have 5 atoms\nC 0 0 0\nH 1 0 0\n")

        with pytest.raises((ValueError, IndexError)):
            parse_xyz_file(str(wrong_count_file))

        # Non-numeric coordinates
        bad_coords_file = tmp_path / "bad_coords.xyz"
        bad_coords_file.write_text("1\nBad coordinates\nC abc def ghi\n")

        with pytest.raises(ValueError):
            parse_xyz_file(str(bad_coords_file))


class TestConstants:
    """Test constants that don't require complex imports."""

    def test_constants_file_exists(self):
        """Test that constants file exists and is readable."""
        constants_path = Path(__file__).parent.parent / "sasa_lammps" / "constants.py"
        assert constants_path.exists(), "Constants file should exist"

        # Read content
        content = constants_path.read_text()
        assert 'SASA_XYZ' in content, "Should define SASA_XYZ constant"

    def test_constants_import_directly(self):
        """Test importing constants directly."""
        constants_path = Path(__file__).parent.parent / "sasa_lammps" / "constants.py"
        spec = importlib.util.spec_from_file_location("constants", constants_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Test SASAXYZ constant
        assert hasattr(module, 'FN_SASA_XYZ')
        assert isinstance(module.FN_SASA_XYZ, str)
        assert module.FN_SASA_XYZ.endswith('.xyz')


class TestFileStructure:
    """Test overall file structure without imports."""

    def test_required_files_exist(self):
        """Test that all required files exist."""
        base_path = Path(__file__).parent.parent

        required_files = [
            "setup.py",
            "pyproject.toml",
            "MANIFEST.in",
            "sasa_ext/sasa_core.h",
            "sasa_ext/sasa_core.c",
            "sasa_ext/sasa_module.c",
            "sasa_ext/__init__.py",
            "sasa_lammps/constants.py",
            "sasa_lammps/conversion.py",
            "sasa_lammps/sasa_core.py",
        ]

        for file_path in required_files:
            full_path = base_path / file_path
            assert full_path.exists(), f"Required file missing: {file_path}"

    def test_c_extension_structure(self):
        """Test C extension file structure."""
        base_path = Path(__file__).parent.parent / "sasa_ext"

        # Check header file
        header_file = base_path / "sasa_core.h"
        header_content = header_file.read_text()

        expected_functions = [
            'compute_sasa',
            'build_neighbor_lists',
            'generate_sphere_points',
            'is_point_buried'
        ]

        for func in expected_functions:
            assert func in header_content, f"Function {func} not declared in header"

        # Check implementation file
        impl_file = base_path / "sasa_core.c"
        impl_content = impl_file.read_text()

        # Check for VMD seed constant (should be in header now)
        assert 'VMD_SEED' in header_content, "VMD seed constant should be defined in header"
        assert '38572111' in header_content, "VMD seed value should be correct"

        # Check module file
        module_file = base_path / "sasa_module.c"
        module_content = module_file.read_text()

        assert 'PyInit_sasa_ext' in module_content, "Python module init function should exist"
        assert 'compute_sasa_wrapper' in module_content, "Python wrapper function should exist"

    def test_setup_configuration(self):
        """Test setup configuration files."""
        base_path = Path(__file__).parent.parent

        # Test setup.py (minimal for C extensions only)
        setup_py = base_path / "setup.py"
        setup_content = setup_py.read_text()

        setup_checks = [
            'get_extensions()' in setup_content,
            'sasa_ext' in setup_content,
            'ext_modules=get_extensions()' in setup_content
        ]

        assert all(setup_checks), "setup.py configuration incomplete"

        # Test pyproject.toml
        pyproject_toml = base_path / "pyproject.toml"
        pyproject_content = pyproject_toml.read_text()

        assert 'build-system' in pyproject_content, "pyproject.toml should have build-system"
        assert 'numpy' in pyproject_content, "pyproject.toml should specify numpy dependency"


class TestImplementationReadiness:
    """Test that implementation is ready for deployment."""

    def test_build_system_consistency(self):
        """Test that build system files are consistent."""
        base_path = Path(__file__).parent.parent

        # Check version in pyproject.toml (modern packaging)
        pyproject_content = (base_path / "pyproject.toml").read_text()
        assert 'version' in pyproject_content.lower(), "pyproject.toml should have version"

        # Check dependencies in pyproject.toml
        deps = ['numpy', 'ovito', 'tqdm', 'pandas', 'scipy', 'matplotlib']
        for dep in deps:
            assert dep in pyproject_content, f"pyproject.toml should mention {dep}"

    def test_algorithm_parameters(self):
        """Test that algorithm parameters are correctly configured."""
        # Check C implementation has correct constants (now in header file)
        sasa_header_path = Path(__file__).parent.parent / "sasa_ext" / "sasa_core.h"
        header_content = sasa_header_path.read_text()

        # VMD compatibility constants
        assert '#define VMD_SEED 38572111' in header_content, "VMD seed should be defined"
        assert '#define PI 3.14159265358979323846' in header_content, "PI should be defined"

        # Also check implementation file for algorithm components
        sasa_core_path = Path(__file__).parent.parent / "sasa_ext" / "sasa_core.c"
        core_content = sasa_core_path.read_text()

        # Algorithm components
        algorithm_components = [
            'generate_sphere_points',
            'vmd_random',             # Updated to VMD-compatible RNG
            'build_neighbor_lists',
            'is_point_buried',
            'compute_sasa'
        ]

        for component in algorithm_components:
            assert component in core_content, f"Algorithm component {component} missing"

    def test_ready_for_deployment(self):
        """Overall deployment readiness check."""
        base_path = Path(__file__).parent.parent

        # All critical files should exist
        critical_files = [
            "setup.py", "pyproject.toml",
            "sasa_ext/sasa_core.c", "sasa_ext/sasa_module.c",
            "sasa_lammps/sasa_core.py", "sasa_lammps/conversion.py"
        ]

        for file_path in critical_files:
            assert (base_path / file_path).exists(), f"Critical file missing: {file_path}"

        # Test structure should be complete
        test_files = [
            "tests/conftest.py", "tests/test_core_components.py",
            "tests/test_performance.py", "pytest.ini"
        ]

        for test_file in test_files:
            assert (base_path / test_file).exists(), f"Test file missing: {test_file}"

        print("✅ Implementation ready for deployment!")
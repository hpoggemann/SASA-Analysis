"""
Integration tests against VMD reference data.

Tests the C extension implementation against VMD reference calculations
to ensure accuracy and compatibility.
"""

import pytest
import numpy as np
from pathlib import Path
import sasa_ext
from sasa_lammps.utils import parse_xyz_file


class TestVMDReferenceComparison:
    """Test against VMD reference data. These tests have a very high priority."""

    @pytest.mark.parametrize("test_case", [
        # Original lysozyme tests
        "lysozyme_small",
        "lysozyme_medium",
        "lysozyme_small_probe",
        "lysozyme_large_probe",
        # Diverse protein tests
        "ubiquitin_standard",
        "ubiquitin_high_samples",
        "ubiquitin_small_probe",
        "ubiquitin_large_probe",
        "ribonuclease_standard",
        "ribonuclease_high_samples",
        "ribonuclease_small_probe",
        "ribonuclease_large_probe",
        "crambin_standard",
        "crambin_high_samples",
        "crambin_small_probe",
        "crambin_large_probe",
        "bpti_standard",
        "bpti_high_samples",
        "bpti_small_probe",
        "bpti_large_probe",
        "myoglobin_standard",
        "myoglobin_high_samples",
        "myoglobin_small_probe",
        "myoglobin_large_probe"
    ])
    def test_against_vmd_reference_data(self, test_case, reference_data, tolerances):
        """Test against pre-generated VMD reference data for specific test case."""

        if not reference_data:
            pytest.skip("No VMD reference data available")

        if test_case not in reference_data:
            pytest.skip(f"No reference data for test case: {test_case}")

        ref_data = reference_data[test_case]

        # Extract parameters from the reference data structure
        params = ref_data.get('parameters', {})
        probe_radius = params.get('srad', 1.4)
        n_samples = params.get('samples', 500)

        # Load test file and run our implementation
        # For lysozyme tests, use local test files
        if test_case.startswith("lysozyme"):
            test_file = Path(__file__).parent / "resources" / f"test_{test_case}.xyz"
            if not test_file.exists():
                pytest.skip(f"Test file does not exist: {test_file}")
            coords, radii = self._load_xyz_file(test_file)
        else:
            # For diverse protein tests, use the input_file from reference data
            input_file = params.get('input_file')
            if not input_file or not Path(input_file).exists():
                pytest.skip(f"Input file not found: {input_file}")
            coords, radii = self._load_xyz_file(Path(input_file))

        our_sasa, our_points = sasa_ext.compute_sasa(
            coords, radii,
            probe_radius=probe_radius,
            n_samples=n_samples,
            seed=38572111  # Match VMD seed
        )

        # Compare total SASA
        results = ref_data.get('results', {})
        ref_sasa = results.get('sasa_area')
        if ref_sasa is None:
            pytest.skip(f"No reference SASA area for {test_case}")

        relative_error = abs(our_sasa - ref_sasa) / ref_sasa

        assert relative_error < tolerances['area_relative'], (
            f"SASA mismatch for {test_case}: "
            f"ours={our_sasa:.2f}, ref={ref_sasa:.2f}, "
            f"error={relative_error*100:.2f}%"
        )

        # Compare point counts (should be similar but not exact due to randomness)
        ref_points = results.get('n_surface_points')
        if ref_points is not None:
            point_error = abs(len(our_points) - ref_points) / ref_points

            assert point_error < tolerances['point_count_relative'], (
                f"Point count mismatch for {test_case}: "
                f"ours={len(our_points)}, ref={ref_points}, "
                f"error={point_error*100:.2f}%"
            )

    def _load_xyz_file(self, xyz_file, vdw_radii=None):
        """Load coordinates and radii from XYZ file."""
        if vdw_radii is None:
            # Default VdW radii if not provided
            vdw_radii = {
                'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
                'S': 1.80, 'P': 1.80, 'He': 1.40
            }

        with open(xyz_file, 'r') as f:
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


class TestVMDAlgorithmFidelity:
    """Test that our algorithm matches VMD's behavior precisely."""

    def test_vmd_seed_reproducibility(self):
        """Test that using VMD's seed gives consistent results."""

        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        radii = np.array([1.5], dtype=np.float32)

        # VMD's fixed seed
        vmd_seed = 38572111

        # Multiple runs with VMD seed
        results = []
        for _ in range(3):
            sasa, points = sasa_ext.compute_sasa(
                coords, radii, probe_radius=1.4, n_samples=500, seed=vmd_seed
            )
            results.append((sasa, len(points), tuple(points[0]) if len(points) > 0 else None))

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result[0] == first_result[0], "SASA should be identical with VMD seed"
            assert result[1] == first_result[1], "Point count should be identical"
            if result[2] is not None and first_result[2] is not None:
                np.testing.assert_array_almost_equal(
                    result[2], first_result[2], decimal=10,
                    err_msg="First surface point should be identical"
                )

    def test_vmd_parameter_compatibility(self, sasa_parameters):
        """Test parameter ranges that VMD typically uses."""

        # Simple test molecule
        coords = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ], dtype=np.float32)
        radii = np.array([1.7, 1.7], dtype=np.float32)  # Carbon atoms

        # Test VMD's typical parameter ranges
        vmd_params = [
            {'probe_radius': 1.4, 'n_samples': 500},   # Standard water probe
            {'probe_radius': 1.4, 'n_samples': 1000},  # High accuracy
            {'probe_radius': 0.0, 'n_samples': 500},   # Van der Waals surface
            {'probe_radius': 2.8, 'n_samples': 500},   # Large probe
        ]

        for params in vmd_params:
            sasa, points = sasa_ext.compute_sasa(
                coords, radii, seed=38572111, **params
            )

            # Basic sanity checks
            assert sasa > 0, f"SASA should be positive for params {params}"
            assert len(points) > 0, f"Should generate points for params {params}"

            # SASA should scale reasonably with probe radius
            expected_min_radius = max(radii) + params['probe_radius']
            # Each atom contributes at least some area
            assert sasa > 0.1 * 4 * np.pi * expected_min_radius**2, \
                f"SASA too small for params {params}"

    def test_sphere_point_generation_matches_vmd(self):
        """Test that sphere point generation follows VMD's method."""

        # Single atom test with various sample counts
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        radii = np.array([1.0], dtype=np.float32)

        for n_samples in [100, 500, 1000]:
            # Use VMD seed
            total_sasa, points = sasa_ext.compute_sasa(
                coords, radii, probe_radius=0.0, n_samples=n_samples, seed=38572111
            )

            # All samples should be exposed for isolated atom
            assert len(points) == n_samples, \
                f"Expected {n_samples} points for isolated atom, got {len(points)}"

            # Points should be on sphere surface
            distances = np.linalg.norm(points - coords[0], axis=1)
            expected_radius = radii[0]
            max_error = np.max(np.abs(distances - expected_radius))
            assert max_error < 1e-6, f"Points not on sphere surface: max error {max_error}"

            # Area should converge to analytical result
            expected_area = 4 * np.pi * radii[0]**2
            relative_error = abs(total_sasa - expected_area) / expected_area
            tolerance = 1.0 / np.sqrt(n_samples)  # Monte Carlo convergence rate
            assert relative_error < 5 * tolerance, \
                f"Area doesn't converge: {relative_error:.4f} > {5*tolerance:.4f}"


class TestParameterSweepAgainstExpectations:
    """Test parameter effects match expected VMD behavior."""

    def test_probe_radius_sweep(self):
        """Test that probe radius has expected monotonic effect."""

        # Two-atom system
        coords = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], dtype=np.float32)
        radii = np.array([1.7, 1.7], dtype=np.float32)

        probe_radii = [0.0, 0.5, 1.0, 1.4, 2.0, 3.0]
        sasa_values = []

        for probe_radius in probe_radii:
            sasa, _ = sasa_ext.compute_sasa(
                coords, radii, probe_radius=probe_radius, n_samples=1000, seed=42
            )
            sasa_values.append(sasa)

        # SASA should increase monotonically with probe radius
        for i in range(1, len(sasa_values)):
            assert sasa_values[i] >= sasa_values[i-1], \
                f"SASA should increase with probe radius: {probe_radii[i-1]:.1f}Å -> {probe_radii[i]:.1f}Å"

        # Check reasonable scaling
        ratio_large_small = sasa_values[-1] / sasa_values[0]
        assert 1.5 < ratio_large_small < 10, \
            f"Probe radius effect seems unreasonable: {ratio_large_small:.2f}x"

    def test_sample_count_convergence(self, tolerances):
        """Test that SASA converges with increasing sample count."""

        # Use a more complex geometry to ensure variance exists
        coords = np.array([[0.0, 0.0, 0.0], [3.1, 0.0, 0.0]], dtype=np.float32)
        radii = np.array([1.5, 1.5], dtype=np.float32)

        sample_counts = [100, 200, 500, 1000, 2000]
        sasa_values = []
        std_values = []

        for n_samples in sample_counts:
            # Run multiple times to estimate variance
            runs = []
            for seed in range(10):
                sasa, _ = sasa_ext.compute_sasa(
                    coords, radii, probe_radius=1.4, n_samples=n_samples, seed=seed
                )
                runs.append(sasa)

            mean_sasa = np.mean(runs)
            std_sasa = np.std(runs)
            sasa_values.append(mean_sasa)
            std_values.append(std_sasa)

        # Standard deviation should generally decrease or stay low
        # If variance is very small, that's actually good convergence
        max_std = max(std_values)
        min_std = min(std_values)

        if max_std > 1e-10:  # Only test convergence if we have measurable variance
            # Standard deviation should not increase significantly
            for i in range(1, len(std_values)):
                if std_values[i-1] > 1e-10:  # Avoid division by very small numbers
                    ratio = std_values[i] / std_values[i-1]
                    # Should generally decrease or stay about the same
                    assert ratio <= 2.0, \
                        f"Standard deviation increased significantly: {std_values[i]:.6f} vs {std_values[i-1]:.6f}"

        # Mean should stabilize for large sample counts
        high_sample_means = sasa_values[-3:]  # Last 3 values
        coefficient_of_variation = np.std(high_sample_means) / np.mean(high_sample_means)
        assert coefficient_of_variation < tolerances['convergence'], \
            f"SASA not converged at high sample counts: CV = {coefficient_of_variation:.4f}"

    def test_distance_dependence(self):
        """Test that SASA depends correctly on inter-atomic distances."""

        base_coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        radii = np.array([1.5, 1.5], dtype=np.float32)

        distances = [2.0, 3.0, 4.0, 6.0, 10.0, 20.0]  # Distance between atoms
        sasa_values = []

        for distance in distances:
            coords = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float32)
            sasa, _ = sasa_ext.compute_sasa(
                coords, radii, probe_radius=1.4, n_samples=1000, seed=42
            )
            sasa_values.append(sasa)

        # SASA should increase with distance (less burial)
        for i in range(1, len(sasa_values)):
            assert sasa_values[i] >= sasa_values[i-1], \
                f"SASA should increase with distance: {distances[i-1]:.1f}Å -> {distances[i]:.1f}Å"

        # At large distances, should approach 2x single atom
        single_atom_sasa, _ = sasa_ext.compute_sasa(
            base_coords, radii[:1], probe_radius=1.4, n_samples=1000, seed=42
        )

        large_distance_sasa = sasa_values[-1]  # Largest distance
        expected_sasa = 2 * single_atom_sasa
        relative_error = abs(large_distance_sasa - expected_sasa) / expected_sasa

        assert relative_error < 0.05, \
            f"Large distance SASA should approach 2x single atom: {relative_error:.3f}"


class TestIntegrationWithSASACoreModule:
    """Test integration between C extension and Python core module."""

    def test_sasa_core_integration(self, xyz_files):
        """Test that SASA core module works with C extension."""
        from sasa_lammps.sasa_core import compute_sasa_from_xyz

        # Test with single atom file
        xyz_file = xyz_files['single_atom']
        coords, radii = parse_xyz_file(str(xyz_file))

        # Test direct C extension call
        sasa1, points1 = sasa_ext.compute_sasa(coords, radii, probe_radius=1.4, n_samples=500)

        # Test through replacement module
        sasa2, points2 = compute_sasa_from_xyz(
            str(xyz_file), srad=1.4, samples=500, points=True
        )

        # Results should be identical (same seed used)
        assert abs(sasa1 - sasa2) < 1e-6, "Direct and replacement module calls should match"

        # Convert points2 back to numpy for comparison
        points2_array = np.array(points2)
        np.testing.assert_array_almost_equal(points1, points2_array, decimal=6)

    def test_xyz_parsing_accuracy(self, xyz_files, vdw_radii):
        """Test that XYZ file parsing assigns correct radii."""
        from sasa_lammps.utils import parse_xyz_file

        # Test mixed elements file
        xyz_file = xyz_files['mixed_elements']
        coords, radii = parse_xyz_file(str(xyz_file))

        expected_elements = ['C', 'H', 'N', 'O']
        expected_radii = [vdw_radii[elem] for elem in expected_elements]

        assert len(coords) == 4, "Should parse 4 atoms"
        assert len(radii) == 4, "Should assign 4 radii"

        np.testing.assert_array_almost_equal(radii, expected_radii, decimal=6)

        # Check coordinates are parsed correctly
        expected_coords = np.array([
            [0.0, 0.0, 0.0],  # C
            [1.0, 0.0, 0.0],  # H
            [0.0, 1.0, 0.0],  # N
            [0.0, 0.0, 1.0],  # O
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(coords, expected_coords, decimal=6)
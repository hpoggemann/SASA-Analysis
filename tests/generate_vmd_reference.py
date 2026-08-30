#!/usr/bin/env python3
"""
Generate VMD reference data for SASA calculation testing.

This script uses VMD-python 3.1.6 (package available in conda-forge: 
https://anaconda.org/conda-forge/vmd-python) to create reference test data
that we can later compare against the C extension implementation.

NOTE: This script should be run from the test/ directory in the 'sasa_vmd_ref' conda environment:
cd test && conda activate sasa_vmd_ref && python generate_vmd_reference.py
"""

import sys
import json
from pathlib import Path

try:
    from vmd import atomsel, molecule
    print("Successfully imported VMD")
except ImportError as e:
    print(f"Failed to import VMD: {e}")
    print("Make sure you're running in the 'sasa' conda environment:")
    print("conda activate sasa && python generate_vmd_reference.py")
    sys.exit(1)

def convert_gro_to_xyz(gro_file, xyz_file):
    """Convert GRO file to XYZ format for VMD."""
    print(f"Converting {gro_file} to {xyz_file}")

    with open(gro_file, 'r') as f:
        lines = f.readlines()

    # Parse GRO file
    title = lines[0].strip()
    n_atoms = int(lines[1].strip())

    atoms = []
    for i in range(2, 2 + n_atoms):
        line = lines[i]
        # GRO format: residue_num, residue_name, atom_name, atom_num, x, y, z
        # Extract element from atom name (first letter)
        atom_name = line[10:15].strip()
        element = atom_name[0]  # First character is usually the element

        # Coordinates in nm, convert to angstrom
        x = float(line[20:28]) * 10
        y = float(line[28:36]) * 10
        z = float(line[36:44]) * 10

        atoms.append((element, x, y, z))

    # Write XYZ file
    with open(xyz_file, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f"{title}\n")
        for element, x, y, z in atoms:
            f.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Converted {n_atoms} atoms to {xyz_file}")
    return xyz_file

def generate_vmd_reference_data():
    """Generate reference SASA data using VMD for different test cases."""

    # Test cases with different parameters
    test_cases = [
        {
            "name": "lysozyme_small",
            "input_file": "../example/lysozyme_part.gro",
            "srad": 1.4,
            "samples": 100,
            "description": "Lysozyme fragment with standard water probe, low sampling"
        },
        {
            "name": "lysozyme_medium",
            "input_file": "../example/lysozyme_part.gro",
            "srad": 1.4,
            "samples": 500,
            "description": "Lysozyme fragment with standard water probe, medium sampling"
        },
        {
            "name": "lysozyme_small_probe",
            "input_file": "../example/lysozyme_part.gro",
            "srad": 1.0,
            "samples": 100,
            "description": "Lysozyme fragment with smaller probe radius"
        },
        {
            "name": "lysozyme_large_probe",
            "input_file": "../example/lysozyme_part.gro",
            "srad": 2.0,
            "samples": 100,
            "description": "Lysozyme fragment with larger probe radius"
        }
    ]

    reference_data = {}

    for test_case in test_cases:
        print(f"\nGenerating reference data for: {test_case['name']}")

        try:
            # Convert GRO to XYZ for VMD
            xyz_file = f"resources/test_{test_case['name']}.xyz"
            convert_gro_to_xyz(test_case["input_file"], xyz_file)

            # Load molecule in VMD
            molid = molecule.load("xyz", xyz_file)
            print(f"Loaded molecule {molid}")

            # Create atom selection (all atoms)
            sel = atomsel("all", molid=molid)
            print(f"Selected {len(sel)} atoms")

            # Calculate SASA with surface points
            print(f"Computing SASA with srad={test_case['srad']}, samples={test_case['samples']}")
            sasa_area, surface_points = sel.sasa(
                srad=test_case["srad"],
                samples=test_case["samples"],
                points=True
            )

            print(f"SASA area: {sasa_area:.6f}")
            print(f"Surface points: {len(surface_points)}")

            # Get atom information for validation
            # Use individual attribute access instead of multi-attribute get
            atom_coords = [[x, y, z] for x, y, z in zip(sel.x, sel.y, sel.z)]
            atom_types = list(sel.element)
            atom_radii = list(sel.radius)

            # Store reference data
            reference_data[test_case["name"]] = {
                "parameters": {
                    "srad": test_case["srad"],
                    "samples": test_case["samples"],
                    "input_file": test_case["input_file"]
                },
                "results": {
                    "sasa_area": float(sasa_area),
                    "n_surface_points": len(surface_points),
                    "surface_points": [list(point) for point in surface_points],
                    "n_atoms": len(sel)
                },
                "atom_data": {
                    "coordinates": [list(coord) for coord in atom_coords],
                    "elements": list(atom_types),
                    "radii": list(atom_radii)
                },
                "description": test_case["description"],
                "files": {
                    "xyz_file": xyz_file
                }
            }

            # Clean up molecule
            molecule.delete(molid)

        except Exception as e:
            print(f"Error generating data for {test_case['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save reference data to JSON
    output_file = "resources/vmd_reference_data.json"
    with open(output_file, 'w') as f:
        json.dump(reference_data, f, indent=2)

    print(f"\nReference data saved to {output_file}")
    print(f"Generated data for {len(reference_data)} test cases")

    # Create summary
    print("\nSummary:")
    for name, data in reference_data.items():
        params = data["parameters"]
        results = data["results"]
        print(f"  {name}:")
        print(f"    srad={params['srad']}, samples={params['samples']}")
        print(f"    atoms={results['n_atoms']}, surface_points={results['n_surface_points']}")
        print(f"    sasa_area={results['sasa_area']:.6f}")

    return reference_data

if __name__ == "__main__":
    # Ensure we're in the right directory
    if not Path("../example/lysozyme_part.gro").exists():
        print("Error: Please run this script from the test/ directory in the SASA-Analysis project")
        sys.exit(1)

    print("Starting VMD reference data generation...")
    reference_data = generate_vmd_reference_data()

    if reference_data:
        print(f"\nSuccessfully generated reference data for {len(reference_data)} test cases")
    else:
        print("\nNo reference data generated - check for errors above")
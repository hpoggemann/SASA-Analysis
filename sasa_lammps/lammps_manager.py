"""
LAMMPS binary manager for automatic download and setup of pre-built LAMMPS binaries.
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path


class LammpsManager:
    """Manages automatic download and setup of LAMMPS pre-built binaries."""

    LAMMPS_URL = "https://download.lammps.org/static/lammps-linux-x86_64-latest.tar.gz"

    @staticmethod
    def _get_cache_dir():
        """Get the cache directory following XDG Base Directory Specification."""
        cache_home = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        return Path(cache_home) / "sasa_lammps"

    def __init__(self):
        self.cache_dir = self._get_cache_dir()
        self.lammps_exe = None
        self._setup_lammps()

    def _setup_lammps(self):
        """Download and setup LAMMPS binary if not already available."""
        lammps_path = Path(self.cache_dir) / "lammps-static" / "bin" / "lmp"
        
        # Check if LAMMPS is already downloaded and available
        if Path(lammps_path).exists() and os.access(lammps_path, os.X_OK):
            self.lammps_exe = lammps_path
            return

        # Download and extract LAMMPS
        self._download_and_extract_lammps()

        # Verify the binary exists after extraction
        if Path(lammps_path).exists() and os.access(lammps_path, os.X_OK):
            self.lammps_exe = lammps_path
        else:
            raise RuntimeError(f"LAMMPS binary not found at {lammps_path} after extraction")

    def _download_and_extract_lammps(self):
        """Download and extract LAMMPS tarball."""
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        tarball_path = Path(self.cache_dir) / "lammps-linux-x86_64-latest.tar.gz"

        print(f"Downloading LAMMPS binary from {self.LAMMPS_URL}...")

        try:
            # Download the tarball
            urllib.request.urlretrieve(self.LAMMPS_URL, tarball_path)

            print(f"Extracting LAMMPS binary to {self.cache_dir}...")

            # Extract the tarball
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(path=self.cache_dir, filter="data")

            # Clean up the tarball
            os.remove(tarball_path)

            print("LAMMPS binary setup complete.")

        except Exception as e:
            raise RuntimeError(f"Failed to download or extract LAMMPS binary: {e}")

    def get_lammps_executable(self):
        """Return the path to the LAMMPS executable."""
        if self.lammps_exe is None:
            raise RuntimeError("LAMMPS executable not available")
        return self.lammps_exe

    def cleanup(self):
        """Remove downloaded LAMMPS files to free up space."""
        if Path(self.cache_dir).exists():
            shutil.rmtree(self.cache_dir)
            print(f"Cleaned up LAMMPS files from {self.cache_dir}")


def get_lammps_executable():
    """
    Convenience function to get the LAMMPS executable path.
    Downloads and sets up LAMMPS automatically if needed.

    Returns
    -------
    str
        Path to the LAMMPS executable
    """
    manager = LammpsManager()
    return manager.get_lammps_executable()
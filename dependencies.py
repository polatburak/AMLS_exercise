#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dependencies.py

This script lists all dependencies required for the AMLS ECG classification project.
It provides functions to check and install the dependencies.
"""

import sys
import subprocess
import pkg_resources


# Define all dependencies by category
DEPENDENCIES = {
    # Core libraries
    "core": [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
    ],
    
    # Deep learning libraries
    "deep_learning": [
        "tensorflow>=2.8.0",
        "keras>=2.8.0",
    ],
    
    # Visualization libraries
    "visualization": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    
    # Biomedical signal processing
    "biomedical": [
        "wfdb>=3.4.0",
    ],
    
    # Utility libraries
    "utilities": [
        "tqdm>=4.62.0",
    ],
}

# Flatten dependencies for easy installation
ALL_DEPENDENCIES = [dep for category in DEPENDENCIES.values() for dep in category]


def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        tuple: (missing_deps, version_mismatch_deps) Lists of missing dependencies and
               dependencies with version mismatches.
    """
    missing_deps = []
    version_mismatch_deps = []
    
    for dependency in ALL_DEPENDENCIES:
        # Parse the package name and version requirement
        if ">=" in dependency:
            package_name, required_version = dependency.split(">=")
        else:
            package_name, required_version = dependency, None
            
        try:
            if required_version:
                installed_version = pkg_resources.get_distribution(package_name).version
                if not _version_satisfies_requirement(installed_version, required_version):
                    version_mismatch_deps.append((package_name, installed_version, required_version))
        except pkg_resources.DistributionNotFound:
            missing_deps.append(package_name)
    
    return missing_deps, version_mismatch_deps


def _version_satisfies_requirement(installed_version, required_version):
    """Check if installed version meets the required version."""
    installed_parts = [int(x) for x in installed_version.split('.')]
    required_parts = [int(x) for x in required_version.split('.')]
    
    for i in range(min(len(installed_parts), len(required_parts))):
        if installed_parts[i] < required_parts[i]:
            return False
        elif installed_parts[i] > required_parts[i]:
            return True
    
    return len(installed_parts) >= len(required_parts)


def install_dependencies(dependencies=None):
    """
    Install the specified dependencies or all dependencies if none specified.
    
    Args:
        dependencies (list): List of dependencies to install. If None, installs all dependencies.
    
    Returns:
        bool: True if installation was successful, False otherwise.
    """
    deps_to_install = dependencies or ALL_DEPENDENCIES
    
    print(f"Installing {len(deps_to_install)} dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps_to_install)
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error occurred during installation.")
        return False


def print_dependency_list():
    """Print the list of all dependencies by category."""
    print("ECG Classification Project Dependencies:")
    print("=" * 40)
    
    for category, deps in DEPENDENCIES.items():
        print(f"\n{category.upper()} DEPENDENCIES:")
        for dep in deps:
            print(f"  • {dep}")
    
    print("\nTotal dependencies:", len(ALL_DEPENDENCIES))


if __name__ == "__main__":
    print_dependency_list()
    
    # Check if dependencies are already installed
    missing, version_mismatch = check_dependencies()
    
    if missing or version_mismatch:
        print("\nMissing or version mismatch found:")
        
        if missing:
            print("\nMissing packages:")
            for pkg in missing:
                print(f"  • {pkg}")
        
        if version_mismatch:
            print("\nVersion mismatch:")
            for pkg, installed, required in version_mismatch:
                print(f"  • {pkg}: installed {installed}, required >={required}")
        
        # Automatically install missing dependencies
        print("\nInstalling missing dependencies...")
        install_dependencies([dep for dep in ALL_DEPENDENCIES if dep.split(">=")[0] in missing])
    else:
        print("\n✅ All dependencies are already installed with correct versions!")
        
    # Additional project-specific dependencies
    print("\nNote: This project also depends on custom modules:")
    print("  • utils.py (local utility functions)")
    print("  • models/baseline_model.py (STFT-CNN-RNN model architecture)")
    print("  • models/resnet_model.py (ResNet model architecture)")
    print("  • models/training_utils.py (training and evaluation utilities)")
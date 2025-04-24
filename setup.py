from setuptools import setup, find_packages
import subprocess
import sys
import os

def install_dependencies_from_yml():
    """Install dependencies specified in environment.yml using Conda."""
    env_file = "environment.yml"
    if os.path.exists(env_file):
        try:
            # Check if Conda is installed
            conda_executable = os.environ.get('CONDA_EXE', 'conda')
            print(f"Using Conda executable: {conda_executable}")

            # Create the Conda environment
            print(f"Creating Conda environment from {env_file}...")
            subprocess.check_call([conda_executable, "env", "create", "-f", env_file])

            # Activate the Conda environment
            env_name = "samselect"
            activate_command = f"conda activate {env_name} && "

            # Install the package in the activated environment
            subprocess.check_call(
                activate_command + [sys.executable, "-m", "pip", "install", "."],
                shell=True
            )

            print("Dependencies and package installed successfully!")
        except Exception as e:
            print(f"Error installing dependencies from {env_file}: {e}")
            sys.exit(1)
    else:
        print(f"{env_file} not found. Skipping dependency installation.")

# Install dependencies from environment.yml before setting up the package
install_dependencies_from_yml()

setup(
    name="SAMSelect",
    version="0.1",
    packages=find_packages(),  # Automatically find packages in the directory
    include_package_data=True,
    description="SAMSelect: An Automated Spectral Index Search using Segment Anything",
    author="Joost van Dalen, Marc Rußwurm",
    author_email="your.email@example.com",
    entry_points={
        "console_scripts": [
            "samselect=samselect:main",  # Map the command to the script's main function
        ],
    },
    license="MIT",
    python_requires=">=3.6",
)

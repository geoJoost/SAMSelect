from setuptools import setup
import subprocess
import sys
import os

def install_dependencies_from_yml():
    """Install dependencies specified in environment.yml."""
    env_file = "environment.yml"
    if os.path.exists(env_file):
        try:
            # Install the dependencies using pip and PyYAML
            print(f"Installing dependencies from {env_file}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
            import yaml

            with open(env_file, "r") as file:
                env = yaml.safe_load(file)
                dependencies = env.get("dependencies", [])
                pip_dependencies = [
                    dep for dep in dependencies if isinstance(dep, str)
                ]
                for pip_dep in pip_dependencies:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_dep])
                print("Dependencies installed successfully!")
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
    py_modules=["samselect"],  # Reference the single script
    include_package_data=True,
    description="A spectral search algorithm for multi-spectral images using SAM",
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
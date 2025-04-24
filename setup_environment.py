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
                f"{activate_command} python -m pip install -e .",
                shell=True
            )

            print("Dependencies and package installed successfully!")
        except Exception as e:
            print(f"Error installing dependencies from {env_file}: {e}")
            sys.exit(1)
    else:
        print(f"{env_file} not found. Skipping dependency installation.")

if __name__ == "__main__":
    install_dependencies_from_yml()

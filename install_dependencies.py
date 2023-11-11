import subprocess
import importlib

# Define the path to your requirements.txt file
requirements_file = "requirements.txt"

# Read the requirements from the file
with open(requirements_file, "r") as file:
    requirements = file.read().splitlines()

# Loop through the requirements and use pip to install or upgrade them if not already installed
for requirement in requirements:
    # Check if the module is installed
    try:
        importlib.import_module(requirement)
        print(f"{requirement} is already installed.")
    except ImportError:
        # If the module is not installed, install it
        print(f"Installing {requirement}...")
        subprocess.run(
            ["pip", "install", "--upgrade", "--upgrade-strategy", "eager", requirement]
        )

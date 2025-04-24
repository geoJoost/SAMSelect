from setuptools import setup, find_packages

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
            "samselect=samselect.samselect:main",  # Map the command to the script's main function
        ],
    },
    license="MIT",
    python_requires=">=3.6",
)

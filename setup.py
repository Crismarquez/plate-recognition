
from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="recoplate",
    version=0.1,
    description="Plate recognition",
    author="Blue Labs",
    author_email="bluelabs.ai@gmail.com",
    python_requires=">=3.10",
    install_requires=[required_packages],
)
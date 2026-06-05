from setuptools import setup, find_packages

setup(
    name="stochastic_gm",
    version="0.1.0",
    description="3D Stochastic Ground-Motion Simulator",
    author="Luis ALVAREZ",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "numba>=0.55",
    ],
    extras_require={
        "examples": ["matplotlib>=3.4", "pandas"],
    },
    python_requires=">=3.8",
)

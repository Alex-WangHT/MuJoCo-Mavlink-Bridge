from setuptools import setup, find_packages

setup(
    name="mujoco-mavlink-bridge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "mujoco": ["mujoco>=3.0.0"],
        "mavlink": ["pymavlink>=2.4.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    author="MuJoCo MAVLink Bridge Team",
    description="Bridge between PX4 SITL and MuJoCo simulation",
    keywords="mujoco mavlink px4 simulation robotics",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
)

# Tracks

## Data Analysis from Experiments particle tracking and Molecular Dynamics simulation

There are 2 main classes of functions:

* TRAJECTORY FUNCTIONS. Deal with single-particle trajectories, with variable length in general (since particle tracking from experiments produces different-length trajectories).

* SYSTEM INSTANTANEOUS-STATE FUNCTIONS. Deal with system's snapshots analysis. They have variable number of particles (since particle tracking from experiments produces different-length trajectories).

Both classes of functions should be able to yield with both system-wide and local-field magnitudes.

A full (growing) list of functions will be provided in short.


### How to use particle tracking software

Here we explain how to use this software to extract particle positions, trajectories and velocities from a .cine file

> All necessary source files are located inside **particle_tracking** folder

#### Dependencies:
 
 * pandas
 * numpy
 * scipy
 * opencv-python
 * tqdm
 * pims
 * trackpy

The key file here is **__init__.py**

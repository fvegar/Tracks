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

The key file in this whole process is **__init__.py**, first one has to define the folder where .cine files are located:

```python
if __name__ == "__main__":

    folder = 'D:/'
    files = glob.glob(folder + '*Aspas*.cine') # List with all .cine files
```
As we can see, a list of all .cine files (matching a certain naming style) is created. Then the function 
**detect_particles_and_save_data** is called. This function takes as an argument a path to a .cine file, it
detects all particles, joins their trajectories, derive its velocities and save all to files (plus an information
text file for each experiment)

> Resulting data files are saved in a **pickle** binary format with **xz** compression.

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
text file for each experiment). Using multiple cores.

> Resulting data files are saved in a **pickle** binary format with **xz** compression.

The parameters that define each experiment are hardcoded in the definition of the function:

```python
    vid = pims.Cine(file)

    original_file = file
    fps = vid.frame_rate
    shape = vid.frame_shape
    date = str(vid.frame_time_stamps[0][0])
    exposure = int(1000000*vid.all_exposures[0]) #in microseconds
    n_frames = vid.image_count
    recording_time = n_frames/fps

    N = int(file.split('_')[-3].split('N')[-1])
    power = int(file.split('_')[-2].split('p')[-1])
    if power>=100:
        power /= 10

    lights = 'luzLejana'
    camera_distance = 0.924 #in meters (bolas, cercana 0.535)
    pixel_ratio = 1950 #in px/meter, CAMBIARLO
    particle_diameter_px = 79
    particle_diameter_m = 0.0725
    particle_shape = 'rotating disk'
    system_diameter = 0.725 #in meters
    packing_fraction = N*(particle_diameter_m/system_diameter)**2
    ROI_center = [649, 392] #in pixels
    ROI_radius = 408
    # Hashing function to asign an unique id to each experiment
    # date+time should be specific enough to tell them apart
    hash_object = hashlib.md5(date.encode())
    experiment_id = str(hash_object.hexdigest())
```

As we can see, some parameters such as the number of particles (N) or fan power (P) are read from the file name. So the naming convetion
for .cine videos is:

> serieAspas_N50_p23_1.cine

Also note that an unique experiment_id (ej: '87387719783ab1ba0d1d2008cd1f2ac5') is generated, this will be used to identify each video in the future.

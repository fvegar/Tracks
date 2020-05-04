# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:34:29 2019

@author: malopez
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import printp, distanceToCenter, angle_from_2D_points
from pandas.api.types import is_numeric_dtype


def remove_outliers(df, column_name, min_quantile=0.01, max_quantile=0.99):
    """
    

    Parameters
    ----------
    df : pandas DataFrame
        Data
    column_name : string
        Name of the column used to search for outliers

    Returns
    -------
    df : pandas DataFrame
        Data with rows removed if an outlier is present

    """

    low = min_quantile
    high = max_quantile
    quant_df = df.quantile([low, high])
    if is_numeric_dtype(df[column_name]):
        df = df[(df[column_name] > quant_df.loc[low, column_name]) & (df[column_name] < quant_df.loc[high, column_name])]
    return df


def calculate_kurtosis(vel_data):
    """ Calculation of the kurtosis from a series of velocities

        Parameters
        ----------
        vel_data : pandas Dataframe
            It must be have the usual structure with at least the folowing columns:
            ['frame', 'track', 'vx', 'vy']

        Returns
        -------
        float
            Value of the kurtosis for the velocity distribution present in given dataframe
    """
    # First we have to select only velocities, if they are not present in the
    # dataframe an error will be raised
    vels = vel_data[['vx', 'vy']].values

    # Modulus of the velocity and kurtosis calculation
    v = np.linalg.norm(vels, axis=1)
    v2 = v**2
    k = (v**4).mean()/((v2).mean())**2
    return k


def compute_a2(vel_data, dimensions):
    """ Calculation of the a2 coefficient for a system of any given dimensions
        from a series of velocities

        Parameters
        ----------
        vel_data : pandas Dataframe
            It must be have the usual structure with at least the folowing columns:
            ['frame', 'track', 'vx', 'vy']
        dimensions : int
            Dimensionality of the system (ie: 2, 3, ...)

        Returns
        -------
        float
            Value of the a2 coefficient for the velocity distribution present in given dataframe
    """
    k = calculate_kurtosis(vel_data)
    a2 = (dimensions/(dimensions+2))*k -1
    return a2


def theoretical_a2(alpha, dimensions):
    """ Theoretical a2 coefficient for a system of any given dimensions and inelastic particles

        Parameters
        ----------
        alpha : float
            Coefficient of normal restitution for particle-particle collisions
        dimensions : int
            Dimensionality of the system (ie: 2, 3, ...)

        Returns
        -------
        float
            Value of the theoretical a2 coefficient 
    """
    a2 = (16*(1-alpha) * (1 - 2*(alpha**2))) / (9 + 24*dimensions - alpha*(41 - 8*dimensions) + 30*(1-alpha)*(alpha**2))
    return a2


def meanSquaredDisplacement(data, max_steps='all', column='x', trajectory=1):
    """ This function calculates the mean square displacement of a particle's
        trayectory, for a given component ('x' or 'y').

        Parameters
        ----------
        data : pandas Dataframe
            It must be have the usual structure with at least the folowing columns:
            ['frame', 'track', 'x', 'y']
        max_steps : int, optional (default: 'all')
            Maximum number of steps (lag time) for which the MSD is calculated
        column : string (default: 'x')
            Name of the column for the component to be analysed
        trajectory : int (default: 1)
            Lets the user decide which trajectory to use for MSD calculation

        Returns
        -------
        results : array
            A numpy array of shape (N, 2) where N is equal to max_steps. The first
            column is only an index indicating the lag time while the second column
            contains the value of the mean square displacement for that interval.
    """

    # First, we select only the data for desired track (trajectory)
    sub_data = data[data.track == trajectory]
    sub_data = sub_data[column].values

    if max_steps == 'all':
        max_steps = len(sub_data) - 1

    results = []
    for i in range(max_steps-1):
        results.append([i+1, simpleMSD(sub_data, i+1)])

    results = np.array(results)
    return results


def simpleMSD(pos_data, lagTime):
    """ Simple MSD function, returns the square displacement given a 1D array
        and a lag time (interval) 
        
        Parameters
        ----------
        pos_data : array
            1D positions array for a single coordinate
        lagTime : float
            Interval (lag time) for which the MSD is calculated

        Returns
        -------
        results : float
            Value of the mean square displacement for given data and interval.
    """

    return np.mean((pos_data[lagTime:] - pos_data[:-lagTime])**2)


def plotMSD(data, timePerFrame=1, max_steps='all', column='x', trajectory=1, log=False):
    """ Plots the mean square displacement for a given track """

    MSD = meanSquaredDisplacement(data, max_steps=max_steps, column=column, trajectory=trajectory)
    if log == True:
        dt = np.log(MSD[:, 0]*timePerFrame)
        y = np.log(MSD[:, 1])

        fig, ax = plt.subplots(figsize=(8, 8), dpi=250)
        #ax.set_xlim([0,dt.max()])
        #ax.set_ylim([0,y.max()])

        ax.set_xlabel('log(t) (s)')
        ax.set_ylabel(r'$log\langle\left( x - x_{0}\right)^{2} \rangle \left( m^{2}\right)$')
        plt.scatter(dt, y, s=1)
    else:
        dt = MSD[:, 0]*timePerFrame
        y = MSD[:, 1]

        fig, ax = plt.subplots(figsize=(8, 8), dpi=250)
        ax.set_xlim([0, dt.max()])
        ax.set_ylim([0, y.max()])

        ax.set_xlabel('t (s)')
        ax.set_ylabel(r'$\langle\left( x - x_{0}\right)^{2} \rangle \left( m^{2}\right)$')
        plt.scatter(dt, y, s=1)


def msd_ensemble(data, max_steps='all', column='x'):
    """ Calculates the mean square displacement using every available track on the Dataframe

        Parameters
        ----------
        data : pandas Dataframe
            It must be have the usual structure with at least the folowing columns:
            ['frame', 'track', 'x', 'y']
        max_steps : int, optional (default: 'all')
            Maximum number of steps (lag time) for which the MSD is calculated
        column : string (default: 'x')
            Name of the column for the component to be analysed

        Returns
        -------
        results : array
            A 1D numpy array of shape N where N is equal to max_steps.
            Containing the value of the mean square displacement for that interval.
    """
    ensemble = pd.DataFrame()
    for traj in set(data.track):
        if data[data.track == traj].shape[0] > 2:
            msd_traj = meanSquaredDisplacement(data, max_steps=max_steps, column=column, trajectory=traj)
            # I only need the data not the steps columns
            msd_traj = pd.DataFrame(msd_traj[:, 1])
            ensemble = pd.concat((ensemble, msd_traj), axis=1)
            printp('Calculating MSD for trajectory: '+ str(traj) + '/'+ str(len(set(data.track))))

    return ensemble.mean(axis=1)


def plot_msd_ensemble(data, timePerFrame=1, max_steps='all', column='x', log=False):
    """ Plots the mean square displacement using every available track on the Dataframe

        Parameters
        ----------
        data : pandas Dataframe
            It must be have the usual structure with at least the folowing columns:
            ['frame', 'track', 'x', 'y']
        timePerFrame : float (default: 1)
            This is the ratio frames/real_time (1/fps). So that the function can plot
            with the axis in the right units.
        max_steps : int, optional (default: 'all')
            Maximum number of steps (lag time) for which the MSD is calculated
        column : string (default: 'x')
            Name of the column for the component to be analysed
        trajectory : int (default: 1)
            Lets the user decide which trajectory to use for MSD calculation
        log : boolean, optional (default: False)
            Choose if the plot should be in logarithm scale
    """
    msd = msd_ensemble(data, max_steps=max_steps, column=column)

    if log == True:
        dt = np.arange(len(msd))
        dt = np.log(dt*timePerFrame)
        y = np.log(msd.values)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=250)
        #ax.set_xlim([0,dt.max()])
        #ax.set_ylim([0,y.max()])

        ax.set_xlabel('log(t) (s)')
        ax.set_ylabel(r'$log\langle\left( x - x_{0}\right)^{2} \rangle \left( m^{2}\right)$')
        plt.scatter(dt, y, s=1)
    else:
        dt = np.arange(len(msd))
        y = msd.values

        fig, ax = plt.subplots(figsize=(8, 8), dpi=250)
        ax.set_xlim([0, dt.max()])
        ax.set_ylim([0, y.max()])

        ax.set_xlabel('t (s)')
        ax.set_ylabel(r'$\langle\left( x - x_{0}\right)^{2} \rangle \left( m^{2}\right)$')
        plt.scatter(dt, y, s=1)


def velocity_autocorrelation(vel_data, trajectory, max_lag='all'):
    """ Calculates the velocity autocorrelation function for a single particle

    Parameters
    ----------
    vel_data : pandas Dataframe
        It must be have the usual structure with at least the folowing columns:
        ['frame', 'track', 'vx', 'vy']
    trayectory : int)
        The index of the track we want to colculate de VAC of.
    max_lag : int, optional (default: 'all')
        Maximum number of steps (lag time) for which the VAC is calculated

    Returns
    -------
    vac : pandas Dataframe
        A 1D pandas dataframe of shape N where N is equal to max_steps.
        Containing the value of the VAC for each interval.
    """
    sub_data = vel_data[vel_data.track == trajectory]
    sub_data = sub_data[['vx', 'vy']]

    if max_lag == 'all':
        max_lag = len(sub_data) - 1

    results = []
    for i in range(max_lag-1):
        if i == 0:
            vel_aut_currentLag = 1
        else:
            vel_aut_currentLag = np.vdot(sub_data[i:], sub_data[:-i]) / np.vdot(sub_data[:-i], sub_data[:-i])
        results.append(vel_aut_currentLag)

    vac = pd.DataFrame(results)
    return vac


def velocity_autocorrelation_ensemble(vel_data, max_lag='all'):
    """ Calculates the velocity autocorrelation function for integrating for
        every particle in the system

    Parameters
    ----------
    vel_data : pandas Dataframe
        It must be have the usual structure with at least the folowing columns:
        ['frame', 'track', 'vx', 'vy']
    max_lag : int, optional (default: 'all')
        Maximum number of steps (lag time) for which the VAC is calculated

    Returns
    -------
    vac : pandas Dataframe
        A 1D pandas dataframe of shape N where N is equal to max_steps.
        Containing the value of the VAC for each interval.
    """
    n_tracks = len(set(vel_data.track))

    ensemble = []
    for traj in set(vel_data.track):
        if vel_data[vel_data.track == traj].shape[0] > 2:
            vel_aut = velocity_autocorrelation(vel_data, traj, max_lag=max_lag)
            ensemble.append(vel_aut)
            #ensemble = pd.concat((ensemble, vel_aut), axis=1)
            printp(f'Integrating velocity autocorrelations: {traj} / {n_tracks}')
    vac = pd.concat(ensemble, axis=1)
    vac = np.mean(vac, axis=1)
    #vac = ensemble.mean(axis=1)
    return vac



def instant_radial_distribution_function(data, frame, roi_center=[650, 400], roi_radius=390, min_step=2):
    """ Calculates the radial distribution function, g(r), for a single frames.
        All parameters must be in the same units (either pixels, cm or natural units)

    Parameters
    ----------
    data : pandas Dataframe
        It must be have the usual structure with at least the folowing columns:
        ['frame', 'track', 'x', 'y']
    frame : int
        Frame number
    roi_center : int list or tuple
        Coordinates of the central point of the region of interest (in the case of a
        circular ROI)
    roi_radius : int
        Self-explanatory
    min_step : float
        This is the size of the distance step we use to calculate g(r). Reducing this
        will increase the resolution of data (at the cost of higher computing times)

    Returns
    -------
    gr : pandas Dataframe
        A pandas dataframe of shape (N, 1) where N is equal to max_radial_distance/min_step.
        Containing the value of the g(r) for each step.
    """


    # The maximum radial distance we can get away from each particle is the distance
    # from the particle to the edge of the ROI. We add a new column with that info
    # ONLY WHEN IS NOT ALREADY PRESENT (save time)
    if 'dist_edge' not in data.columns:
        distances_to_center = distanceToCenter(data.x, data.y, roi_center[0], roi_center[1])
        distances_to_edge = roi_radius - distances_to_center
        data = data.assign(dist_edge=distances_to_edge)

    # Select data only for desired frame
    sub_data = data[data.frame == frame]

    # First we have to loop over all particles present in that frame and calculate
    # g(r) for every particle (trajectory)
    instant_gr = pd.DataFrame() # Empty dataframe to save results
    for index, current_row in sub_data.iterrows():
        # Distances for every particle to current particle
        distances_to_particle = distanceToCenter(sub_data.x, sub_data.y, current_row.x, current_row.y)
        # min_step tells us which is the unit of distance for each step increasing
        # the radial distance from particle's center. Here we create a list with the
        # distances we have to cover

        covered_distances = np.arange(0, current_row['dist_edge']+min_step, step=min_step)
        current_particle_gr = np.histogram(distances_to_particle, bins=covered_distances)[0]

        # Normalization
        # # El [:-1] es para que tenga la misma longitud current_particle_gr
        normalization_constants = np.pi * min_step * covered_distances[:-1] # El [:-1] es para que tenga la misma longitud current_particle_gr
        normalization_constants[0] = 1 # To avoid dividing by 0
        current_particle_gr = current_particle_gr / normalization_constants

        # Append current particle's g(r) to the frame's global g(r)
        instant_gr = pd.concat((instant_gr, pd.DataFrame(current_particle_gr)), axis=1)

    # Mean for all particles in that frame
    instant_gr = instant_gr.mean(axis=1)
    return instant_gr



def radial_distribution_function(data, roi_center=[650, 400], roi_radius=390, min_step=2, particle_diameter=78):
    """ Calculates the radial distribution function, averaging g(r) for all frames.
        All parameters must be in the same units (either pixels, cm or natural units)

    Parameters
    ----------
    data : pandas Dataframe
        It must be have the usual structure with at least the folowing columns:
        ['frame', 'track', 'x', 'y']
    roi_center : int list or tuple
        Coordinates of the central point of the region of interest (in the case of a
        circular ROI)
    roi_radius : int
        Self-explanatory
    min_step : float
        This is the size of the distance step we use to calculate g(r). Reducing this
        will increase the resolution of data (at the cost of higher computing times)
    particle_diameter : float
        Used to express the results in terms of the particle's diameter (useful for plotting)
        Use 1 if no conversion desired

    Returns
    -------
    gr : pandas Dataframe
        A pandas dataframe of shape (N, 2) where N is equal to max_radial_distance/min_step.
        Containing the value of the g(r) for each step. Column names are: ['r', 'gr']
    """
    # The maximum radial distance we can get away from each particle is the distance
    # from the particle to the edge of the ROI. We add a new column with that info
    distances_to_center = distanceToCenter(data.x, data.y, roi_center[0], roi_center[1])
    distances_to_edge = roi_radius - distances_to_center
    data = data.assign(dist_edge=distances_to_edge)

    list_grs = []
    N = 0
    # Looping for each frame and storing the instant g(r) in a dataframe
    for f in set(data.frame):
        printp('Calculating g(r), frame: '+ str(f) + ' / '+ str(len(set(data.frame))))
        gr = instant_radial_distribution_function(data, f, roi_center=roi_center, roi_radius=roi_radius, min_step=min_step)
        list_grs.append(gr)
        N += len(data[data.frame == f]) # To calculate the number of particles in the ROI

    ensemble = pd.concat(list_grs, axis=1)
    # Mean for all frames
    gr = ensemble.mean(axis=1)
    # Firs row's g(r) must be zero (0)
    gr.iloc[0] = 0
    # New column with distances (in terms of particle's diameter)
    distances = pd.DataFrame(np.arange(0, min_step*len(gr), step=min_step) / particle_diameter)
    gr = pd.concat((distances, gr), axis=1)
    # Naming columns
    gr.columns = ['r', 'gr']
    
    # Last normalization
    area = np.pi * (roi_radius**2)
    mean_N = N / len(set(data.frame))
    gr.gr = gr.gr * area / (2* mean_N)
    return gr



def variance_per_particle(vel_data):
    """ Returns the variance (proportional to TEMPERATURE) in both axis for each particle """
    var = []
    for traj in set(vel_data.track):
        sub = vel_data[vel_data.track == traj][['vx', 'vy']]
        v = np.var(sub).values
        var.append([traj, v[0], v[1]])
    var = pd.DataFrame(var, columns=['track', 'var_x', 'var_y'])
    return var


def calculate_fields_annuli(vel_data, n_annuli, frame_interval='all', system_center=(625,375), system_radius=500, particle_diameter=78, is_in_natural_units=False):

    n_frames = len(set(vel_data['frame']))
    area_particle = np.pi * (particle_diameter/2)**2
    # First, we calculate the easiest thing, the starting R of each annulus
    R = np.linspace(start=0, stop=system_radius, num=n_annuli+1)
    # Then, for each detected particle we calculate its distance to the center of the System
    d = np.sqrt((vel_data['x'] - system_center[0])**2 + (vel_data['y'] - system_center[1])**2)
    vel_data['distance_to_center'] = d
    # Then we divide the dataframe into bins using R
    vel_data['R'] = pd.cut(vel_data['distance_to_center'], R, labels=R[:-1])

    if frame_interval=='all':
        groups = vel_data.groupby('R', as_index=False)
    
        fields = []
        delta_R = R[1]
        R_0 = 0
        for R_current_annulus, data_current_annulus in groups:
            #print(R_current_annulus)
            # We now calculate the different thermodynamic fields:
            # -Density:
            area_annulus = np.pi * ((R_current_annulus+delta_R)**2 - R_0**2)
            particles_per_frame = len(data_current_annulus) / n_frames
            density = (particles_per_frame*area_particle) / area_annulus # Packing fraction
            R_0 = R_current_annulus + delta_R
            # -Temperature:
            if is_in_natural_units==True:
                temperature = 0.5*(np.linalg.norm((data_current_annulus[['vx','vy']]), axis=1)**2).mean()
            else:
                temperature = 0.5*((250/particle_diameter)**2)*(np.linalg.norm((data_current_annulus[['vx','vy']]), axis=1)**2).mean()
            # -Flow velocity:
            ux = data_current_annulus['vx'].mean()
            uy = data_current_annulus['vy'].mean()
            # -Angular velocity:
            # First we need to calculate the angle between the radial direction and particle velocity
            d_x = data_current_annulus['x'] - system_center[0]
            d_y = data_current_annulus['y'] - system_center[1]
            vector_d = np.stack((d_x, d_y), axis=1)
            dot_product = np.sum((data_current_annulus[['vx','vy']].values * vector_d), axis=1)
            v_modulus = np.linalg.norm(data_current_annulus[['vx','vy']], axis=1)
            alpha = np.arccos(dot_product / (v_modulus*data_current_annulus['distance_to_center'])) # In radians
            v_t = v_modulus * np.sin(alpha) # Tangential velocity
            v_r = v_modulus * np.cos(alpha)

            angle_d = angle_from_2D_points(d_x, d_y)
            angle_v = angle_from_2D_points(data_current_annulus['vx'].values, data_current_annulus['vy'].values)
            direction = np.sign(angle_v - angle_d) # Direccion de giro (positivo en sentido antihorario)

            v_t = v_t * direction
            w = v_t / data_current_annulus['distance_to_center'] # Angular velocity (omega)
            angular_velocity = w.mean()
            # -Kurtosis:
            k = calculate_kurtosis(data_current_annulus)

            # Append all the calculated fields to the list
            fields.append([R_current_annulus+delta_R, density, temperature, k, ux, uy, v_r.mean(), v_t.mean(), angular_velocity])
    
        fields = pd.DataFrame(fields, columns=['R', 'density', 'temperature', 'kurtosis', 'ux', 'uy', 'u_radial', 'u_tangential', 'angular_velocity'])

    else:
        frame_bins = np.arange(start=0, stop=n_frames+frame_interval, step=frame_interval)
        vel_data['frame_bin'] = pd.cut(vel_data['frame'], frame_bins, labels=frame_bins[:-1])
        frame_groups = vel_data.groupby('frame_bin', as_index=False)

        fields = []
        for f, data_f in frame_groups:
            groups = data_f.groupby('R', as_index=False)
            delta_R = R[1]
            R_0 = 0

            df = frame_interval
            if n_frames - f <= frame_interval:
                df = n_frames - f

            for R_current_annulus, data_current_annulus in groups:
                #print(R_current_annulus)
                # We now calculate the different thermodynamic fields:
                # -Density:
                area_annulus = np.pi * ((R_current_annulus+delta_R)**2 - R_0**2)
                particles_per_frame = len(data_current_annulus) / df # Esto para el ultimo intervalo de fotogramas FALLA (hay menos de frame_interval fotogramas)
                density = (particles_per_frame*area_particle) / area_annulus # Packing fraction
                R_0 = R_current_annulus + delta_R
                # -Temperature:
                if is_in_natural_units==True:
                    temperature = 0.5*(np.linalg.norm((data_current_annulus[['vx','vy']]), axis=1)**2).mean()
                else:
                    temperature = 0.5*((250/particle_diameter)**2)*(np.linalg.norm((data_current_annulus[['vx','vy']]), axis=1)**2).mean()
                # -Flow velocity:
                ux = data_current_annulus['vx'].mean()
                uy = data_current_annulus['vy'].mean()
                # -Angular velocity:
                # First we need to calculate the angle between the radial direction and particle velocity
                d_x = data_current_annulus['x'] - system_center[0]
                d_y = data_current_annulus['y'] - system_center[1]
                vector_d = np.stack((d_x, d_y), axis=1)
                dot_product = np.sum((data_current_annulus[['vx','vy']].values * vector_d), axis=1)
                v_modulus = np.linalg.norm(data_current_annulus[['vx','vy']], axis=1)
                alpha = np.arccos(dot_product / (v_modulus*data_current_annulus['distance_to_center'])) # In radians
                v_t = v_modulus * np.sin(alpha) # Tangential velocity
                v_r = v_modulus * np.cos(alpha)

                angle_d = angle_from_2D_points(d_x, d_y)
                angle_v = angle_from_2D_points(data_current_annulus['vx'].values, data_current_annulus['vy'].values)
                direction = np.sign(angle_v - angle_d) # Direccion de giro (positivo en sentido antihorario)

                v_t = v_t * direction
                w = v_t / data_current_annulus['distance_to_center'] # Angular velocity (omega)
                angular_velocity = w.mean()
                # -Kurtosis:
                k = calculate_kurtosis(data_current_annulus)
        
        
                fields.append([f, R_current_annulus+delta_R, density, temperature, k, ux, uy, v_r.mean(), v_t.mean(), angular_velocity])

        fields = pd.DataFrame(fields, columns=['frame', 'R', 'density', 'temperature', 'kurtosis', 'ux', 'uy', 'u_radial', 'u_tangential', 'angular_velocity'])

    return fields









































def instant_radial_distribution_function_square_region(data, min_step=2):
    """ Calculates the radial distribution function, g(r), for a single frames.
        All parameters must be in the same units (either pixels, cm or natural units)

    Parameters
    ----------
    data : pandas Dataframe
        It must be have the usual structure with at least the folowing columns:
        ['frame', 'track', 'x', 'y']
    frame : int
        Frame number
    roi_center : int list or tuple
        Coordinates of the central point of the region of interest (in the case of a
        circular ROI)
    roi_radius : int
        Self-explanatory
    min_step : float
        This is the size of the distance step we use to calculate g(r). Reducing this
        will increase the resolution of data (at the cost of higher computing times)

    Returns
    -------
    gr : pandas Dataframe
        A pandas dataframe of shape (N, 1) where N is equal to max_radial_distance/min_step.
        Containing the value of the g(r) for each step.
    """


    # First we have to loop over all particles present in that frame and calculate
    # g(r) for every particle (trajectory)
    instant_gr = [] # Empty dataframe to save results
    for index, current_row in data.iterrows():
        # Distances for every particle to current particle
        distances_to_particle = distanceToCenter(data.x, data.y, current_row.x, current_row.y)
        # min_step tells us which is the unit of distance for each step increasing
        # the radial distance from particle's center. Here we create a list with the
        # distances we have to cover

        covered_distances = np.arange(0, current_row['distance_to_closest_wall']+min_step, step=min_step)
        current_particle_gr = np.histogram(distances_to_particle, bins=covered_distances)[0]

        # Normalization
        # # El [:-1] es para que tenga la misma longitud current_particle_gr
        normalization_constants = np.pi * min_step * covered_distances[:-1] # El [:-1] es para que tenga la misma longitud current_particle_gr
        normalization_constants[0] = 1 # To avoid dividing by 0
        current_particle_gr = current_particle_gr / normalization_constants
        # Append current particle's g(r) to the frame's global g(r)
        instant_gr.append(pd.DataFrame(current_particle_gr))

    instant_gr = pd.concat(instant_gr, axis=1)

    # Mean for all particles in that frame
    instant_gr = instant_gr.mean(axis=1)
    return instant_gr



def radial_distribution_function_square_region(data, roi_corner=[250, 50], roi_width=750, roi_height=600, min_step=2, particle_diameter=78):
    from tqdm import tqdm
    """ Calculates the radial distribution function, averaging g(r) for all frames.
        All parameters must be in the same units (either pixels, cm or natural units)

    Parameters
    ----------
    data : pandas Dataframe
        It must be have the usual structure with at least the folowing columns:
        ['frame', 'track', 'x', 'y']
    roi_center : int list or tuple
        Coordinates of the bottom left corner of the ROI
    roi_radius : int
        Self-explanatory
    min_step : float
        This is the size of the distance step we use to calculate g(r). Reducing this
        will increase the resolution of data (at the cost of higher computing times)
    particle_diameter : float
        Used to express the results in terms of the particle's diameter (useful for plotting)
        Use 1 if no conversion desired

    Returns
    -------
    gr : pandas Dataframe
        A pandas dataframe of shape (N, 2) where N is equal to max_radial_distance/min_step.
        Containing the value of the g(r) for each step. Column names are: ['r', 'gr']
    """
    # The maximum radial distance we can get away from each particle is the distance
    # from the particle to the edge of the ROI. We add a new column with that info
    data = data.assign(distances_to_left_roi_wall = (data.x - roi_corner[0]),
                       distances_to_right_roi_wall = (roi_corner[0] + roi_width - data.x),
                       distances_to_top_roi_wall = (roi_corner[1] + roi_height - data.y),
                       distances_to_bottom_roi_wall = (data.y - roi_corner[1]))

    distance_to_closest_wall = np.abs(data.loc[:, ['distances_to_left_roi_wall',
                                                   'distances_to_right_roi_wall',
                                                   'distances_to_top_roi_wall',
                                                   'distances_to_bottom_roi_wall']].min(axis=1))

    data = data.assign(distance_to_closest_wall=distance_to_closest_wall)


    list_grs = []
    N = 0
    n_frames = len(set(data.frame))
    # Looping for each frame and storing the instant g(r) in a dataframe
    for f in tqdm(set(data.frame)):
        printp(f'Calculating g(r), frame: {f} / {n_frames}')
        sub_data = data[data['frame']==f]
        gr = instant_radial_distribution_function_square_region(sub_data, min_step=min_step)
        list_grs.append(gr)
        N += len(data[data.frame == f]) # To calculate the number of particles in the ROI

    ensemble = pd.concat(list_grs, axis=1)
    # Mean for all frames
    gr = ensemble.mean(axis=1)
    # Firs row's g(r) must be zero (0)
    gr.iloc[0] = 0
    # New column with distances (in terms of particle's diameter)
    distances = pd.DataFrame(np.arange(0, min_step*len(gr), step=min_step) / particle_diameter)
    gr = pd.concat((distances, gr), axis=1)
    # Naming columns
    gr.columns = ['r', 'gr']
    
    # Last normalization
    area = roi_width * roi_height
    mean_N = N / len(set(data.frame))
    gr.gr = gr.gr * area / (2* mean_N)
    return gr
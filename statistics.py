# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:34:29 2019

@author: malopez
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from utils import printp


def velocityDistribution(vel_data):
    """ This function returns an histogram plot of the velocity distribution of
        particles """
    # First we have to select only velocities
    vels = vel_data[['vx', 'vy']].values    

# =============================================================================
#     seaborn.set_style('whitegrid')
#     seaborn.kdeplot(vels[:,0], bw=0.5)
# =============================================================================

    histPlot = plt.hist(vels, density=True, bins='auto')
    # Another interesting visualization
#    histPlot2D = plt.hist2d(vel[:,0], vel[:,1], bins=b)
    return histPlot


def computeKurtosis(vel_data):
    """ Better method """
    # First we have to select only velocities, if they are not present in the
    # dataframe an error will be raised
    vels = vel_data[['vx', 'vy']].values
    
    v = np.linalg.norm(vels, axis=1)
    v2 = v**2
    k = (v**4).mean()/((v2).mean())**2
    
    return k


def compute_a2(vel_data, dimensions):
    """ Calculates a2 coefficient for a system of any given dimensions """
    kurtosis = computeKurtosis(vel_data)
    a2 = (dimensions/(dimensions+2))*kurtosis -1
    
    return a2


def theoretical_a2(alpha, d):
    a2 = (16*(1-alpha) * (1 - 2*(alpha**2))) / (9 + 24*d - alpha*(41 - 8*d) + 30*(1-alpha)*(alpha**2))
    
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
    
    sub_data = data[data.track==trajectory]
    sub_data = sub_data[column].values

    if max_steps == 'all':
        max_steps = len(sub_data) - 1
    
    results = []
    for i in range(max_steps-1):
        results.append([i+1, simpleMSD(sub_data, i+1)])
    
    results = np.array(results)
    return results


def simpleMSD(data, lagTime):
    """ Simple MSD function, returns the square displacement given a 1D array
        and a lag time (interval) """
    return np.mean((data[lagTime:] - data[:-lagTime])**2)


def plotMSD(data, timePerFrame=1, max_steps='all', column='x', trajectory=1, log=False):
    """ Plots the mean square displacement for a given track """
    
    MSD = meanSquaredDisplacement(data, max_steps=max_steps, column=column, trajectory=trajectory)
    if log==True:
        dt = np.log(MSD[:,0]*timePerFrame)
        y = np.log(MSD[:,1])
            
        fig, ax = plt.subplots(figsize=(8,8), dpi=250)
        #ax.set_xlim([0,dt.max()])
        #ax.set_ylim([0,y.max()])
    
        ax.set_xlabel('log(t) (s)')
        ax.set_ylabel(r'$log\langle\left( x - x_{0}\right)^{2} \rangle \left( m^{2}\right)$')
        plt.scatter(dt, y, s=1)
    else:       
        dt = MSD[:,0]*timePerFrame
        y = MSD[:,1]
    
        fig, ax = plt.subplots(figsize=(8,8), dpi=250)
        ax.set_xlim([0,dt.max()])
        ax.set_ylim([0,y.max()])
    
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
        if data[data.track==traj].shape[0] > 2:
            msd_traj = meanSquaredDisplacement(data, max_steps=max_steps, column=column, trajectory=traj)
            # I only need the data not the steps columns
            msd_traj = pd.DataFrame(msd_traj[:,1])
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
    
    if log==True: 
        dt = np.arange(len(msd))
        dt = np.log(dt*timePerFrame)
        y = np.log(msd.values)
            
        fig, ax = plt.subplots(figsize=(8,8), dpi=250)
        #ax.set_xlim([0,dt.max()])
        #ax.set_ylim([0,y.max()])
    
        ax.set_xlabel('log(t) (s)')
        ax.set_ylabel(r'$log\langle\left( x - x_{0}\right)^{2} \rangle \left( m^{2}\right)$')
        plt.scatter(dt, y, s=1)
    else:       
        dt = np.arange(len(msd))
        y = msd.values
    
        fig, ax = plt.subplots(figsize=(8,8), dpi=250)
        ax.set_xlim([0,dt.max()])
        ax.set_ylim([0,y.max()])
    
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
    sub_data = vel_data[vel_data.track==trajectory]
    sub_data = sub_data[['vx','vy']]

    if max_lag == 'all':
        max_lag = len(sub_data) - 1
    
    results = []
    for i in range(max_lag-1):
        if i==0:
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
    ensemble = pd.DataFrame()
    for traj in set(vel_data.track):
        if vel_data[vel_data.track==traj].shape[0] > 2:
            vel_aut = velocity_autocorrelation(vel_data, traj, max_lag=max_lag)
            ensemble = pd.concat((ensemble, vel_aut), axis=1)
            printp('Integrating velocity autocorrelations: '+ str(traj) + '/'+ str(len(set(vel_data.track))))
    vac = ensemble.mean(axis=1)
    return vac



def instant_radial_distribution_function(data, frame):
    pass

def radial_distribution_function(data, frame):
    
    ensemble = pd.DataFrame()
    for f in set(data.frame):
        gr = instant_radial_distribution_function(data, f)
        ensemble = pd.concat((ensemble, gr), axis=1)
    pass

# =============================================================================
# def computeExcessKurtosis_a2(kurtosis, dimensions):
#     a2 = (dimensions/(dimensions+2))*kurtosis -1
#     
#     return a2
# =============================================================================

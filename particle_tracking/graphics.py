# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:06:04 2019

@author: malopez
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import trackpy as tp
from utils import select_tracks_by_lenght
from calculateVelocity import calculate_jumps
from graphics_utils import center_spines


def densityPlot(position_data, nbins=40):
    
    # Construct 2D histogram from data using the 'plasma' colormap
    plt.hist2d(x=position_data['x'], y=position_data['y'], bins=nbins, cmap='RdYlBu_r')
    
    # Plot a colorbar with label.
    cb = plt.colorbar()
    cb.set_label('Number of entries')
    
    # Add title and labels to plot.
    plt.title('Heatmap')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    
    # Show the plot.
    plt.show()
    
    
def smooth_densityPlot(position_data, nbins=40):
     
    # create data
    x = position_data['x']
    y = position_data['y']
    
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
     
    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='RdYlBu_r')
    plt.colorbar()
    plt.show()
    
    
def plot_instantaneus_state(trajectories_data, frame=1):
    
    sub_data = trajectories_data[trajectories_data['frame']==frame]
    
    x,y = 5,5
    fig = plt.figure(figsize=(x,y), dpi=350)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
# =============================================================================
#     px_size = particle_size_px/fig.dpi
#     plt.scatter(sub_data.x, sub_data.y, s=(particle_size_px*px_size)**2, edgecolors='k', alpha=0.6)
# =============================================================================
    size = np.sqrt((np.pi/20)*x*y*(fig.dpi**2))
    plt.scatter(sub_data.x, sub_data.y, s=size, edgecolors='k', alpha=0.6)
    
    
def plot_trajectories_by_lenght(data, min_lenght=0, max_lenght=25000):
    selected_tracks = select_tracks_by_lenght(data, min_lenght, max_lenght)
    data = data.rename(columns={'track':'particle'})
    sub_data = data[data['particle'].isin(selected_tracks)]
    tp.plot_traj(sub_data)
    
    
def plot_2D_jumps(data, interval=1, trajectory=1):
    sub_data = data[data.track==trajectory]
    differences = calculate_jumps(sub_data, interval=interval)
    
    fig, ax = plt.subplots()
    center_spines() # Origin as center of axes
    plt.axis('equal')
    ax.tick_params(axis='both', which='major', labelsize=5)
    fig.set_size_inches(4,4)
    fig.set_dpi(350)
    ax.scatter(x=differences.dx, y=differences.dy, s=0.05, alpha=0.5)
    radius = np.sqrt(differences.dx.std()**2 + differences.dy.std()**2)
    c = plt.Circle((differences.dx.mean(), differences.dy.mean()), radius, color='r', fill=False, alpha=0.5)
    ax.add_artist(c)

import os
import json
import pandas as pd
from graphics import densityPlot, plot_2D_jumps
from statistics import plotMSD, computeKurtosis, compute_a2, plot_msd_ensemble


if __name__ == "__main__":
    folder = 'D:/serieDensidad 21-05-2019/'
    files = []
    files += [file for file in os.listdir(folder) if file.endswith('_roi_trajectories.pkl')]
    
    for file in files[2:3]:      
        # Read information:
        experiment_id = file.split('_')[0]
        info_file = os.path.join(folder, experiment_id+'_experiment_info.txt')
        with open(info_file) as f:
            jsonstr = json.load(f)
            
        info = pd.io.json.json_normalize(jsonstr) 


        # --ACTUAL DATA ANALISYS--
      
        # READING DATA
        data = pd.read_pickle(os.path.join(folder, str(experiment_id)+'_raw_data.pkl'), compression='xz')
        roi_traj = pd.read_pickle(os.path.join(folder, str(experiment_id)+'_roi_trajectories.pkl'), compression='xz')
        roi_vels = pd.read_pickle(os.path.join(folder, str(experiment_id)+'_roi_velocities.pkl'), compression='xz')
        

        print('Density plot including all data:')
        densityPlot(data, nbins=80)
        print('Density plot inside ROI:')
        densityPlot(roi_traj, nbins=80)
        
        print('Nº of particles:' + str(info.N[0]))
        print('a_2 coefficient: '+ str(compute_a2(roi_vels, 2)))
        
        reduced_df = roi_traj[roi_traj.track < 700]
        plot_msd_ensemble(reduced_df, max_steps=80)
        plot_2D_jumps(roi_traj, interval=5, trajectory=3)
        

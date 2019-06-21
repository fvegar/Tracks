import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphics import densityPlot, plot_2D_jumps, plot_trajectories_by_lenght
from utils import select_tracks_by_lenght, reset_track_indexes
from stats import plotMSD, compute_kurtosis, compute_a2, plot_msd_ensemble, velocity_autocorrelation_ensemble, variance_per_particle, radial_distribution_function


if __name__ == "__main__":
    folder = 'D:/serieDensidad 21-05-2019/'
    files = []
    files += [file for file in os.listdir(folder) if file.endswith('_roi_trajectories.pkl')]
    
    autocorrelations = []
    variances = pd.DataFrame()
    for file in files:      
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
        

# =============================================================================
#         print('Density plot including all data:')
#         densityPlot(data, nbins=80)
#         print('Density plot inside ROI:')
#         densityPlot(roi_traj, nbins=80)
# =============================================================================
        print(' ')
        print('Nº of particles: ' + str(info.N[0]))
# =============================================================================
#         print('a_2 coefficient: '+ str(compute_a2(roi_vels, 2)))
#         
#         reduced_df = roi_traj[roi_traj.track < 100]
#         plot_msd_ensemble(reduced_df, max_steps=80)
#         plot_2D_jumps(roi_traj, interval=5, trajectory=138)
#         
# =============================================================================
        
        
        
        # --- SELECT TRACKS BY LENGHT AND PLOT THEM ---
        long_traj_tags = select_tracks_by_lenght(roi_vels, min_lenght=1000, max_lenght=25000)
        n_long_traj = len(long_traj_tags)
        long_traj = roi_vels[roi_vels.track.isin(long_traj_tags)]
# =============================================================================
#         if (info.N[0] == 180):
#             plot_trajectories_by_lenght(roi_traj, min_lenght=10000, max_lenght=25000)
# =============================================================================
        
        
        
# =============================================================================
#         # --- CALCULATE AND SAVE VELOCITY AUTOCORRELATIONS (using long traj) ---
#         aut = velocity_autocorrelation_ensemble(long_traj, max_lag=900)
#         pd.to_pickle(aut, os.path.join(folder, str(experiment_id)+'_vel_autocorrelation.pkl'))
# =============================================================================
        
        
# =============================================================================
#         # CALCULATE VARIANCE PER TRACK (only for long trajectories)
#         vpp = variance_per_particle(long_traj)
#         vpp['packing_fraction'] = info.packing_fraction[0]#'{:1.3f}'.format(info.packing_fraction[0])
#         variances = pd.concat((variances, vpp), axis=0)
#         # ----------------------------------------------------------
# =============================================================================
        
        
        
        # CALCULATE g(r) (for a subset of frames)
        bunch_of_frames = long_traj[long_traj.frame <= 25000]
        gr = radial_distribution_function(bunch_of_frames, min_step=2)
        pd.to_pickle(gr, os.path.join(folder, str(experiment_id)+'_gr_res4px.pkl'))
        # ----------------------------------------------------------
  



      
# =============================================================================
# # --- PLOT VARIANCE PER TRACK (only for long trajectories)  ---
# fig, ax = plt.subplots(figsize=(4,4), dpi=350)
# ax.set_xlim(left=0, right=0.4)
# ax.set_ylim(bottom=0, top=0.4)
# scatter = ax.scatter(x=variances['var_x'], y=variances['var_y'], s=1, 
#                      c=variances['packing_fraction'].values, label=variances['packing_fraction'].values)
# ax.plot([0,1], alpha=0.5, lw=0.5, c='black')
# 
# # =============================================================================
# # # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/scatter_with_legend.html
# # legend = ax.legend(*scatter.legend_elements(), loc="best", title="Packing fraction")
# # ax.add_artist(legend)
# # =============================================================================
# # ax.legend(labels=variances.packing_fraction, loc='best')
# variances.plot.scatter(x='var_x', y='var_y', c='packing_fraction')
# # =============================================================================
# # handles, labels = ax.get_legend_handles_labels()
# # handles = [x for _,x in np.unique(zip(labels,handles))] # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
# # labels.sort()
# # ax.legend(labels = np.unique(variances['packing_fraction']), loc='best')
# # =============================================================================
# # NOW COMPUTING THE MEAN FOR EVERY PARTICLE
# var_means = variances.groupby(['packing_fraction']).mean()
# fig, ax = plt.subplots(figsize=(4,4), dpi=350)
# ax.set_xlim(left=0, right=0.4)
# ax.set_ylim(bottom=0, top=0.4)
# scatter = ax.scatter(x=var_means['var_x'], y=var_means['var_y'], s=1, 
#                      c=var_means['track'])
# ax.plot([0,1], alpha=0.5, lw=0.5, c='black')
# # ----------------------------------------------------------        
# =============================================================================
        



# --- READ AND PLOT g(r) ---
plt.style.use('seaborn-notebook')
fig, ax = plt.subplots(figsize=(5,4), dpi=350)
ax.set_xlabel('r (particle diameters)')
ax.set_ylabel('g(r)')

for file in files:      
    # Read information:
    experiment_id = file.split('_')[0]
    info_file = os.path.join(folder, experiment_id+'_experiment_info.txt')
    with open(info_file) as f:
        jsonstr = json.load(f)            
    info = pd.io.json.json_normalize(jsonstr) 
    pf = info.packing_fraction[0]
    # Read g(r) files and create plots 
    gr = pd.read_pickle(os.path.join(folder, str(experiment_id)+'_gr.pkl'))
    lab = 'Packing fraction = ' + '{:1.3f}'.format(pf)
# =============================================================================
#     ax.plot(aut, lw=0.75, alpha=0.9, label=lab, c=next(colors))
# =============================================================================
    ax.plot(gr.r, gr.gr, lw=0.75, alpha=0.9, label=lab)
# Reorder labels from legend and show it    
handles, labels = ax.get_legend_handles_labels() # handles are color lines in the legend, labels well....
handles = [x for _,x in sorted(zip(labels,handles))] # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
labels.sort()
fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.8), prop={'size': 6})



        
# =============================================================================
# # --- READ AND PLOT VELOCITY AUTOCORRELATIONS ---
# plt.style.use('seaborn-notebook')
# fig, ax = plt.subplots(figsize=(7,7), dpi=370)
# ax.set_xlabel('lag time (frame)')
# ax.set_ylabel('VACF')
# ax.set_xlim(left=0, right=800)
# ax.set_ylim(bottom=-1, top=1)
# ax.hlines(y=0, xmin = 0, xmax=1000, colors='black', alpha=0.6, lw=1)
# # =============================================================================
# # # https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
# # colors=iter(plt.cm.tab20(np.linspace(0,1,len(files))))
# # =============================================================================
# for file in files:      
#     # Read information:
#     experiment_id = file.split('_')[0]
#     info_file = os.path.join(folder, experiment_id+'_experiment_info.txt')
#     with open(info_file) as f:
#         jsonstr = json.load(f)            
#     info = pd.io.json.json_normalize(jsonstr) 
#     pf = info.packing_fraction[0]
#     # Read autocorrelation files and create plots 
#     aut = pd.read_pickle(os.path.join(folder, str(experiment_id)+'_vel_autocorrelation.pkl'))
#     lab = 'Packing fraction = ' + '{:1.3f}'.format(pf)
# # =============================================================================
# #     ax.plot(aut, lw=0.75, alpha=0.9, label=lab, c=next(colors))
# # =============================================================================
#     ax.plot(aut, lw=0.75, alpha=0.9, label=lab)
# # Reorder labels from legend and show it    
# handles, labels = ax.get_legend_handles_labels() # handles are color lines in the legend, labels well....
# handles = [x for _,x in sorted(zip(labels,handles))] # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
# labels.sort()
# fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.8), prop={'size': 6})
# # ------------------------------------------------------------------------------
# 
# =============================================================================

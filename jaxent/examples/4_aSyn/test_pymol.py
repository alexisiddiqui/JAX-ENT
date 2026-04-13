import pymol
from pymol import cmd

cmd.load('data/_cluster_aSyn/clusters/all_clusters.xtc', 'aSyn_ensemble')
print("States:", cmd.count_states('aSyn_ensemble'))

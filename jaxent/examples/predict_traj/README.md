
ValDXer trajectories can be found in:
https://drive.google.com/drive/folders/1Y9294af-SLca80Xk4D2gmg460NXlt7wX?usp=sharing

please pick the most upto date directory within this - I would advise against downloading the entire directory.


Short, 500 frame MD simulations can be obtained from:
https://drive.google.com/drive/folders/1ijiwctie_GGEDQrNuz5IKcVbz1WCR1FD?usp=sharing

We provide methods to cluster the trajectories

# Note: when running CLI options, please ensure that any other environments (conda etc) are deactivate fully.

# Examples:

## Predict entire directory of trajectories

./run_predict_dir.sh [dir_name]
- to change this to predict uptake, provide the time points in the bash scripts
## Predict on clustered trajectories

./run_predict_dir_cluster.sh [dir_name] [num_clusters] [num_pca_components]
- to change this to predict uptake, provide the time points in the bash scripts
This module contains pymol scripts used for visualisation

There are 3 main scripts:

1. RMSD by res

2. Top N frames

3. Project Protection Factors


As pymol can not be installed by pip, conda environments (PYMOL_ANA) are used to run these scripts.
For development purposes, the interpreter has been set to this environment. This environment includes pymol and jaxent.

These scripts are operated using yaml config but can also be run from the command line using argparse. The config options are unified across the scripts, so that they can be easily combined and extended in the future.

Current non extendable references are in: jaxent/src/analysis/pymol/_reference

Create example yaml configs in jaxent/src/analysis/pymol/test_configs/

# Config

The unified config options are as follows:

## Render options:
--spectrum_colours: string of comma-separated colours to use for the spectrum (e.g. "blue,white,red") or a predefined colour scheme (e.g. "blue_white_red")
--spectrum_range: string of comma-separated values defining the range for the spectrum (e.g. "0,1" or "0,100")
--putty_transform: integer defining the putty transform to use (0-8, https://pymolwiki.org/Cartoon_putty_transform)
--putty_range: string of comma-separated values defining the range for the putty transform (e.g. "0,1" or "0,100")
--reference_transparency: float between 0 and 1 defining the transparency to use for reference structures (0 = opaque, 1 = fully transparent)
--trajectory_transparency: float between 0 and 1 defining the transparency to use for trajectory frames (0 = opaque, 1 = fully transparent)
--other_transparency: float between 0 and 1 defining the transparency to use for other objects (e.g. axes, labels) (0 = opaque, 1 = fully transparent)
--transparency_mode: integer defining the transparency mode to use unilayer/multilayer (0-3, https://pymolwiki.org/Transparency_mode)
--orthoscopic_view: boolean defining whether to use orthoscopic view (true) or perspective view (false)
--antialias: integer defining the level of antialiasing to use (0-2, https://pymolwiki.org/Antialias)
--ray_trace_mode: integer defining the ray trace mode to use (0-3, https://pymolwiki.org/Ray)
--ray_transparency_oblique: boolean defining whether to use angle dependent transparency when ray tracing (true) or not (false)
--view: matrix of 16 comma-separated values defining the viewport to use (e.g. "1,0,0,0,0,1,0,0,0,0,1,0,0,0,-100,1"), default (None) will orient the view on the scene.


## New options to add
--ray_trace_disco_factor: float defining appearance of lines ray tracing (0-1, https://pymolwiki.org/Ray#Ray_trace_disco_factor)
--ray_trace_gain: float defining the thickness of lines when ray tracing mode=1 (0-10+, https://pymolwiki.org/Ray#Ray_trace_gain)




## General options:
--references: string of comma-separated paths to reference structures (e.g. "ref1.pdb,ref2.pdb")
--reference_labels: string of comma-separated labels for reference structures (e.g. "Reference-1,Reference-2")
--reference_colors: string of comma-separated colours (names or [R,G,B] values) for reference structures (e.g. "red,blue" or "1,0,0;0,0,1")
--trajectory_label: string defining the label for the trajectory/target (e.g. "Trajectory")
--align_atoms: string of comma-separated atom selections for alignment (e.g. "name CA")
--align_selection: string defining the residue selection for alignment (e.g. "resid 1-100")
--working_dir: path to working directory - if none provided, the script will assume absolute paths to inputs and uses the current working directory to save outputs.


# Scripts:

RMSD by res: This is a direct adaptation of the original RMSD_by_res script - modified to share unified config options.
--trajectory: path to trajectory file and topology file (e.g. "trajectory.xtc,topology.pdb") or a multiframe structure file (e.g. "trajectory.pdb")
--trajectory_label: string defining the label for the trajectory frames (e.g. "Trajectory")


Top N frames: This script is a combination of Tea_weighted_SASA_test.py and MoPrP_weighted_SASA_test.py - loads in a trajectory and references and identifies the top N frames closest to each reference. The top N frames are then aligned to the reference and visualised in pymol, coloured either by RMSD to a given reference or by the average frame weight assigned to the frame.

--trajectory: path to trajectory file and topology file (e.g. "trajectory.xtc,topology.pdb") or a multiframe structure file (e.g. "trajectory.pdb")
--weights: path to numpy file containing per-replciate frame weights (e.g. "frame_weights.npz"), shape: (n_replicates, n_frames)
--metric: string defining the metric to use for identifying the top N frames (e.g. "RMSD" or "weight")
--top_n: integer defining the number of top frames to identify and visualise for each reference (e.g. 5 or 20)

Project Protection Factors: This script projects per-replicate protection factors onto a reference structure, colouring the structure by the replicate-average protection factor values. By providing reference protection factors, the script also visualises: Uncertainty (SD and RSD%) and the difference between the replicate-average and reference protection factors (singed and absolute).
--reference_data: path to text file reference protection factor data (e.g. "reference_protection_factors.dat"), columns: residue, protection factor
--target_data: path to numpy file containing per-replicate protection factor data (e.g. "replicate_protection_factors.npy"), shape: (n_replicates, n_residues) - excludes Prolines and termini residues
--target_topology: path to topology file (e.g. "topology.pdb") containing the full sequence of the protein, including Prolines and termini residues, for mapping residue numbers to the reference structure
--metric: string defining the metric to use for colouring the structure (e.g. "protection_factor", "uncertainty_sd", "uncertainty_rsd", "difference_signed", "difference_absolute")
import copy
import os
import tempfile
from unittest.mock import MagicMock, patch

import MDAnalysis
import numpy as np
import pytest

from jaxent.src.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology.factory import TopologyFactory
from jaxent.src.interfaces.topology.pairwise import PairwiseTopologyComparisons

test_pdb = """"TITLE     MDANALYSIS FRAME 0: Created by PDBWriter
CRYST1   63.649   63.649   63.649  90.00  90.00  90.00 P 1           1
ATOM      1  N   ARG X   1      16.258  29.576  29.578  1.00  0.00      SYST N  
ATOM      2  CA  ARG X   1      17.443  28.946  30.259  1.00  0.00      SYST C  
ATOM      3  HA  ARG X   1      18.348  29.493  29.995  1.00  0.00      SYST H  
ATOM      4  CB  ARG X   1      17.367  28.956  31.812  1.00  0.00      SYST C  
ATOM      5  CG  ARG X   1      18.568  28.539  32.669  1.00  0.00      SYST C  
ATOM      6  CD  ARG X   1      18.271  28.623  34.182  1.00  0.00      SYST C  
ATOM      7  NE  ARG X   1      18.000  29.977  34.729  1.00  0.00      SYST N  
ATOM      8  CZ  ARG X   1      16.963  30.323  35.449  1.00  0.00      SYST C  
ATOM      9  NH1 ARG X   1      16.003  29.521  35.793  1.00  0.00      SYST N  
ATOM     10  NH2 ARG X   1      16.999  31.555  35.859  1.00  0.00      SYST N  
ATOM     11  C   ARG X   1      17.842  27.584  29.753  1.00  0.00      SYST C  
ATOM     12  O   ARG X   1      17.317  26.609  30.336  1.00  0.00      SYST O  
ATOM     13  N   PRO X   2      18.624  27.417  28.715  1.00  0.00      SYST N  
ATOM     14  CD  PRO X   2      18.966  28.481  27.841  1.00  0.00      SYST C  
ATOM     15  CG  PRO X   2      19.357  27.772  26.492  1.00  0.00      SYST C  
ATOM     16  CB  PRO X   2      19.851  26.361  26.848  1.00  0.00      SYST C  
ATOM     17  CA  PRO X   2      19.234  26.146  28.220  1.00  0.00      SYST C  
ATOM     18  HA  PRO X   2      18.447  25.416  28.032  1.00  0.00      SYST H  
ATOM     19  C   PRO X   2      20.205  25.353  29.253  1.00  0.00      SYST C  
ATOM     20  O   PRO X   2      20.590  25.948  30.285  1.00  0.00      SYST O  
ATOM     21  N   ASP X   3      20.546  24.124  29.001  1.00  0.00      SYST N  
ATOM     22  H   ASP X   3      20.163  23.604  28.224  1.00  0.00      SYST H  
ATOM     23  CA  ASP X   3      21.181  23.301  30.074  1.00  0.00      SYST C  
ATOM     24  HA  ASP X   3      20.495  23.442  30.909  1.00  0.00      SYST H  
ATOM     25  CB  ASP X   3      21.271  21.804  29.846  1.00  0.00      SYST C  
ATOM     26  CG  ASP X   3      19.854  21.147  29.864  1.00  0.00      SYST C  
ATOM     27  OD1 ASP X   3      18.783  21.786  30.084  1.00  0.00      SYST O  
ATOM     28  OD2 ASP X   3      19.839  19.937  29.768  1.00  0.00      SYST O  
ATOM     29  C   ASP X   3      22.602  23.851  30.417  1.00  0.00      SYST C  
ATOM     30  O   ASP X   3      22.993  23.685  31.567  1.00  0.00      SYST O  
ATOM     31  N   PHE X   4      23.360  24.436  29.457  1.00  0.00      SYST N  
ATOM     32  H   PHE X   4      23.039  24.402  28.500  1.00  0.00      SYST H  
ATOM     33  CA  PHE X   4      24.697  24.987  29.702  1.00  0.00      SYST C  
ATOM     34  HA  PHE X   4      25.304  24.205  30.159  1.00  0.00      SYST H  
ATOM     35  CB  PHE X   4      25.461  25.443  28.465  1.00  0.00      SYST C  
ATOM     36  CG  PHE X   4      24.881  26.596  27.732  1.00  0.00      SYST C  
ATOM     37  CD1 PHE X   4      23.835  26.539  26.758  1.00  0.00      SYST C  
ATOM     38  CE1 PHE X   4      23.280  27.689  26.217  1.00  0.00      SYST C  
ATOM     39  CZ  PHE X   4      23.752  28.933  26.642  1.00  0.00      SYST C  
ATOM     40  CE2 PHE X   4      24.800  29.045  27.503  1.00  0.00      SYST C  
ATOM     41  CD2 PHE X   4      25.360  27.894  28.086  1.00  0.00      SYST C  
ATOM     42  C   PHE X   4      24.793  25.981  30.819  1.00  0.00      SYST C  
ATOM     43  O   PHE X   4      25.910  26.060  31.368  1.00  0.00      SYST O  
ATOM     44  N   CYS X   5      23.848  26.866  30.957  1.00  0.00      SYST N  
ATOM     45  H   CYS X   5      22.979  26.815  30.445  1.00  0.00      SYST H  
ATOM     46  CA  CYS X   5      23.780  28.016  31.900  1.00  0.00      SYST C  
ATOM     47  HA  CYS X   5      24.533  28.763  31.652  1.00  0.00      SYST H  
ATOM     48  CB  CYS X   5      22.359  28.608  31.879  1.00  0.00      SYST C  
ATOM     49  SG  CYS X   5      21.714  29.344  30.294  1.00  0.00      SYST S  
ATOM     50  C   CYS X   5      23.908  27.559  33.361  1.00  0.00      SYST C  
ATOM     51  O   CYS X   5      24.333  28.339  34.183  1.00  0.00      SYST O  
ATOM     52  N   LEU X   6      23.652  26.267  33.732  1.00  0.00      SYST N  
ATOM     53  H   LEU X   6      23.487  25.632  32.964  1.00  0.00      SYST H  
ATOM     54  CA  LEU X   6      23.725  25.783  35.096  1.00  0.00      SYST C  
ATOM     55  HA  LEU X   6      23.502  26.643  35.727  1.00  0.00      SYST H  
ATOM     56  CB  LEU X   6      22.684  24.606  35.311  1.00  0.00      SYST C  
ATOM     57  CG  LEU X   6      21.197  24.929  35.505  1.00  0.00      SYST C  
ATOM     58  CD1 LEU X   6      20.961  25.856  36.680  1.00  0.00      SYST C  
ATOM     59  CD2 LEU X   6      20.650  25.737  34.329  1.00  0.00      SYST C  
ATOM     60  C   LEU X   6      25.182  25.415  35.483  1.00  0.00      SYST C  
ATOM     61  O   LEU X   6      25.428  25.041  36.644  1.00  0.00      SYST O  
ATOM     62  N   GLU X   7      26.197  25.480  34.612  1.00  0.00      SYST N  
ATOM     63  H   GLU X   7      26.122  26.172  33.880  1.00  0.00      SYST H  
ATOM     64  CA  GLU X   7      27.589  24.995  34.884  1.00  0.00      SYST C  
ATOM     65  HA  GLU X   7      27.649  24.474  35.840  1.00  0.00      SYST H  
ATOM     66  CB  GLU X   7      28.050  23.994  33.791  1.00  0.00      SYST C  
ATOM     67  CG  GLU X   7      27.179  22.670  33.728  1.00  0.00      SYST C  
ATOM     68  CD  GLU X   7      27.381  21.774  34.962  1.00  0.00      SYST C  
ATOM     69  OE1 GLU X   7      28.204  22.055  35.877  1.00  0.00      SYST O  
ATOM     70  OE2 GLU X   7      26.656  20.751  35.010  1.00  0.00      SYST O  
ATOM     71  C   GLU X   7      28.646  26.188  34.965  1.00  0.00      SYST C  
ATOM     72  O   GLU X   7      28.448  27.201  34.270  1.00  0.00      SYST O  
ATOM     73  N   PRO X   8      29.759  26.018  35.752  1.00  0.00      SYST N  
ATOM     74  CD  PRO X   8      30.053  25.004  36.748  1.00  0.00      SYST C  
ATOM     75  CG  PRO X   8      30.897  25.705  37.831  1.00  0.00      SYST C  
ATOM     76  CB  PRO X   8      31.742  26.603  36.927  1.00  0.00      SYST C  
ATOM     77  CA  PRO X   8      30.732  27.096  35.822  1.00  0.00      SYST C  
ATOM     78  HA  PRO X   8      30.248  27.976  36.245  1.00  0.00      SYST H  
ATOM     79  C   PRO X   8      31.438  27.488  34.435  1.00  0.00      SYST C  
ATOM     80  O   PRO X   8      31.478  26.747  33.450  1.00  0.00      SYST O  
ATOM     81  N   PRO X   9      32.006  28.664  34.390  1.00  0.00      SYST N  
ATOM     82  CD  PRO X   9      32.277  29.526  35.566  1.00  0.00      SYST C  
ATOM     83  CG  PRO X   9      33.130  30.707  35.061  1.00  0.00      SYST C  
ATOM     84  CB  PRO X   9      32.872  30.671  33.556  1.00  0.00      SYST C  
ATOM     85  CA  PRO X   9      32.671  29.215  33.176  1.00  0.00      SYST C  
ATOM     86  HA  PRO X   9      32.062  29.077  32.283  1.00  0.00      SYST H  
ATOM     87  C   PRO X   9      34.047  28.531  33.015  1.00  0.00      SYST C  
ATOM     88  O   PRO X   9      34.544  27.835  33.930  1.00  0.00      SYST O  
ATOM     89  N   TYR X  10      34.633  28.631  31.833  1.00  0.00      SYST N  
ATOM     90  H   TYR X  10      34.234  29.289  31.178  1.00  0.00      SYST H  
ATOM     91  CA  TYR X  10      35.802  27.884  31.409  1.00  0.00      SYST C  
ATOM     92  HA  TYR X  10      36.147  27.487  32.364  1.00  0.00      SYST H  
ATOM     93  CB  TYR X  10      35.370  26.718  30.444  1.00  0.00      SYST C  
ATOM     94  CG  TYR X  10      36.390  25.731  30.117  1.00  0.00      SYST C  
ATOM     95  CD1 TYR X  10      36.967  25.626  28.867  1.00  0.00      SYST C  
ATOM     96  CE1 TYR X  10      37.896  24.616  28.469  1.00  0.00      SYST C  
ATOM     97  CZ  TYR X  10      38.278  23.738  29.533  1.00  0.00      SYST C  
ATOM     98  OH  TYR X  10      39.134  22.746  29.239  1.00  0.00      SYST O  
ATOM     99  CE2 TYR X  10      37.723  23.844  30.823  1.00  0.00      SYST C  
ATOM    100  CD2 TYR X  10      36.743  24.808  31.063  1.00  0.00      SYST C  
ATOM    101  C   TYR X  10      36.858  28.742  30.662  1.00  0.00      SYST C  
ATOM    102  O   TYR X  10      36.552  29.530  29.738  1.00  0.00      SYST O  
ATOM    103  N   THR X  11      38.069  28.661  31.011  1.00  0.00      SYST N  
ATOM    104  H   THR X  11      38.304  28.092  31.812  1.00  0.00      SYST H  
ATOM    105  CA  THR X  11      39.195  29.366  30.440  1.00  0.00      SYST C  
ATOM    106  HA  THR X  11      38.961  30.056  29.629  1.00  0.00      SYST H  
ATOM    107  CB  THR X  11      39.904  30.255  31.522  1.00  0.00      SYST C  
ATOM    108  CG2 THR X  11      41.139  31.079  31.120  1.00  0.00      SYST C  
ATOM    109  OG1 THR X  11      39.080  31.185  32.112  1.00  0.00      SYST O  
ATOM    110  C   THR X  11      40.247  28.415  29.895  1.00  0.00      SYST C  
ATOM    111  O   THR X  11      40.934  28.872  28.964  1.00  0.00      SYST O  
ATOM    112  N   GLY X  12      40.355  27.150  30.357  1.00  0.00      SYST N  
ATOM    113  H   GLY X  12      39.877  26.835  31.189  1.00  0.00      SYST H  
ATOM    114  CA  GLY X  12      41.497  26.349  29.913  1.00  0.00      SYST C  
ATOM    115  C   GLY X  12      42.892  26.871  30.313  1.00  0.00      SYST C  
ATOM    116  O   GLY X  12      43.003  27.905  30.951  1.00  0.00      SYST O  
ATOM    117  N   PRO X  13      43.982  26.303  29.736  1.00  0.00      SYST N  
ATOM    118  CD  PRO X  13      43.904  25.027  28.961  1.00  0.00      SYST C  
ATOM    119  CG  PRO X  13      45.269  24.419  29.133  1.00  0.00      SYST C  
ATOM    120  CB  PRO X  13      46.181  25.631  29.032  1.00  0.00      SYST C  
ATOM    121  CA  PRO X  13      45.397  26.737  29.754  1.00  0.00      SYST C  
ATOM    122  HA  PRO X  13      45.580  26.763  30.828  1.00  0.00      SYST H  
ATOM    123  C   PRO X  13      45.744  28.150  29.131  1.00  0.00      SYST C  
ATOM    124  O   PRO X  13      46.922  28.422  28.847  1.00  0.00      SYST O  
ATOM    125  N   CYS X  14      44.758  28.967  28.822  1.00  0.00      SYST N  
ATOM    126  H   CYS X  14      43.792  28.755  29.027  1.00  0.00      SYST H  
ATOM    127  CA  CYS X  14      45.003  30.295  28.271  1.00  0.00      SYST C  
ATOM    128  HA  CYS X  14      45.792  30.285  27.519  1.00  0.00      SYST H  
ATOM    129  CB  CYS X  14      43.687  30.819  27.624  1.00  0.00      SYST C  
ATOM    130  SG  CYS X  14      44.146  32.246  26.695  1.00  0.00      SYST S  
ATOM    131  C   CYS X  14      45.399  31.294  29.354  1.00  0.00      SYST C  
ATOM    132  O   CYS X  14      44.788  31.445  30.407  1.00  0.00      SYST O  
ATOM    133  N   LYS X  15      46.398  32.178  29.117  1.00  0.00      SYST N  
ATOM    134  H   LYS X  15      46.857  32.180  28.217  1.00  0.00      SYST H  
ATOM    135  CA  LYS X  15      46.992  33.067  30.157  1.00  0.00      SYST C  
ATOM    136  HA  LYS X  15      46.655  32.706  31.129  1.00  0.00      SYST H  
ATOM    137  CB  LYS X  15      48.533  33.077  30.022  1.00  0.00      SYST C  
ATOM    138  CG  LYS X  15      49.133  31.729  30.244  1.00  0.00      SYST C  
ATOM    139  CD  LYS X  15      50.543  31.677  30.901  1.00  0.00      SYST C  
ATOM    140  CE  LYS X  15      51.552  32.397  30.038  1.00  0.00      SYST C  
ATOM    141  NZ  LYS X  15      52.829  32.782  30.701  1.00  0.00      SYST N  
ATOM    142  C   LYS X  15      46.364  34.507  29.982  1.00  0.00      SYST C  
ATOM    143  O   LYS X  15      47.088  35.518  29.955  1.00  0.00      SYST O  
ATOM    144  N   ALA X  16      45.106  34.528  29.669  1.00  0.00      SYST N  
ATOM    145  H   ALA X  16      44.524  33.736  29.901  1.00  0.00      SYST H  
ATOM    146  CA  ALA X  16      44.369  35.680  29.101  1.00  0.00      SYST C  
ATOM    147  HA  ALA X  16      45.054  36.054  28.340  1.00  0.00      SYST H  
ATOM    148  CB  ALA X  16      42.983  35.136  28.603  1.00  0.00      SYST C  
ATOM    149  C   ALA X  16      44.024  36.796  30.057  1.00  0.00      SYST C  
ATOM    150  O   ALA X  16      43.931  36.554  31.185  1.00  0.00      SYST O  
ATOM    151  N   ARG X  17      43.601  37.974  29.565  1.00  0.00      SYST N  
ATOM    152  H   ARG X  17      43.551  38.025  28.557  1.00  0.00      SYST H  
ATOM    153  CA  ARG X  17      43.228  39.186  30.328  1.00  0.00      SYST C  
ATOM    154  HA  ARG X  17      42.996  38.818  31.328  1.00  0.00      SYST H  
ATOM    155  CB  ARG X  17      44.363  40.201  30.615  1.00  0.00      SYST C  
ATOM    156  CG  ARG X  17      44.937  40.989  29.454  1.00  0.00      SYST C  
ATOM    157  CD  ARG X  17      45.062  42.492  29.546  1.00  0.00      SYST C  
ATOM    158  NE  ARG X  17      43.714  43.063  29.847  1.00  0.00      SYST N  
ATOM    159  CZ  ARG X  17      42.776  43.432  28.975  1.00  0.00      SYST C  
ATOM    160  NH1 ARG X  17      43.037  43.587  27.746  1.00  0.00      SYST N  
ATOM    161  NH2 ARG X  17      41.610  43.664  29.420  1.00  0.00      SYST N  
ATOM    162  C   ARG X  17      41.932  39.859  29.875  1.00  0.00      SYST C  
ATOM    163  O   ARG X  17      41.346  40.648  30.614  1.00  0.00      SYST O  
ATOM    164  N   ILE X  18      41.372  39.521  28.646  1.00  0.00      SYST N  
ATOM    165  H   ILE X  18      41.891  38.897  28.043  1.00  0.00      SYST H  
ATOM    166  CA  ILE X  18      40.010  40.010  28.142  1.00  0.00      SYST C  
ATOM    167  HA  ILE X  18      39.975  41.096  28.231  1.00  0.00      SYST H  
ATOM    168  CB  ILE X  18      39.692  39.771  26.629  1.00  0.00      SYST C  
ATOM    169  CG2 ILE X  18      38.187  40.073  26.402  1.00  0.00      SYST C  
ATOM    170  CG1 ILE X  18      40.571  40.506  25.598  1.00  0.00      SYST C  
ATOM    171  CD  ILE X  18      40.357  40.201  24.109  1.00  0.00      SYST C  
ATOM    172  C   ILE X  18      38.922  39.497  29.125  1.00  0.00      SYST C  
ATOM    173  O   ILE X  18      39.033  38.352  29.489  1.00  0.00      SYST O  
ATOM    174  N   ILE X  19      37.950  40.318  29.504  1.00  0.00      SYST N  
ATOM    175  H   ILE X  19      37.984  41.280  29.201  1.00  0.00      SYST H  
ATOM    176  CA  ILE X  19      36.823  40.007  30.483  1.00  0.00      SYST C  
ATOM    177  HA  ILE X  19      37.088  39.081  30.992  1.00  0.00      SYST H  
ATOM    178  CB  ILE X  19      36.793  41.021  31.603  1.00  0.00      SYST C  
ATOM    179  CG2 ILE X  19      35.503  40.785  32.418  1.00  0.00      SYST C  
ATOM    180  CG1 ILE X  19      38.056  40.963  32.546  1.00  0.00      SYST C  
ATOM    181  CD  ILE X  19      38.328  42.216  33.418  1.00  0.00      SYST C  
ATOM    182  C   ILE X  19      35.449  39.780  29.654  1.00  0.00      SYST C  
ATOM    183  O   ILE X  19      35.098  40.503  28.725  1.00  0.00      SYST O  
ATOM    184  N   ARG X  20      34.597  38.843  30.072  1.00  0.00      SYST N  
ATOM    185  H   ARG X  20      34.812  38.212  30.831  1.00  0.00      SYST H  
ATOM    186  CA  ARG X  20      33.288  38.590  29.459  1.00  0.00      SYST C  
ATOM    187  HA  ARG X  20      32.885  39.464  28.948  1.00  0.00      SYST H  
ATOM    188  CB  ARG X  20      33.327  37.432  28.410  1.00  0.00      SYST C  
ATOM    189  CG  ARG X  20      33.895  37.756  27.015  1.00  0.00      SYST C  
ATOM    190  CD  ARG X  20      33.432  36.637  26.031  1.00  0.00      SYST C  
ATOM    191  NE  ARG X  20      33.738  37.054  24.674  1.00  0.00      SYST N  
ATOM    192  CZ  ARG X  20      33.752  36.319  23.626  1.00  0.00      SYST C  
ATOM    193  NH1 ARG X  20      33.415  35.046  23.592  1.00  0.00      SYST N  
ATOM    194  NH2 ARG X  20      34.189  36.871  22.517  1.00  0.00      SYST N  
ATOM    195  C   ARG X  20      32.280  38.186  30.528  1.00  0.00      SYST C  
ATOM    196  O   ARG X  20      32.664  37.727  31.585  1.00  0.00      SYST O  
ATOM    197  N   TYR X  21      30.965  38.297  30.257  1.00  0.00      SYST N  
ATOM    198  H   TYR X  21      30.686  38.695  29.372  1.00  0.00      SYST H  
ATOM    199  CA  TYR X  21      29.886  37.949  31.169  1.00  0.00      SYST C  
ATOM    200  HA  TYR X  21      30.140  38.341  32.154  1.00  0.00      SYST H  
ATOM    201  CB  TYR X  21      28.613  38.780  30.794  1.00  0.00      SYST C  
ATOM    202  CG  TYR X  21      28.936  40.312  30.706  1.00  0.00      SYST C  
ATOM    203  CD1 TYR X  21      28.870  41.179  31.870  1.00  0.00      SYST C  
ATOM    204  CE1 TYR X  21      29.172  42.560  31.775  1.00  0.00      SYST C  
ATOM    205  CZ  TYR X  21      29.567  43.067  30.538  1.00  0.00      SYST C  
ATOM    206  OH  TYR X  21      29.758  44.449  30.396  1.00  0.00      SYST O  
ATOM    207  CE2 TYR X  21      29.472  42.261  29.363  1.00  0.00      SYST C  
ATOM    208  CD2 TYR X  21      29.214  40.950  29.443  1.00  0.00      SYST C  
ATOM    209  C   TYR X  21      29.649  36.407  31.173  1.00  0.00      SYST C  
ATOM    210  O   TYR X  21      30.091  35.699  30.235  1.00  0.00      SYST O  
ATOM    211  N   PHE X  22      29.103  35.753  32.196  1.00  0.00      SYST N  
ATOM    212  H   PHE X  22      28.934  36.291  33.034  1.00  0.00      SYST H  
ATOM    213  CA  PHE X  22      28.782  34.270  32.294  1.00  0.00      SYST C  
ATOM    214  HA  PHE X  22      28.416  33.891  31.340  1.00  0.00      SYST H  
ATOM    215  CB  PHE X  22      30.019  33.419  32.703  1.00  0.00      SYST C  
ATOM    216  CG  PHE X  22      30.307  33.299  34.211  1.00  0.00      SYST C  
ATOM    217  CD1 PHE X  22      31.261  34.207  34.814  1.00  0.00      SYST C  
ATOM    218  CE1 PHE X  22      31.488  34.119  36.209  1.00  0.00      SYST C  
ATOM    219  CZ  PHE X  22      30.885  33.127  36.982  1.00  0.00      SYST C  
ATOM    220  CE2 PHE X  22      29.946  32.192  36.415  1.00  0.00      SYST C  
ATOM    221  CD2 PHE X  22      29.691  32.283  35.018  1.00  0.00      SYST C  
ATOM    222  C   PHE X  22      27.655  34.184  33.397  1.00  0.00      SYST C  
ATOM    223  O   PHE X  22      27.514  35.165  34.115  1.00  0.00      SYST O  
ATOM    224  N   TYR X  23      26.776  33.155  33.504  1.00  0.00      SYST N  
ATOM    225  H   TYR X  23      26.945  32.385  32.873  1.00  0.00      SYST H  
ATOM    226  CA  TYR X  23      25.806  33.009  34.577  1.00  0.00      SYST C  
ATOM    227  HA  TYR X  23      25.559  34.000  34.959  1.00  0.00      SYST H  
ATOM    228  CB  TYR X  23      24.497  32.425  34.030  1.00  0.00      SYST C  
ATOM    229  CG  TYR X  23      23.376  32.327  35.071  1.00  0.00      SYST C  
ATOM    230  CD1 TYR X  23      23.021  31.122  35.780  1.00  0.00      SYST C  
ATOM    231  CE1 TYR X  23      21.812  31.114  36.550  1.00  0.00      SYST C  
ATOM    232  CZ  TYR X  23      20.986  32.271  36.550  1.00  0.00      SYST C  
ATOM    233  OH  TYR X  23      19.786  32.201  37.278  1.00  0.00      SYST O  
ATOM    234  CE2 TYR X  23      21.393  33.439  35.943  1.00  0.00      SYST C  
ATOM    235  CD2 TYR X  23      22.537  33.458  35.109  1.00  0.00      SYST C  
ATOM    236  C   TYR X  23      26.427  32.190  35.737  1.00  0.00      SYST C  
ATOM    237  O   TYR X  23      26.680  30.995  35.545  1.00  0.00      SYST O  
ATOM    238  N   ASN X  24      26.558  32.766  36.924  1.00  0.00      SYST N  
ATOM    239  H   ASN X  24      26.325  33.746  36.996  1.00  0.00      SYST H  
ATOM    240  CA  ASN X  24      26.889  31.973  38.109  1.00  0.00      SYST C  
ATOM    241  HA  ASN X  24      27.572  31.149  37.903  1.00  0.00      SYST H  
ATOM    242  CB  ASN X  24      27.639  32.930  39.056  1.00  0.00      SYST C  
ATOM    243  CG  ASN X  24      28.120  32.280  40.400  1.00  0.00      SYST C  
ATOM    244  OD1 ASN X  24      27.521  31.328  40.893  1.00  0.00      SYST O  
ATOM    245  ND2 ASN X  24      29.163  32.840  40.999  1.00  0.00      SYST N  
ATOM    246  C   ASN X  24      25.595  31.384  38.744  1.00  0.00      SYST C  
ATOM    247  O   ASN X  24      24.736  32.148  39.278  1.00  0.00      SYST O  
ATOM    248  N   ALA X  25      25.297  30.076  38.638  1.00  0.00      SYST N  
ATOM    249  H   ALA X  25      26.038  29.525  38.228  1.00  0.00      SYST H  
ATOM    250  CA  ALA X  25      24.025  29.441  39.017  1.00  0.00      SYST C  
ATOM    251  HA  ALA X  25      23.247  30.013  38.511  1.00  0.00      SYST H  
ATOM    252  CB  ALA X  25      23.949  28.014  38.554  1.00  0.00      SYST C  
ATOM    253  C   ALA X  25      23.822  29.555  40.526  1.00  0.00      SYST C  
ATOM    254  O   ALA X  25      22.700  29.903  40.853  1.00  0.00      SYST O  
ATOM    255  N   LYS X  26      24.850  29.453  41.426  1.00  0.00      SYST N  
ATOM    256  H   LYS X  26      25.800  29.483  41.085  1.00  0.00      SYST H  
ATOM    257  CA  LYS X  26      24.592  29.621  42.871  1.00  0.00      SYST C  
ATOM    258  HA  LYS X  26      23.750  29.020  43.214  1.00  0.00      SYST H  
ATOM    259  CB  LYS X  26      25.770  29.071  43.762  1.00  0.00      SYST C  
ATOM    260  CG  LYS X  26      25.606  27.523  43.877  1.00  0.00      SYST C  
ATOM    261  CD  LYS X  26      26.791  26.973  44.689  1.00  0.00      SYST C  
ATOM    262  CE  LYS X  26      26.599  25.462  44.842  1.00  0.00      SYST C  
ATOM    263  NZ  LYS X  26      27.762  24.890  45.545  1.00  0.00      SYST N  
ATOM    264  C   LYS X  26      24.258  31.039  43.301  1.00  0.00      SYST C  
ATOM    265  O   LYS X  26      23.545  31.187  44.278  1.00  0.00      SYST O  
ATOM    266  N   ALA X  27      24.741  32.046  42.642  1.00  0.00      SYST N  
ATOM    267  H   ALA X  27      25.377  31.901  41.870  1.00  0.00      SYST H  
ATOM    268  CA  ALA X  27      24.355  33.343  42.837  1.00  0.00      SYST C  
ATOM    269  HA  ALA X  27      24.354  33.524  43.912  1.00  0.00      SYST H  
ATOM    270  CB  ALA X  27      25.388  34.304  42.215  1.00  0.00      SYST C  
ATOM    271  C   ALA X  27      22.907  33.644  42.267  1.00  0.00      SYST C  
ATOM    272  O   ALA X  27      22.123  34.437  42.790  1.00  0.00      SYST O  
ATOM    273  N   GLY X  28      22.632  33.001  41.130  1.00  0.00      SYST N  
ATOM    274  H   GLY X  28      23.375  32.406  40.791  1.00  0.00      SYST H  
ATOM    275  CA  GLY X  28      21.415  32.992  40.383  1.00  0.00      SYST C  
ATOM    276  C   GLY X  28      21.326  34.219  39.489  1.00  0.00      SYST C  
ATOM    277  O   GLY X  28      20.209  34.789  39.256  1.00  0.00      SYST O  
ATOM    278  N   LEU X  29      22.465  34.757  39.039  1.00  0.00      SYST N  
ATOM    279  H   LEU X  29      23.273  34.151  39.033  1.00  0.00      SYST H  
ATOM    280  CA  LEU X  29      22.581  35.908  38.125  1.00  0.00      SYST C  
ATOM    281  HA  LEU X  29      21.829  35.789  37.345  1.00  0.00      SYST H  
ATOM    282  CB  LEU X  29      22.334  37.202  38.866  1.00  0.00      SYST C  
ATOM    283  CG  LEU X  29      22.834  37.249  40.313  1.00  0.00      SYST C  
ATOM    284  CD1 LEU X  29      24.393  37.220  40.311  1.00  0.00      SYST C  
ATOM    285  CD2 LEU X  29      22.393  38.681  40.853  1.00  0.00      SYST C  
ATOM    286  C   LEU X  29      23.924  35.966  37.451  1.00  0.00      SYST C  
ATOM    287  O   LEU X  29      24.891  35.205  37.762  1.00  0.00      SYST O  
ATOM    288  N   CYS X  30      24.021  36.781  36.449  1.00  0.00      SYST N  
ATOM    289  H   CYS X  30      23.238  37.392  36.264  1.00  0.00      SYST H  
ATOM    290  CA  CYS X  30      25.124  37.057  35.546  1.00  0.00      SYST C  
ATOM    291  HA  CYS X  30      25.539  36.068  35.351  1.00  0.00      SYST H  
ATOM    292  CB  CYS X  30      24.684  37.727  34.292  1.00  0.00      SYST C  
ATOM    293  SG  CYS X  30      23.565  36.732  33.247  1.00  0.00      SYST S  
ATOM    294  C   CYS X  30      26.327  37.759  36.240  1.00  0.00      SYST C  
ATOM    295  O   CYS X  30      26.185  38.720  36.961  1.00  0.00      SYST O  
ATOM    296  N   GLN X  31      27.490  37.230  35.997  1.00  0.00      SYST N  
ATOM    297  H   GLN X  31      27.552  36.471  35.334  1.00  0.00      SYST H  
ATOM    298  CA  GLN X  31      28.798  37.693  36.614  1.00  0.00      SYST C  
ATOM    299  HA  GLN X  31      28.702  38.684  37.057  1.00  0.00      SYST H  
ATOM    300  CB  GLN X  31      29.254  36.701  37.667  1.00  0.00      SYST C  
ATOM    301  CG  GLN X  31      28.420  36.871  38.947  1.00  0.00      SYST C  
ATOM    302  CD  GLN X  31      29.002  36.192  40.188  1.00  0.00      SYST C  
ATOM    303  OE1 GLN X  31      29.968  35.402  40.269  1.00  0.00      SYST O  
ATOM    304  NE2 GLN X  31      28.430  36.457  41.339  1.00  0.00      SYST N  
ATOM    305  C   GLN X  31      29.895  37.846  35.528  1.00  0.00      SYST C  
ATOM    306  O   GLN X  31      29.662  37.462  34.338  1.00  0.00      SYST O  
ATOM    307  N   THR X  32      31.101  38.340  35.892  1.00  0.00      SYST N  
ATOM    308  H   THR X  32      31.264  38.826  36.762  1.00  0.00      SYST H  
ATOM    309  CA  THR X  32      32.223  38.310  34.924  1.00  0.00      SYST C  
ATOM    310  HA  THR X  32      31.863  37.986  33.948  1.00  0.00      SYST H  
ATOM    311  CB  THR X  32      32.701  39.828  34.698  1.00  0.00      SYST C  
ATOM    312  CG2 THR X  32      31.626  40.673  34.077  1.00  0.00      SYST C  
ATOM    313  OG1 THR X  32      33.252  40.359  35.841  1.00  0.00      SYST O  
ATOM    314  C   THR X  32      33.337  37.338  35.261  1.00  0.00      SYST C  
ATOM    315  O   THR X  32      33.622  36.942  36.409  1.00  0.00      SYST O  
ATOM    316  N   PHE X  33      34.092  37.023  34.164  1.00  0.00      SYST N  
ATOM    317  H   PHE X  33      33.806  37.306  33.237  1.00  0.00      SYST H  
ATOM    318  CA  PHE X  33      35.319  36.226  34.117  1.00  0.00      SYST C  
ATOM    319  HA  PHE X  33      35.835  36.352  35.069  1.00  0.00      SYST H  
ATOM    320  CB  PHE X  33      34.900  34.782  34.142  1.00  0.00      SYST C  
ATOM    321  CG  PHE X  33      34.662  34.191  32.779  1.00  0.00      SYST C  
ATOM    322  CD1 PHE X  33      33.494  34.476  32.035  1.00  0.00      SYST C  
ATOM    323  CE1 PHE X  33      33.275  33.790  30.748  1.00  0.00      SYST C  
ATOM    324  CZ  PHE X  33      34.263  32.891  30.286  1.00  0.00      SYST C  
ATOM    325  CE2 PHE X  33      35.365  32.612  31.052  1.00  0.00      SYST C  
ATOM    326  CD2 PHE X  33      35.550  33.216  32.275  1.00  0.00      SYST C  
ATOM    327  C   PHE X  33      36.274  36.577  32.991  1.00  0.00      SYST C  
ATOM    328  O   PHE X  33      35.945  37.291  32.011  1.00  0.00      SYST O  
ATOM    329  N   VAL X  34      37.519  36.072  33.116  1.00  0.00      SYST N  
ATOM    330  H   VAL X  34      37.663  35.412  33.867  1.00  0.00      SYST H  
ATOM    331  CA  VAL X  34      38.567  36.229  31.997  1.00  0.00      SYST C  
ATOM    332  HA  VAL X  34      38.313  37.175  31.519  1.00  0.00      SYST H  
ATOM    333  CB  VAL X  34      40.000  36.289  32.666  1.00  0.00      SYST C  
ATOM    334  CG1 VAL X  34      40.266  37.645  33.173  1.00  0.00      SYST C  
ATOM    335  CG2 VAL X  34      40.139  35.217  33.745  1.00  0.00      SYST C  
ATOM    336  C   VAL X  34      38.435  35.176  30.853  1.00  0.00      SYST C  
ATOM    337  O   VAL X  34      38.652  33.944  31.043  1.00  0.00      SYST O  
ATOM    338  N   TYR X  35      38.163  35.688  29.599  1.00  0.00      SYST N  
ATOM    339  H   TYR X  35      38.328  36.678  29.492  1.00  0.00      SYST H  
ATOM    340  CA  TYR X  35      37.904  34.914  28.414  1.00  0.00      SYST C  
ATOM    341  HA  TYR X  35      37.327  34.085  28.824  1.00  0.00      SYST H  
ATOM    342  CB  TYR X  35      36.956  35.606  27.418  1.00  0.00      SYST C  
ATOM    343  CG  TYR X  35      36.875  34.968  26.065  1.00  0.00      SYST C  
ATOM    344  CD1 TYR X  35      37.504  35.673  25.026  1.00  0.00      SYST C  
ATOM    345  CE1 TYR X  35      37.436  35.117  23.723  1.00  0.00      SYST C  
ATOM    346  CZ  TYR X  35      36.643  33.976  23.495  1.00  0.00      SYST C  
ATOM    347  OH  TYR X  35      36.515  33.606  22.182  1.00  0.00      SYST O  
ATOM    348  CE2 TYR X  35      35.958  33.292  24.512  1.00  0.00      SYST C  
ATOM    349  CD2 TYR X  35      36.054  33.840  25.847  1.00  0.00      SYST C  
ATOM    350  C   TYR X  35      39.208  34.500  27.789  1.00  0.00      SYST C  
ATOM    351  O   TYR X  35      40.033  35.327  27.453  1.00  0.00      SYST O  
ATOM    352  N   GLY X  36      39.413  33.177  27.533  1.00  0.00      SYST N  
ATOM    353  H   GLY X  36      38.656  32.588  27.849  1.00  0.00      SYST H  
ATOM    354  CA  GLY X  36      40.609  32.490  27.046  1.00  0.00      SYST C  
ATOM    355  C   GLY X  36      40.538  31.912  25.573  1.00  0.00      SYST C  
ATOM    356  O   GLY X  36      41.569  31.794  24.906  1.00  0.00      SYST O  
ATOM    357  N   GLY X  37      39.333  31.695  25.073  1.00  0.00      SYST N  
ATOM    358  H   GLY X  37      38.665  31.826  25.819  1.00  0.00      SYST H  
ATOM    359  CA  GLY X  37      38.922  31.529  23.574  1.00  0.00      SYST C  
ATOM    360  C   GLY X  37      39.246  30.128  22.961  1.00  0.00      SYST C  
ATOM    361  O   GLY X  37      38.408  29.591  22.253  1.00  0.00      SYST O  
ATOM    362  N   CYS X  38      40.336  29.439  23.250  1.00  0.00      SYST N  
ATOM    363  H   CYS X  38      40.918  29.883  23.947  1.00  0.00      SYST H  
ATOM    364  CA  CYS X  38      40.583  28.078  22.684  1.00  0.00      SYST C  
ATOM    365  HA  CYS X  38      40.421  28.184  21.612  1.00  0.00      SYST H  
ATOM    366  CB  CYS X  38      42.107  27.771  22.793  1.00  0.00      SYST C  
ATOM    367  SG  CYS X  38      42.533  26.175  22.069  1.00  0.00      SYST S  
ATOM    368  C   CYS X  38      39.624  27.019  23.306  1.00  0.00      SYST C  
ATOM    369  O   CYS X  38      39.601  26.833  24.544  1.00  0.00      SYST O  
ATOM    370  N   ARG X  39      38.746  26.387  22.518  1.00  0.00      SYST N  
ATOM    371  H   ARG X  39      38.939  26.401  21.527  1.00  0.00      SYST H  
ATOM    372  CA  ARG X  39      37.696  25.442  22.966  1.00  0.00      SYST C  
ATOM    373  HA  ARG X  39      36.977  25.274  22.164  1.00  0.00      SYST H  
ATOM    374  CB  ARG X  39      38.222  24.015  23.383  1.00  0.00      SYST C  
ATOM    375  CG  ARG X  39      39.157  23.512  22.272  1.00  0.00      SYST C  
ATOM    376  CD  ARG X  39      39.683  22.114  22.547  1.00  0.00      SYST C  
ATOM    377  NE  ARG X  39      38.588  21.054  22.698  1.00  0.00      SYST N  
ATOM    378  CZ  ARG X  39      38.575  19.779  22.307  1.00  0.00      SYST C  
ATOM    379  NH1 ARG X  39      39.585  19.200  21.749  1.00  0.00      SYST N  
ATOM    380  NH2 ARG X  39      37.456  19.162  22.294  1.00  0.00      SYST N  
ATOM    381  C   ARG X  39      36.730  25.958  24.033  1.00  0.00      SYST C  
ATOM    382  O   ARG X  39      36.192  25.238  24.903  1.00  0.00      SYST O  
ATOM    383  N   ALA X  40      36.479  27.265  24.018  1.00  0.00      SYST N  
ATOM    384  H   ALA X  40      37.041  27.770  23.347  1.00  0.00      SYST H  
ATOM    385  CA  ALA X  40      35.636  27.991  24.996  1.00  0.00      SYST C  
ATOM    386  HA  ALA X  40      36.150  27.898  25.953  1.00  0.00      SYST H  
ATOM    387  CB  ALA X  40      35.643  29.464  24.636  1.00  0.00      SYST C  
ATOM    388  C   ALA X  40      34.213  27.408  25.074  1.00  0.00      SYST C  
ATOM    389  O   ALA X  40      33.629  27.220  24.001  1.00  0.00      SYST O  
ATOM    390  N   LYS X  41      33.622  27.158  26.247  1.00  0.00      SYST N  
ATOM    391  H   LYS X  41      34.102  27.593  27.022  1.00  0.00      SYST H  
ATOM    392  CA  LYS X  41      32.275  26.598  26.544  1.00  0.00      SYST C  
ATOM    393  HA  LYS X  41      32.039  25.934  25.713  1.00  0.00      SYST H  
ATOM    394  CB  LYS X  41      32.290  25.820  27.881  1.00  0.00      SYST C  
ATOM    395  CG  LYS X  41      33.437  24.748  27.849  1.00  0.00      SYST C  
ATOM    396  CD  LYS X  41      33.329  23.689  28.946  1.00  0.00      SYST C  
ATOM    397  CE  LYS X  41      34.554  22.795  29.290  1.00  0.00      SYST C  
ATOM    398  NZ  LYS X  41      34.284  21.855  30.370  1.00  0.00      SYST N  
ATOM    399  C   LYS X  41      31.235  27.681  26.513  1.00  0.00      SYST C  
ATOM    400  O   LYS X  41      31.458  28.879  26.817  1.00  0.00      SYST O  
ATOM    401  N   ARG X  42      29.979  27.265  26.311  1.00  0.00      SYST N  
ATOM    402  H   ARG X  42      29.760  26.294  26.141  1.00  0.00      SYST H  
ATOM    403  CA  ARG X  42      28.786  28.036  25.983  1.00  0.00      SYST C  
ATOM    404  HA  ARG X  42      28.894  28.529  25.016  1.00  0.00      SYST H  
ATOM    405  CB  ARG X  42      27.651  26.964  25.818  1.00  0.00      SYST C  
ATOM    406  CG  ARG X  42      27.864  25.809  24.827  1.00  0.00      SYST C  
ATOM    407  CD  ARG X  42      26.586  24.987  24.600  1.00  0.00      SYST C  
ATOM    408  NE  ARG X  42      25.447  25.702  23.948  1.00  0.00      SYST N  
ATOM    409  CZ  ARG X  42      24.412  25.044  23.545  1.00  0.00      SYST C  
ATOM    410  NH1 ARG X  42      24.222  23.729  23.584  1.00  0.00      SYST N  
ATOM    411  NH2 ARG X  42      23.400  25.744  23.043  1.00  0.00      SYST N  
ATOM    412  C   ARG X  42      28.482  29.164  27.018  1.00  0.00      SYST C  
ATOM    413  O   ARG X  42      27.895  30.152  26.611  1.00  0.00      SYST O  
ATOM    414  N   ASN X  43      28.754  28.908  28.338  1.00  0.00      SYST N  
ATOM    415  H   ASN X  43      29.310  28.115  28.626  1.00  0.00      SYST H  
ATOM    416  CA  ASN X  43      28.537  29.986  29.308  1.00  0.00      SYST C  
ATOM    417  HA  ASN X  43      27.615  30.539  29.131  1.00  0.00      SYST H  
ATOM    418  CB  ASN X  43      28.359  29.297  30.680  1.00  0.00      SYST C  
ATOM    419  CG  ASN X  43      27.776  30.191  31.741  1.00  0.00      SYST C  
ATOM    420  OD1 ASN X  43      27.395  31.266  31.386  1.00  0.00      SYST O  
ATOM    421  ND2 ASN X  43      27.582  29.710  32.981  1.00  0.00      SYST N  
ATOM    422  C   ASN X  43      29.629  31.108  29.266  1.00  0.00      SYST C  
ATOM    423  O   ASN X  43      30.494  31.046  30.051  1.00  0.00      SYST O  
ATOM    424  N   ASN X  44      29.447  32.042  28.345  1.00  0.00      SYST N  
ATOM    425  H   ASN X  44      28.660  31.981  27.714  1.00  0.00      SYST H  
ATOM    426  CA  ASN X  44      30.313  33.202  28.189  1.00  0.00      SYST C  
ATOM    427  HA  ASN X  44      30.491  33.767  29.104  1.00  0.00      SYST H  
ATOM    428  CB  ASN X  44      31.671  32.796  27.678  1.00  0.00      SYST C  
ATOM    429  CG  ASN X  44      31.733  32.613  26.173  1.00  0.00      SYST C  
ATOM    430  OD1 ASN X  44      31.974  33.575  25.450  1.00  0.00      SYST O  
ATOM    431  ND2 ASN X  44      31.718  31.420  25.664  1.00  0.00      SYST N  
ATOM    432  C   ASN X  44      29.521  34.093  27.245  1.00  0.00      SYST C  
ATOM    433  O   ASN X  44      28.920  33.607  26.308  1.00  0.00      SYST O  
ATOM    434  N   PHE X  45      29.520  35.410  27.531  1.00  0.00      SYST N  
ATOM    435  H   PHE X  45      30.046  35.808  28.296  1.00  0.00      SYST H  
ATOM    436  CA  PHE X  45      28.749  36.404  26.688  1.00  0.00      SYST C  
ATOM    437  HA  PHE X  45      28.493  35.908  25.752  1.00  0.00      SYST H  
ATOM    438  CB  PHE X  45      27.383  36.688  27.321  1.00  0.00      SYST C  
ATOM    439  CG  PHE X  45      26.447  35.535  27.491  1.00  0.00      SYST C  
ATOM    440  CD1 PHE X  45      25.635  35.038  26.462  1.00  0.00      SYST C  
ATOM    441  CE1 PHE X  45      24.778  33.968  26.708  1.00  0.00      SYST C  
ATOM    442  CZ  PHE X  45      24.768  33.385  27.982  1.00  0.00      SYST C  
ATOM    443  CE2 PHE X  45      25.542  33.892  28.992  1.00  0.00      SYST C  
ATOM    444  CD2 PHE X  45      26.444  34.973  28.768  1.00  0.00      SYST C  
ATOM    445  C   PHE X  45      29.441  37.743  26.620  1.00  0.00      SYST C  
ATOM    446  O   PHE X  45      29.972  38.258  27.620  1.00  0.00      SYST O  
ATOM    447  N   LYS X  46      29.291  38.456  25.479  1.00  0.00      SYST N  
ATOM    448  H   LYS X  46      28.710  38.038  24.766  1.00  0.00      SYST H  
ATOM    449  CA  LYS X  46      29.761  39.865  25.320  1.00  0.00      SYST C  
ATOM    450  HA  LYS X  46      30.719  39.973  25.828  1.00  0.00      SYST H  
ATOM    451  CB  LYS X  46      29.911  40.133  23.789  1.00  0.00      SYST C  
ATOM    452  CG  LYS X  46      31.064  39.271  23.188  1.00  0.00      SYST C  
ATOM    453  CD  LYS X  46      31.612  39.577  21.824  1.00  0.00      SYST C  
ATOM    454  CE  LYS X  46      32.403  40.872  21.832  1.00  0.00      SYST C  
ATOM    455  NZ  LYS X  46      31.641  42.044  21.364  1.00  0.00      SYST N  
ATOM    456  C   LYS X  46      28.791  40.994  25.872  1.00  0.00      SYST C  
ATOM    457  O   LYS X  46      29.136  42.136  25.916  1.00  0.00      SYST O  
ATOM    458  N   SER X  47      27.568  40.622  26.290  1.00  0.00      SYST N  
ATOM    459  H   SER X  47      27.364  39.655  26.080  1.00  0.00      SYST H  
ATOM    460  CA  SER X  47      26.478  41.445  26.774  1.00  0.00      SYST C  
ATOM    461  HA  SER X  47      26.847  42.464  26.882  1.00  0.00      SYST H  
ATOM    462  CB  SER X  47      25.473  41.583  25.621  1.00  0.00      SYST C  
ATOM    463  OG  SER X  47      24.210  42.012  26.043  1.00  0.00      SYST O  
ATOM    464  C   SER X  47      25.904  40.911  28.091  1.00  0.00      SYST C  
ATOM    465  O   SER X  47      25.745  39.716  28.281  1.00  0.00      SYST O  
ATOM    466  N   ALA X  48      25.666  41.777  29.089  1.00  0.00      SYST N  
ATOM    467  H   ALA X  48      25.786  42.775  28.995  1.00  0.00      SYST H  
ATOM    468  CA  ALA X  48      24.844  41.477  30.234  1.00  0.00      SYST C  
ATOM    469  HA  ALA X  48      25.246  40.594  30.729  1.00  0.00      SYST H  
ATOM    470  CB  ALA X  48      24.869  42.502  31.317  1.00  0.00      SYST C  
ATOM    471  C   ALA X  48      23.402  41.174  29.733  1.00  0.00      SYST C  
ATOM    472  O   ALA X  48      22.920  40.132  30.186  1.00  0.00      SYST O  
ATOM    473  N   GLU X  49      22.720  41.919  28.855  1.00  0.00      SYST N  
ATOM    474  H   GLU X  49      23.083  42.835  28.633  1.00  0.00      SYST H  
ATOM    475  CA  GLU X  49      21.359  41.704  28.368  1.00  0.00      SYST C  
ATOM    476  HA  GLU X  49      20.631  41.806  29.173  1.00  0.00      SYST H  
ATOM    477  CB  GLU X  49      21.070  42.881  27.393  1.00  0.00      SYST C  
ATOM    478  CG  GLU X  49      19.561  42.891  26.876  1.00  0.00      SYST C  
ATOM    479  CD  GLU X  49      18.602  43.584  27.860  1.00  0.00      SYST C  
ATOM    480  OE1 GLU X  49      17.450  43.860  27.431  1.00  0.00      SYST O  
ATOM    481  OE2 GLU X  49      18.975  43.920  28.979  1.00  0.00      SYST O  
ATOM    482  C   GLU X  49      21.126  40.305  27.717  1.00  0.00      SYST C  
ATOM    483  O   GLU X  49      20.180  39.621  28.072  1.00  0.00      SYST O  
ATOM    484  N   ASP X  50      22.106  39.805  26.918  1.00  0.00      SYST N  
ATOM    485  H   ASP X  50      22.849  40.435  26.653  1.00  0.00      SYST H  
ATOM    486  CA  ASP X  50      22.282  38.449  26.420  1.00  0.00      SYST C  
ATOM    487  HA  ASP X  50      21.460  38.260  25.730  1.00  0.00      SYST H  
ATOM    488  CB  ASP X  50      23.451  38.223  25.438  1.00  0.00      SYST C  
ATOM    489  CG  ASP X  50      23.426  38.962  24.033  1.00  0.00      SYST C  
ATOM    490  OD1 ASP X  50      22.422  39.647  23.706  1.00  0.00      SYST O  
ATOM    491  OD2 ASP X  50      24.355  38.764  23.224  1.00  0.00      SYST O  
ATOM    492  C   ASP X  50      22.286  37.479  27.514  1.00  0.00      SYST C  
ATOM    493  O   ASP X  50      21.462  36.571  27.436  1.00  0.00      SYST O  
ATOM    494  N   CYS X  51      23.226  37.653  28.442  1.00  0.00      SYST N  
ATOM    495  H   CYS X  51      23.671  38.556  28.520  1.00  0.00      SYST H  
ATOM    496  CA  CYS X  51      23.385  36.690  29.517  1.00  0.00      SYST C  
ATOM    497  HA  CYS X  51      23.624  35.728  29.064  1.00  0.00      SYST H  
ATOM    498  CB  CYS X  51      24.499  37.075  30.433  1.00  0.00      SYST C  
ATOM    499  SG  CYS X  51      24.766  35.960  31.793  1.00  0.00      SYST S  
ATOM    500  C   CYS X  51      22.050  36.485  30.272  1.00  0.00      SYST C  
ATOM    501  O   CYS X  51      21.672  35.421  30.636  1.00  0.00      SYST O  
ATOM    502  N   MET X  52      21.413  37.625  30.640  1.00  0.00      SYST N  
ATOM    503  H   MET X  52      21.899  38.488  30.442  1.00  0.00      SYST H  
ATOM    504  CA  MET X  52      20.122  37.740  31.299  1.00  0.00      SYST C  
ATOM    505  HA  MET X  52      20.185  37.128  32.199  1.00  0.00      SYST H  
ATOM    506  CB  MET X  52      19.749  39.198  31.686  1.00  0.00      SYST C  
ATOM    507  CG  MET X  52      18.309  39.369  32.230  1.00  0.00      SYST C  
ATOM    508  SD  MET X  52      16.939  39.435  31.049  1.00  0.00      SYST S  
ATOM    509  CE  MET X  52      17.411  40.917  30.109  1.00  0.00      SYST C  
ATOM    510  C   MET X  52      18.991  37.113  30.495  1.00  0.00      SYST C  
ATOM    511  O   MET X  52      18.161  36.309  31.015  1.00  0.00      SYST O  
ATOM    512  N   ARG X  53      18.862  37.398  29.146  1.00  0.00      SYST N  
ATOM    513  H   ARG X  53      19.522  38.085  28.810  1.00  0.00      SYST H  
ATOM    514  CA  ARG X  53      17.827  36.829  28.319  1.00  0.00      SYST C  
ATOM    515  HA  ARG X  53      17.050  36.708  29.074  1.00  0.00      SYST H  
ATOM    516  CB  ARG X  53      17.374  37.801  27.200  1.00  0.00      SYST C  
ATOM    517  CG  ARG X  53      18.264  37.801  25.899  1.00  0.00      SYST C  
ATOM    518  CD  ARG X  53      18.314  39.066  25.040  1.00  0.00      SYST C  
ATOM    519  NE  ARG X  53      19.369  39.201  24.016  1.00  0.00      SYST N  
ATOM    520  CZ  ARG X  53      19.271  39.024  22.706  1.00  0.00      SYST C  
ATOM    521  NH1 ARG X  53      18.188  38.656  22.156  1.00  0.00      SYST N  
ATOM    522  NH2 ARG X  53      20.361  39.175  21.981  1.00  0.00      SYST N  
ATOM    523  C   ARG X  53      18.088  35.329  27.837  1.00  0.00      SYST C  
ATOM    524  O   ARG X  53      17.144  34.648  27.485  1.00  0.00      SYST O  
ATOM    525  N   THR X  54      19.334  34.827  27.892  1.00  0.00      SYST N  
ATOM    526  H   THR X  54      20.084  35.504  27.932  1.00  0.00      SYST H  
ATOM    527  CA  THR X  54      19.709  33.416  27.599  1.00  0.00      SYST C  
ATOM    528  HA  THR X  54      18.936  33.090  26.903  1.00  0.00      SYST H  
ATOM    529  CB  THR X  54      21.051  33.336  26.856  1.00  0.00      SYST C  
ATOM    530  CG2 THR X  54      21.422  31.931  26.384  1.00  0.00      SYST C  
ATOM    531  OG1 THR X  54      21.256  34.336  25.917  1.00  0.00      SYST O  
ATOM    532  C   THR X  54      19.669  32.529  28.818  1.00  0.00      SYST C  
ATOM    533  O   THR X  54      19.307  31.335  28.628  1.00  0.00      SYST O  
ATOM    534  N   CYS X  55      19.964  33.047  30.020  1.00  0.00      SYST N  
ATOM    535  H   CYS X  55      20.279  34.006  30.054  1.00  0.00      SYST H  
ATOM    536  CA  CYS X  55      20.175  32.197  31.195  1.00  0.00      SYST C  
ATOM    537  HA  CYS X  55      19.909  31.168  30.952  1.00  0.00      SYST H  
ATOM    538  CB  CYS X  55      21.684  32.136  31.487  1.00  0.00      SYST C  
ATOM    539  SG  CYS X  55      22.596  31.176  30.271  1.00  0.00      SYST S  
ATOM    540  C   CYS X  55      19.427  32.688  32.407  1.00  0.00      SYST C  
ATOM    541  O   CYS X  55      19.431  31.997  33.436  1.00  0.00      SYST O  
ATOM    542  N   GLY X  56      18.871  33.866  32.448  1.00  0.00      SYST N  
ATOM    543  H   GLY X  56      18.796  34.371  31.577  1.00  0.00      SYST H  
ATOM    544  CA  GLY X  56      18.215  34.527  33.597  1.00  0.00      SYST C  
ATOM    545  C   GLY X  56      16.729  34.141  33.727  1.00  0.00      SYST C  
ATOM    546  O   GLY X  56      16.376  32.996  33.404  1.00  0.00      SYST O  
ATOM    547  N   GLY X  57      15.910  35.073  34.170  1.00  0.00      SYST N  
ATOM    548  H   GLY X  57      16.288  35.922  34.564  1.00  0.00      SYST H  
ATOM    549  CA  GLY X  57      14.440  34.770  34.299  1.00  0.00      SYST C  
ATOM    550  C   GLY X  57      14.091  33.871  35.468  1.00  0.00      SYST C  
ATOM    551  O   GLY X  57      14.913  33.702  36.337  1.00  0.00      SYST O  
ATOM    552  N   ALA X  58      12.869  33.301  35.590  1.00  0.00      SYST N  
ATOM    553  H   ALA X  58      12.200  33.528  34.868  1.00  0.00      SYST H  
ATOM    554  CA  ALA X  58      12.460  32.330  36.689  1.00  0.00      SYST C  
ATOM    555  HA  ALA X  58      12.759  32.710  37.666  1.00  0.00      SYST H  
ATOM    556  CB  ALA X  58      10.950  32.234  36.729  1.00  0.00      SYST C  
ATOM    557  C   ALA X  58      13.162  31.027  36.497  1.00  0.00      SYST C  
ATOM    558  OC1 ALA X  58      13.282  30.501  35.347  1.00  0.00      SYST O  
ATOM    559  OC2 ALA X  58      13.604  30.398  37.500  1.00  0.00      SYST O  
"""


# Mock ExpD_Datapoint for testing (reusing from existing tests)
class MockExpD_Datapoint(ExpD_Datapoint):
    def __init__(self, top, data_id=None):
        self.top = top
        self.key = f"mock_key_{data_id}" if data_id else "mock_key"
        self.data_id = data_id

    def extract_features(self):
        return np.array([1.0])  # Mock feature

    def __repr__(self):
        return f"MockExpD_Datapoint(id={self.data_id}, top={self.top})"


# Helper functions (reusing from existing tests)
def create_single_chain_topologies(chain="X", count=15):
    """Create diverse single-chain topologies with good separation, fitting within test_pdb."""
    topologies = []
    start = 1
    # Adjust gap and length to ensure residues stay within 1-58
    gap = 5
    length = 5

    for i in range(count):
        current_start = start + i * (length + gap)
        current_end = current_start + length - 1
        if current_end > 58:  # Ensure we don't go beyond residue 58
            break
        topologies.append(
            TopologyFactory.from_range(
                chain, current_start, current_end, fragment_name=f"frag_{chain}_{i + 1}"
            )
        )
    return topologies


def create_multi_chain_topologies(chains=["X"], count_per_chain=5):
    """Create diverse multi-chain topologies with good separation, fitting within test_pdb's single chain X."""
    topologies = []

    for chain in chains:
        start = 1
        gap = 8  # Adjusted gap
        length = 5  # Adjusted length

        for i in range(count_per_chain):
            current_start = start + i * (length + gap)
            current_end = current_start + length - 1
            if current_end > 58:  # Ensure we don't go beyond residue 58
                break
            topologies.append(
                TopologyFactory.from_range(
                    chain, current_start, current_end, fragment_name=f"frag_{chain}_{i + 1}"
                )
            )
    return topologies


def create_peptide_topologies(chains=["X"], count_per_chain=5):
    """Create diverse peptide topologies with trimming and good separation, fitting within test_pdb."""
    topologies = []
    trim_values = [1, 2]

    for chain in chains:
        start = 1
        gap = 8  # Adjusted gap
        length = 8  # Adjusted length

        for i in range(count_per_chain):
            trim = trim_values[i % len(trim_values)]
            current_start = start + i * (length + gap)
            current_end = current_start + length - 1
            if current_end > 58:  # Ensure we don't go beyond residue 58
                break
            topologies.append(
                TopologyFactory.from_range(
                    chain,
                    current_start,
                    current_end,
                    fragment_name=f"pep_{chain}_{i + 1}",
                    peptide=True,
                    peptide_trim=trim,
                )
            )
    return topologies


def create_common_residues_for_chains(chains=["X"], coverage_factor=0.7):
    """Create common residue topologies that cover a portion of each chain, fitting within test_pdb."""
    common_residues = set()

    # For test_pdb, we only have chain 'X' and residues 1-58
    chain = "X"
    # Define ranges that fit within 1-58
    range1_end = min(58, int(20 * coverage_factor))
    range2_start = 30
    range2_end = min(58, int(range2_start + (58 - range2_start) * coverage_factor))

    if range1_end >= 1:
        common_residues.add(
            TopologyFactory.from_range(chain, 1, range1_end, fragment_name=f"common_{chain}_1")
        )

    if range2_end >= range2_start:
        common_residues.add(
            TopologyFactory.from_range(
                chain, range2_start, range2_end, fragment_name=f"common_{chain}_2"
            )
        )

    # Ensure at least two common residues for testing purposes if needed
    while len(common_residues) < 2:
        # Add dummy small fragments if needed, ensuring they are within bounds
        dummy_start = max(1, 58 - (2 - len(common_residues)) * 2)  # Ensure it's within 1-58
        common_residues.add(
            TopologyFactory.from_range(
                chain,
                dummy_start,
                min(58, dummy_start + 1),
                fragment_name=f"common_{chain}_dummy_{len(common_residues)}",
            )
        )

    return common_residues


@pytest.fixture
def create_datapoints_from_topologies():
    """Factory to create datapoints from topologies."""

    def _create(topologies):
        return [MockExpD_Datapoint(topo, i) for i, topo in enumerate(topologies)]

    return _create


@pytest.fixture
def setup_splitter(request):
    """Factory to set up DataSplitter with mocked dependencies."""

    def _setup(datapoints, common_residues, **kwargs):
        mock_loader = MagicMock(spec=ExpD_Dataloader)
        mock_loader.data = copy.deepcopy(datapoints)
        mock_loader.y_true = np.array([1.0] * len(datapoints))

        def mock_filter_func(dataset, common_topos, check_trim=False):
            if not common_topos:
                return []
            return [
                dp
                for dp in dataset
                if any(
                    PairwiseTopologyComparisons.intersects(dp.top, ct, check_trim=check_trim)
                    for ct in common_topos
                )
            ]

        patcher1 = patch("jaxent.src.interfaces.topology.utils.calculate_fragment_redundancy")
        patcher2 = patch(
            "jaxent.src.data.splitting.split.filter_common_residues", side_effect=mock_filter_func
        )

        mock_calc = patcher1.start()
        mock_filter = patcher2.start()

        request.addfinalizer(patcher1.stop)
        request.addfinalizer(patcher2.stop)

        mock_calc.return_value = [0.5] * len(datapoints)

        default_kwargs = {
            "random_seed": 42,
            "train_size": 0.6,
            "centrality": False,
            "check_trim": True,
            "min_split_size": 2,
            "exclude_selection": "",  # Override default for tests
        }
        default_kwargs.update(kwargs)

        mock_filter.side_effect = None
        mock_filter.return_value = copy.deepcopy(datapoints)

        splitter = DataSplitter(
            dataset=mock_loader, common_residues=common_residues, **default_kwargs
        )

        mock_filter.side_effect = mock_filter_func
        mock_filter.return_value = None

        return splitter

    return _setup


@pytest.fixture
def real_universe(request):
    """Create a real MDAnalysis Universe for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pdb") as tmp_pdb_file:
        # Write the test_pdb content multiple times to create a multi-frame PDB
        tmp_pdb_file.write(test_pdb)
        tmp_pdb_path = tmp_pdb_file.name

    universe = MDAnalysis.Universe(tmp_pdb_path)

    # Clean up the temporary file after the test
    def finalizer():
        os.remove(tmp_pdb_path)

    request.addfinalizer(finalizer)

    return universe


@pytest.fixture
def patch_partial_distances():
    """Patch partial_topology_pairwise_distances and yield the mock."""
    with patch(
        "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
    ) as mock_distances:
        yield mock_distances


@pytest.fixture(autouse=True)
def patch_pairwise_distances(request):
    """Automatically patch partial_topology_pairwise_distances for all tests."""
    patcher = patch(
        "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
    )
    mock_func = patcher.start()

    def make_distance_matrix(dataset, **kwargs):
        n = len(dataset)
        if n == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))
        mat = np.random.rand(n, n)
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0)
        std = np.zeros_like(mat)
        return mat, std

    mock_func.side_effect = lambda dataset, **kwargs: make_distance_matrix(dataset, **kwargs)

    request.addfinalizer(patcher.stop)


class TestSpatialSplitBasicFunctionality:
    """Test basic functionality of spatial_split method."""

    def test_basic_spatial_split_returns_two_lists(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        real_universe,
        patch_partial_distances,
    ):
        """Test that spatial_split returns two lists."""
        topologies = create_single_chain_topologies(chain="X", count=15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["X"])

        splitter = setup_splitter(datapoints, common_residues)

        train_data, val_data = splitter.spatial_split(real_universe)

        assert isinstance(train_data, list)
        assert isinstance(val_data, list)
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_spatial_split_with_different_train_sizes(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        real_universe,
        patch_partial_distances,
    ):
        """Test spatial split with different training set sizes."""
        topologies = create_single_chain_topologies(chain="X", count=15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(chains=["X"])

        for train_size in [0.4, 0.5, 0.7, 0.8]:
            splitter = setup_splitter(datapoints, common_residues, train_size=train_size)

            train_data, val_data = splitter.spatial_split(real_universe)

            expected_train_size = int(train_size * len(datapoints))
            expected_val_size = len(datapoints) - expected_train_size

            assert len(train_data) == expected_train_size
            assert len(val_data) == expected_val_size

    def test_numpy_import_error(
        self, create_datapoints_from_topologies, setup_splitter, real_universe
    ):
        """Test handling when numpy import fails."""
        topologies = create_single_chain_topologies(chain="X", count=15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(chains=["X"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("builtins.__import__", side_effect=ImportError("No module named 'numpy'")):
            with pytest.raises(ImportError, match="NumPy is required"):
                splitter.spatial_split(real_universe)

#!/usr/bin/env bash
set -euo pipefail
cd /home/alexi/Documents/JAX-ENT

if [[ $# != 1 || ! "$1" =~ ^[0-9]{8}$ ]]; then
  echo 'Usage: run_moprp_fresh_sidecars.sh YYYYMMDD' >&2
  exit 2
fi
moprp_stamp="$1"

for moprp_mode in rate frame_uptake linear_uptake; do
  case "$moprp_mode" in
    rate) moprp_suffix=_rate ;;
    frame_uptake) moprp_suffix= ;;
    linear_uptake) moprp_suffix=_linear_bv ;;
  esac
  moprp_campaign="_moprp_policy_sidecars${moprp_suffix}_fresh_${moprp_stamp}"
  for moprp_example in 2_CrossValidation 3_CrossValidationBV; do
    moprp_runner=run_moprp_policy_sidecars.py
    if [[ "$moprp_example" == 3_CrossValidationBV ]]; then
      moprp_runner=run_moprp_bv_policy_sidecars.py
    fi
    moprp_root="/home/alexi/Documents/JAX-ENT/jaxent/examples/$moprp_example"
    moprp_fit_dir="$moprp_root/fitting/jaxENT/$moprp_campaign"
    moprp_analysis_dir="$moprp_root/analysis/$moprp_campaign"
    mkdir -p "$moprp_fit_dir/logs"
    echo "Run $moprp_example $moprp_mode: $moprp_campaign"
    .venv/bin/python jaxent/examples/2_CrossValidation/fitting/jaxENT/run_moprp_sidecars_background.py \
      --jobs 10 --runner "$moprp_root/fitting/jaxENT/$moprp_runner" \
      --output-dir "$moprp_fit_dir" --analysis-dir "$moprp_analysis_dir" \
      >> "$moprp_fit_dir/logs/background.log" 2>&1
  done
done
echo 'All six fresh MoPrP campaigns and analyses complete'

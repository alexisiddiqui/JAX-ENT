#!/usr/bin/env bash
set -euo pipefail
cd /home/alexi/Documents/JAX-ENT

moprp_suffix=_moprp_policy_sidecars_rate
moprp_smoke=false
for moprp_argument in "$@"; do
  case "$moprp_argument" in
    --linear-bv) moprp_suffix=_moprp_policy_sidecars_linear_bv ;;
    --smoke) moprp_smoke=true ;;
    *) echo 'Usage: run_moprp_rate_sidecars.sh [--linear-bv] [--smoke]' >&2; exit 2 ;;
  esac
done
if "$moprp_smoke"; then moprp_suffix+=_smoke; fi

for moprp_example in 2_CrossValidation 3_CrossValidationBV; do
  moprp_runner=run_moprp_policy_sidecars.py
  if [[ "$moprp_example" == 3_CrossValidationBV ]]; then
    moprp_runner=run_moprp_bv_policy_sidecars.py
  fi
  moprp_fit_dir="/home/alexi/Documents/JAX-ENT/jaxent/examples/$moprp_example/fitting/jaxENT/$moprp_suffix"
  moprp_analysis_dir="/home/alexi/Documents/JAX-ENT/jaxent/examples/$moprp_example/analysis/$moprp_suffix"
  mkdir -p "$moprp_fit_dir/logs"
  echo "Run $moprp_example uptake campaign: $moprp_suffix"
  .venv/bin/python jaxent/examples/2_CrossValidation/fitting/jaxENT/run_moprp_sidecars_background.py \
    --jobs 10 \
    --runner "/home/alexi/Documents/JAX-ENT/jaxent/examples/$moprp_example/fitting/jaxENT/$moprp_runner" \
    --output-dir "$moprp_fit_dir" \
    --analysis-dir "$moprp_analysis_dir" \
    >> "$moprp_fit_dir/logs/background.log" 2>&1
done
echo 'Both uptake campaigns and analyses complete'

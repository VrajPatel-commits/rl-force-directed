#!/bin/bash
# Submit the whole pipeline with dependencies. Run from the project folder on the cluster:
#   ACCOUNT=<your-slurm-account> bash scripts/cluster/submit_all.sh       (all stages)
#   ACCOUNT=<your-slurm-account> bash scripts/cluster/submit_all.sh 3     (start from stage 3)
set -e
: "${ACCOUNT:?set ACCOUNT to your SLURM account, e.g. ACCOUNT=def-xyz}"
cd "$(dirname "$0")/../.."
mkdir -p logs
FROM=${1:-1}
S=scripts/cluster
submit() { sbatch --parsable --account="$ACCOUNT" "$@"; }
after() { local ids; ids=$(echo "$@" | tr -s ' ' ':'); ids=${ids#:}; ids=${ids%:}; [ -n "$ids" ] && echo "--dependency=afterok:$ids"; }
J1=""; J2=""; J3=""; J4=""; J5=""; J6=""; J7=""
[ "$FROM" -le 1 ] && J1=$(submit $S/01_calibrate.sbatch) && echo "1 calibrate         $J1"
[ "$FROM" -le 2 ] && J2=$(submit $(after $J1) $S/02_calibrate_finish.sbatch) && echo "2 calibrate_finish  $J2"
[ "$FROM" -le 3 ] && J3=$(submit $(after $J2) $S/03_default_runs.sbatch) && echo "3 default_runs      $J3"
[ "$FROM" -le 4 ] && J4=$(submit $(after $J3) $S/04_prepare.sbatch) && echo "4 prepare           $J4"
[ "$FROM" -le 5 ] && J5=$(submit $(after $J2) $S/05_tune_rules.sbatch) && echo "5 tune_rules        $J5"
[ "$FROM" -le 6 ] && J6=$(submit $(after $J4 $J5) $S/06_standardizer.sbatch) && echo "6 standardizer      $J6"
[ "$FROM" -le 7 ] && J7=$(submit $(after $J6) $S/07_train.sbatch) && echo "7 train             $J7"
J8=$(submit $(after $J7) $S/08_eval_agent.sbatch) && echo "8 eval_agent        $J8"
J9=$(submit $(after $J5) $S/09_eval_baselines.sbatch) && echo "9 eval_baselines    $J9"
J10=$(submit $(after $J8 $J9) $S/10_analyze.sbatch) && echo "10 analyze          $J10"
echo "submitted. check with: squeue -u \$USER   or   python scripts/cluster/status.py"

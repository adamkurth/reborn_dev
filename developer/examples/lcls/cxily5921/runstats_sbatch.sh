#!/bin/bash
run="${1#0}"
if [ "$run" == "all" ]; then
	for r in $(./available_runs.sh); do ./runstats_sbatch.sh "$r"; done
	exit
fi
[ "$run" == "" ] && exit
n=12
sbatch -p psfehq --ntasks=1 --cpus-per-task=$n --mem-per-cpu=8G --job-name=rs"$(printf '%04d' $run)"  --output=results/sbatch/runstats_$run.out \
--error=results/sbatch/runstats_$run.err --wrap="python -u runstats.py --run=$run --overwrite --n_processes=$n"

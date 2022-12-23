#!/bin/bash
run="${1#0}"
if [ "$run" == "all" ]; then
	for r in $(./available_runs.sh); do ./runstats_sbatch.sh "$r"; done
	exit
fi
[ "$run" == "" ] && exit
n=12
sbatch -p psfehq --ntasks=1 --cpus-per-task=$n --mem-per-cpu=8G --job-name=p$(printf '%04d' $run) --output=results/sbatch/profiles_$run.out --error=results/sbatch/profiles_$run.err --wrap="source setup.sh; python -u profiles.py -r $run --overwrite"

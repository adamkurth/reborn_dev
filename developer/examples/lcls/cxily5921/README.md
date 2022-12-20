# Example Analysis Code for the LY59 Experiment.

## Basic setup

We assume that you have cloned reborn into your scratch directory.  If not, do this:

```bash
mkdir /reg/d/psdm/cxi/cxily5921/scratch/<username>
git clone https://gitlab.com/kirianlab/reborn.git
```

The directory with the examples for beamtime LY59 is here:

```bash
cd reborn/developer/examples/lcls/cxily5921
```

Before you start running any scripts on the LCLS computers, source the setup script:

```bash
source setup.sh
```

The above will ensure that your python environment is configured (on psana machines) and 
the reborn package is in your python path.

## Viewing individual frames

```bash
./quickview -r <run number>
```

## Launching "runstats" analysis jobs

The "runstats" processor will gather detector statistics for a given run.  It returns the 
mean of all the PAD images, the mean squared image, standard deviation, minimum/maximum 
pixel values, and pixel-by-pixel histograms of intensity values for the entire run.  To 
launch a job interactively, do this:

```bash
./runstats.py -r <run number> -j 12 --view
```

The `j` flag sets the number of processes to run in parallel.  If you just want to see the
first N frames, you can add the `--max_events` flag.  For other options, do this

```bash
./runstats --help
```
 







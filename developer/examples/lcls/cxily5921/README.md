# Example Analysis Code for the LY59 Experiment.

## Connecting to SLAC

A convenient way to connect to SLAC's analysis computers is through NoMachine.

Working configuration:
```
          Name: SLAC
      Protocol: SSH
          Host: psnxserv.slac.stanford.edu
          Port: 22
Authentication: Password
         Proxy: No
```

The main analysis computer, psana, is not connected to the internet.
Once you are on a login node, connect to psana. Do this:

```bash
ssh psana
```

The base psana environment is not suitable for analysis (it is based on python 2.7).
To get the proper the analysis run:

```bash
source /reg/g/psdm/etc/psconda.sh -py3
```

This is included in the setup script (see Basic Setup).

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

## Configurations

The `config.py` module contains the various configurations for analysis.  See the contents
of the file for more details.  Ask Roberto if there are questions.

## Example data

The configurations are presently set to load and analyze data from experiment ID `cxix53120`.  If you don't
have access you can ask Andy for access.  When the beamtime starts we will switch to the 
proper `cxily5921` experiment ID.

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








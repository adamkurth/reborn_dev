#!/usr/bin/env python3
import psana
# We begin with an experiment ID and a run number:
experiment_id = 'cxil2316'
run_number = 116
# We create a psana "DataSource" instance:
ds = psana.DataSource('exp=%s:run=%d:smd' % (experiment_id, run_number))
# And a Run instance:
run = ds.runs().__next__()
# The following is useful if you just want to know what detectors are present in the data:
detnames = psana.DetNames(nametype='detectors')
for d in detnames:
    print(d)
# The previous lines may be used to identify the CSPAD detector ID: DscCsPad
pad_detector_id = 'DscCsPad'
# Now we search for additional parameters related to this detector:
detnames = psana.DetNames(nametype='epics')
for d in detnames:
    if 'DscCsPad' in d[0]:
        print(d)
# The detector transtion stage is determined from an EPICS PV.  
pad_detector_distance_pv = 'DscCsPad_z'
# We do not initially know how many events there are in the run.  We must make a first pass
# thorugh the run and collect all the timestamps.
max_events = 100  # For demonstration purposes, let's only look at part of the run
timestamps = []
for i, event in enumerate(run.events()):
    event_id = event.get(psana.EventId)
    timestamps.append((event_id.time()[0], event_id.time()[1], event_id.fiducials()))
    if i >= max_events-1:
        break
# Now we know how many events we are dealing with:
n_events = len(timestamps)
print('Found', n_events, 'events')
# Now we make a new DataSource to begin pulling out some data:
ds = psana.DataSource('exp=%s:run=%d:smd' % (experiment_id, run_number))
run = ds.runs().__next__()
# This is the CSPAD Detector instance:
pad_det = psana.Detector(pad_detector_id)
# We will need the EBeam Detector instance to get photon energy:
ebeam_det = psana.Detector('EBeam')
# Here is the EPICS PV Detector for the stage that moves the CSPAD (approximately) along the beam path
pad_z_det = psana.Detector(pad_detector_distance_pv)
# The EVR (event code reader) can be used to check for events such as "laser on"
evr = psana.Detector('evr0')
# Here we go... wish us luck...
for i, ts in enumerate(timestamps):
#for i, event in enumerate(run.events()):
    if i > n_events:
        break
    print('Event', i)
    et = psana.EventTime(int((ts[0] << 32) | ts[1]), ts[2])
    print('Time', et)
    event = run.event(et)
    print('hello')
    # It is common for some events to have missing data.  We skip those events.
    if event is None:
        print('Event', i, 'is None.  Skipping')
        continue
    # Electron beam info for this shot:
    ebeam = ebeam_det.get(event)
    if ebeam is None:
        print('EBeam', i, 'is None.  Skipping')
        continue
    # Here is the photon energy (just one photon; not pulse energy...) in eV: 
    photon_energy = ebeam.ebeamPhotonEnergy()
    print('Photon energy:', photon_energy)
    # CSPAD data for this shot:
    pad_dat = pad_det.calib(event)
    # pad = pad_det.raw(event)  # Alternatively, raw data if you want to do your own pre-processing
    if pad_dat is None:
        print('PAD data', i, 'is None.  Skipping')
        continue
    print('PAD data shape:', pad_dat.shape)
    # Here are some relative coordinates of the PAD pixels in micron units:
    x, y, z = pad_det.coords_xyz(event)
    print('Corner pixel position:', x.flat[0], y.flat[0], z.flat[0])
    # Here is the z stage postion for this event in mm units.
    pad_z = pad_z_det(event)
    print('PAD z stage position:', pad_z)
    # We check what event codes are present:
    event_codes = evr.eventCodes(event)
    # Check if the x-ray and laser codes are present:
    xray_on = 40 in event_codes
    laser_on = 41 in event_codes
    print('Xrays:', xray_on, ', Laser:', laser_on)

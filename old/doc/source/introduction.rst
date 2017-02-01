Introduction
************

bornagain is meant to be a flexible, intuitive and easy-to-use library of analysis tools for x-ray diffraction data.  It is meant for quickly hashing out ideas for projects that do not fit within existing analysis pipelines. 

Firstly, let's get the simple things out of the way:

- All units will be SI, even photon energy.  This way, we will always avoid confusion.  Angles are measured in radians.
- Ease of use and flexibility are more important than speed.  We want to be able to flesh out ideas quickly, even if it takes a while for the program to run.  Bottlenecks can be sorted out later, e.g. by profiling a program.
- Ease of use is more important than flexibility.  The interface should be transparent and easy to remember.
- The programmer shouldn't need to do much book keeping.
- We won't worry much about memory - we will cache information whenever possible so that things are not computed more frequently than need be.


Here are the expectations we have for an x-ray diffraction experiment, which defines the scope of this package.  First and foremost, there is an x-ray source that illuminates the target.  The x-ray source is defined by the following properties:

.. blockdiag::

    blockdiag {
    
        "x-ray source" -> "mean wavelength";
        "x-ray source" -> "spectral width";
        "x-ray source" -> "beam direction";
        "x-ray source" -> "beam divergence";
        "x-ray source" -> "polarization vector";
        "x-ray source" -> "polarization ratio";
        "x-ray source" -> "pulse energy";
    
    }

The primary data class is a detector panel - 2D array of pixels that measure diffracted intensity.  Since many detectors now contain multiple panels, the panelList class is used to abstract away from the specific details of each panel.

.. blockdiag:: 
    :desctable:

    blockdiag {
        
        bornagain -> detector;
        bornagain -> source;
        bornagain -> dataio;
        
        detector -> panel;
        detector -> panelList;
        
        source -> beam;
        
        dataio -> readers;
        
        detector [description = "Things relevant to x-ray detectors."];
        source [description = "Things relevant to x-ray source."];
        dataio [description = "Things related to reading/writing data."];
        readers [description = "Classes for reading various file formats."];
        
    }
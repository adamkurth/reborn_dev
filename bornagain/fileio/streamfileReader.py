"""
A module to read and parse streamfiles from CrystFEL.

Date Created: 14 Apr 2019
Last Modified: 15 Apr 2019
Author: Joe Chen
"""

import numpy as np

# Delimiters
sta_geom    = "----- Begin geometry file -----"
sta_chunk   = "----- Begin chunk -----"
sta_crystal = "--- Begin crystal"

end_geom    = "----- End geometry file -----"
end_chunk   = "----- End chunk -----"
end_crystal = "--- End crystal"


def get_total_number_of_frames(streamfile_name):

    # Load the streamfile
    f = open(streamfile_name, 'r') 

    count = 0
    for line in f:
    	if sta_chunk in line:
        	count += 1

    # close the file
    f.close()

    return count


def get_nth_frame(streamfile_name, n):

    # Load the streamfile
    f = open(streamfile_name, 'r') 

    count = 0
    for line in f:
        if sta_chunk in line:
            count += 1

        if count == n:
            break


    # Initialising 
    A = np.zeros((3,3))
    cxiFilepath = 0
    cxiFileFrameNumber = 0

    astar_exist = False

    for line in f:
        if "Image filename:" in line:
            cxiFilepath = line[16:-1]

        if "Event:" in line:
            cxiFileFrameNumber = int(line[9:])

        if "astar = " in line:
            A[0,0] = float(line[8:19])
            A[0,1] = float(line[19:30])
            A[0,2] = float(line[29:40])
            astar_exist = True

        if "bstar = " in line:
            A[1,0] = float(line[8:19])
            A[1,1] = float(line[19:30])
            A[1,2] = float(line[29:40])

        if "cstar = " in line:
            A[2,0] = float(line[8:19])
            A[2,1] = float(line[19:30])
            A[2,2] = float(line[29:40])

        if end_chunk in line:
            break

    # If frame does not contain a star, etc. set the A star matrix to None
    if astar_exist == False:
        A = None


    # close the file
    f.close()

    return A, cxiFilepath, cxiFileFrameNumber




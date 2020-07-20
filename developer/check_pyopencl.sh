#!/bin/bash

echo $(which python)
python << EOF
def print_device_info(device):
    d = device
    print("")
    # Print out some information about the devices
    print("    Name:", d.name)
    print("    Version:", d.opencl_c_version)
    print("    Max. Compute Units:", d.max_compute_units)
    if 'cl_khr_fp64' not in d.extensions.split():
        print("    Double Precision Support: No")
    else:
        print("    Double Precision Support: Yes")
    print("    Local Memory Size:", d.local_mem_size / 1024, "KB")
    print("    Global Memory Size:", d.global_mem_size / (1024 * 1024), "MB")
    print("    Max Alloc Size:", d.max_mem_alloc_size / (1024 * 1024), "MB")
    print("    Max Work-group Total Size:", d.max_work_group_size)
    print("    Cache Size:", d.global_mem_cacheline_size)
    # Find the maximum dimensions of the work-groups
    dim = d.max_work_item_sizes
    print("    Max Work-group Dims:(", dim[0], " ".join(map(str, dim[1:])), ")")
import pyopencl as cl
print(75 * "=")
print('Summary of platforms and devices:')
print(75 * "=")
# Create a list of all the platform IDs
platforms = cl.get_platforms()
print("\nNumber of OpenCL platforms:", len(platforms))
# Investigate each platform
for p in platforms:
    # Print out some information about the platforms
    print("\n-------------------------\n")
    print("Platform:", p.name)
    print("Vendor:", p.vendor)
    print("Version:", p.version)
    # Discover all devices
    devices = p.get_devices()
    print("Number of devices:", len(devices))
    # Investigate each device
    for d in devices:
        print_device_info(d)
context = cl.create_some_context()
queue = cl.CommandQueue(context)
kern_str = open('../reborn/simulate/clcore.cpp').read()
programs = cl.Program(context, kern_str).build()
EOF
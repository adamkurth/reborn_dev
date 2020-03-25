


# Following installation structures from the arrayfire wiki
# https://github.com/arrayfire/arrayfire/wiki


xcode-select --install




to uninstall homebrew do this:
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall)"


/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install cmake 
brew install freeimage
brew install fftw
brew install boost
brew install openblas 
===================================
====================================
For compilers to find openblas you may need to set:
  export LDFLAGS="-L/usr/local/opt/openblas/lib"
  export CPPFLAGS="-I/usr/local/opt/openblas/include"
=========================================================
brew install glbinding
brew install glfw
brew install pkg-config
brew install libomp         
===========================================
==========================================
On Apple Clang, you need to add several options to use OpenMP's front end
instead of the standard driver option. This usually looks like
  -Xpreprocessor -fopenmp -lomp

You might need to make sure the lib and include directories are discoverable
if /usr/local is not searched:

  -L/usr/local/opt/libomp/lib -I/usr/local/opt/libomp/include

For CMake, the following flags will cause the OpenMP::OpenMP_CXX target to
be set up correctly:
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES="omp" -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib
=========================================
brew install lapack
=========================================
==========================================
lapack is keg-only, which means it was not symlinked into /usr/local,
because macOS already provides this software and installing another version in
parallel can cause all kinds of trouble.

For compilers to find lapack you may need to set:
  export LDFLAGS="-L/usr/local/opt/lapack/lib"
  export CPPFLAGS="-I/usr/local/opt/lapack/include"

For pkg-config to find lapack you may need to set:
  export PKG_CONFIG_PATH="/usr/local/opt/lapack/lib/pkgconfig"
====================================

git clone --recursive https://github.com/arrayfire/arrayfire.git
git submodule init
git submodule update
cd arrayfire
mkdir build && cd build
export LDFLAGS="-L/usr/local/opt/lapack/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/opt/lapack/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/opt/lapack/lib/pkgconfig $PKG_CONFIG_PATH"
export LDFLAGS="-L/usr/local/opt/openblas/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/opt/openblas/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig $PKG_CONFIG_PATH"
export LDFLAGS="-L/usr/local/opt/libomp/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/opt/libomp/include $CPPFLAGS"



mkdir ~/miniconda3/envs/reborn3/share/arrayfire
cmake .. -DCMAKE_BUILD_TYPE=Release -DAF_BUILD_CUDA=OFF -DAF_BUILD_OPENCL=ON -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES="omp" -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib
make -j6
cmake .. -DCMAKE_INSTALL_PREFIX=~/miniconda3/envs/reborn3/share/arrayfire  
make install
export DYLD_LIBRARY_PATH=~/miniconda3/envs/reborn3/share/arrayfire/lib:$DYLD_LIBRARY_PATH
make test         # see test issues below




################## openmp ###############
==> Caveats
On Apple Clang, you need to add several options to use OpenMP's front end
instead of the standard driver option. This usually looks like
  -Xpreprocessor -fopenmp -lomp

You might need to make sure the lib and include directories are discoverable
if /usr/local is not searched:

  -L/usr/local/opt/libomp/lib -I/usr/local/opt/libomp/include

For CMake, the following flags will cause the OpenMP::OpenMP_CXX target to
be set up correctly:
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES="omp" -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib






########## cmake ##########################
CUDA_TOOLKIT_ROOT_DIR not found or specified
-- Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_NVCC_EXECUTABLE CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY) (Required is at least version "7.0")
-- Found OpenCL: /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk/System/Library/Frameworks/OpenCL.framework (found suitable version "1.2", minimum required is "1.2") 
-- Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES) (found version "1.0")
-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES) (found version "1.0")
-- Checking for module 'cblas'
--   No package 'cblas' found
-- Checking for [Accelerate]
-- Includes found
-- CBLAS Symbols FOUND
-- CBLAS library found
-- Could NOT find LAPACK (missing: LAPACK_INCLUDE_DIR LAPACK_LIBRARIES) 
-- Could NOT find Doxygen (missing: DOXYGEN_EXECUTABLE) 
-- Could NOT find MKL (missing: MKL_INCLUDE_DIR MKL_Core_LINK_LIBRARY) 
-- Boost version: 1.68.0
-- A library with LAPACK API found.
-- Found OpenCL: /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk/System/Library/Frameworks/OpenCL.framework (found version "1.2") 
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/rkirian/work/projects/reborn/ignore_rick/arrayfire/install1/arrayfire/build









############# test issues ################
24/243 Test  #24: test_blas_opencl ....................***Exception: SegFault  1.08 sec
hangs on Start 116: test_jit_opencl for a long time
hangs on Start 118: test_join_opencl also
haons on:
	Start 118: test_join_opencl
	Start 120: test_lu_dense_opencl

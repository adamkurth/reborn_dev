
// Group size may be set with a flag a compile time
#ifndef GROUP_SIZE
    #define GROUP_SIZE 1
#endif

// The use of double is enabled by the CONFIG_USE_DOUBLE flag at compile time
#if CONFIG_USE_DOUBLE
    #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #define DOUBLE_SUPPORT_AVAILABLE
    #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #define DOUBLE_SUPPORT_AVAILABLE
    #endif
    #ifdef DOUBLE_SUPPORT_AVAILABLE
        #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
    #endif
#endif

// Note that dsfloat* (i.e. double/single) types will become either double or single as configured above
// Please be mindful of this in the code below

#if defined(DOUBLE_SUPPORT_AVAILABLE)
    typedef double dsfloat;
    typedef double2 dsfloat2;
    typedef double4 dsfloat4;
    typedef double16 dsfloat16;
    typedef long dsint;
    typedef long2 dsint2;
    #define PI 3.14159265358979323846
    #define PI2 6.28318530717958647693
    #define ATOMIC_CMPXCHG atom_cmpxchg
#else
    typedef float dsfloat;
    typedef float2 dsfloat2;
    typedef float4 dsfloat4;
    typedef float16 dsfloat16;
    typedef int dsint;
    typedef int2 dsint2;
    #define PI 3.14159265359f
    #define PI2 6.28318530718f
    #define ATOMIC_CMPXCHG atomic_cmpxchg
#endif


// There is no atomic add for floats.  We are therefore trying the suggestion found here:
//     https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
// This appears to be the basic idea:
// (1) Copy the current value from an element in gloabl memory
// (2) Pre-compute the next value using that copy and the amount to be added
// (3) Lock the global memory
// (4) Check if any other thread has changed the global memory since step (1)
// (5) If the memory has not been changed, update the global value with the next value, else start over at (1)
// The above seems silly because the lock should occur before step 1, and perhaps there is a way to achieve that,
// but of course this is what an atomic add function would do.  The reason for the above strategy is that it utilizes
// the following "compare and exchange" function:
//     int atomic_cmpxchg (	volatile __global int *addr , int cmp, int val)
// This will read the 32-bit value (referred to as old) stored at location pointed by addr.
// Compute (addr == cmp) ? val : old and store result at location pointed by p. The function returns old.

inline void atomic_add_real(volatile global dsfloat *addr, dsfloat val){
union {dsint u; dsfloat f;} next, expected, current;
current.f = *addr;
do {
expected.f = current.f;
next.f     = expected.f + val;
current.u  = ATOMIC_CMPXCHG( (volatile global dsint *)addr, expected.u, next.u);
} while( current.u != expected.u );
}

inline void atomic_add_int(volatile global int *addr, int val){
union {int u; int f;} next, expected, current;
current.f = *addr;
do {
expected.f = current.f;
next.f     = expected.f + val;
current.u  = ATOMIC_CMPXCHG( (volatile global int *)addr, expected.u, next.u);
} while( current.u != expected.u );
}

// For testing: every global index tries to add all elements of b to the first element of a
kernel void test_atomic_add_int(global int * a, global int * b, int length){
    int gi = get_global_id(0);
    if (gi < length){
        for (int g=0; g<length; g++){
            atomic_add_int(&a[0], b[g]);
}}}

// For testing: every global index tries to add all elements of b to the first element of a
kernel void test_atomic_add_real(global dsfloat * a, global dsfloat * b, int length){
    int gi = get_global_id(0);
    if (gi < length){
        for (int g=0; g<length; g++){
            atomic_add_real(&a[0], b[g]);
}}}

//inline void atomic_add_complex(volatile global dsfloat2 *addr, dsfloat2 val)
//{
//union {dsint2 u; dsfloat2 f;} next, expected, current;
//current.f = *addr;
//do {
//expected.f = current.f;
//next.f     = expected.f + val;
//current.u  = ATOMIC_CMPXCHG( (volatile global unsigned int *)addr, expected.u, next.u);
//} while( current.u != expected.u );
//}

//static int linear_congruential(int n)
//{
//// Experimental... random number using linear congruential and constants from Numerical Recipes
//    n = (1664525*n + 1013904223) % 4294967296;
//    return n;
//}

// Rotate a vector
static dsfloat4 rotate_vec(
    const dsfloat16 R,
    const dsfloat4 v)
{
    dsfloat4 v_rot = (dsfloat4)(0.0,0.0,0.0,0.0);
    v_rot.x = R.s0*v.x + R.s1*v.y + R.s2*v.z;
    v_rot.y = R.s3*v.x + R.s4*v.y + R.s5*v.z;
    v_rot.z = R.s6*v.x + R.s7*v.y + R.s8*v.z;
    return v_rot;
}

// Rotate a vector but use the transpose of the rotation matrix
static dsfloat4 transpose_rotate_vec(
    const dsfloat16 R,
    const dsfloat4 v)
{
    dsfloat4 v_rot = (dsfloat4)(0.0,0.0,0.0,0.0);
    v_rot.x = R.s0*v.x + R.s3*v.y + R.s6*v.z;
    v_rot.y = R.s1*v.x + R.s4*v.y + R.s7*v.z;
    v_rot.z = R.s2*v.x + R.s5*v.y + R.s8*v.z;
    return v_rot;
}

// Rotate then translate a vector
kernel void rotate_translate_vectors(
    const dsfloat16 R,
    const dsfloat4 U,
    global dsfloat *v_in,
    global dsfloat *v_out,
    int n_vectors
){
    const int gi = get_global_id(0); /* Global index */
    dsfloat4 tmp;
    if (gi < n_vectors){
        dsfloat4 vec = (dsfloat4)(v_in[gi*3],v_in[gi*3+1],v_in[gi*3+2],0.0f);
        tmp = rotate_vec(R, vec) + U;
        v_out[gi*3] = tmp.x;
        v_out[gi*3+1] = tmp.y;
        v_out[gi*3+2] = tmp.z;
    }
}

// Test rotate vector
kernel void test_rotate_vec(
    const dsfloat16 R,
    const dsfloat4 U,
    const dsfloat4 v,
    global dsfloat4 *v_out
){
    const int gi = get_global_id(0); /* Global index */
    if (gi == 0){
        dsfloat4 v_temp = rotate_vec(R, v) + U;
        v_out[0].x = v_temp.x;
        v_out[0].y = v_temp.y;
        v_out[0].z = v_temp.z;
    }
}

// Test simple summation
kernel void test_simple_sum(
    global dsfloat *in,
    global dsfloat *out,
    const int n)
{
    const int gi = get_global_id(0); /* Global index */
    dsfloat tot = 0;
    if (gi == 0){
        for (int g=0; g<n; g++){
            tot = tot + in[g];
        }
        out[0] = tot;
    }
}

// Calculate the scattering vectors for a pixel-array detector
static dsfloat4 q_pad(
    const int i,       // Pixel fast-scan index
    const int j,       // Pixel slow-scan index
    const dsfloat w,   // Photon wavelength
    const dsfloat4 T,  // Translation of detector
    const dsfloat4 F,  // Fast-scan basis vector
    const dsfloat4 S,  // Slow-scan basis vector
    const dsfloat4 B   // Incident beam unit vector
){
    dsfloat4 V = T + i*F + j*S;
    V /= sqrt(dot(V,V));
    dsfloat4 q = (V-B)*PI2/w;
    return q;
}

// Given a bunch of vectors q, sum the amplitudes from a collection of atoms for a given scattering vector:
//
//    SUM_i f_i * exp(-i*q.(r_i+U))
//
// ** There is one complication: we attempt to speed up the summation by making workers move atomic coordinates
// and scattering factors to a local memory buffer in parallel, in hopes of faster computation (is it really faster?)
static dsfloat2 phase_factor(
    const dsfloat4 q,         // Scattering vector
    const dsfloat16 R,        // Rotation applied to positions
    const dsfloat4 U,         // Shift added to positions (after rotation)
    global const dsfloat *r,  // Atomic coordinates
    global const dsfloat2 *f, // Atomic scattering factors
    const int n_atoms,        // Number of atoms
    local dsfloat4 *rg,       // Local storage for chunk of atom positions          (local dsfloat4 rg[GROUP_SIZE];)
    local dsfloat2 *fg,       // Local storage for chunk of atom scattering factors (local dsfloat2 fg[GROUP_SIZE];)
    const int li              // Local index of this worker (i.e. group member ID)
){
    int ai;
    dsfloat ph, sinph, cosph;
    dsfloat2 a_temp;
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);
    for (int g=0; g<n_atoms; g+=GROUP_SIZE){
        // Here we will move a chunk of atoms to local memory.  Each worker in a
        // group moves one atom.
        ai = g+li; // Index of the global array of atoms that this worker will move in this particular iteration
        if (ai < n_atoms ){
            rg[li] = rotate_vec(R, (dsfloat4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f)) + U;
            fg[li] = f[ai];
        } else {
            rg[li] = (dsfloat4)(0.0f,0.0f,0.0f,0.0f);
            fg[li] = (dsfloat2)(0.0f,0.0f);
        }
        // Don't proceed until **all** members of the group have finished moving
        // atom information into local memory.
        barrier(CLK_LOCAL_MEM_FENCE);
        // We use a local real and imaginary part to avoid floating point overflow
        a_temp = (dsfloat2)(0.0f,0.0f);
        // Now sum up the amplitudes from this subset of atoms
        for (int n=0; n < GROUP_SIZE; n++){
            ph = -dot(q,rg[n]);
            sinph = native_sin(ph);
            cosph = native_cos(ph);
            a_temp.x += fg[n].x*cosph - fg[n].y*sinph;
            a_temp.y += fg[n].x*sinph + fg[n].y*cosph;
        }
        a_sum += a_temp;
        // Don't proceed until this subset of atoms are completed.
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return a_sum;
}


// Given a bunch of vectors q, sum the amplitudes from a collection of atoms for a given scattering vector:
//
//    SUM_i f_i * exp(-i*q.(r_i+U))
//
// ** This does not attempt to use the local memory trick (as in phase_factor_qrf)
static dsfloat2 phase_factor_global(
    const dsfloat4 q,         // Scattering vector
    const dsfloat16 R,        // Rotation applied to positions
    const dsfloat4 U,         // Shift added to positions (after rotation)
    global const dsfloat *r,  // Atomic coordinates
    global const dsfloat2 *f, // Atomic scattering factors
    const int n_atoms         // Number of atoms
){
    dsfloat ph, sinph, cosph;
    dsfloat4 rn;
    dsfloat2 fn;
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);
    for (int n=0; n<n_atoms; n++){
        rn = rotate_vec(R, (dsfloat4)(r[n*3],r[n*3+1],r[n*3+2],0.0f)) + U;
        fn = f[n];
        ph = -dot(q,rn);
        sinph = native_sin(ph);
        cosph = native_cos(ph);
        a_sum.x += fn.x*cosph - fn.y*sinph;
        a_sum.y += fn.x*sinph + fn.y*cosph;
    }
    return a_sum;
}

// Just return the group size; helper function to check that the macro is defined properly
kernel void get_group_size(
    global int *g){
    const int gi = get_global_id(0);
    if (gi == 0){
        g[gi] = GROUP_SIZE;
    }
}

// Take the mod square of an array of complex numbers.  This is supposed to be
// possible with the pyopencl.array.Array class, but I only get seg. faults...
kernel void mod_squared_complex_to_real(global const dsfloat2 *A, global dsfloat *I, const int n, const int add){
    const int gi = get_global_id(0);
    if (gi < n){
        dsfloat2 a = A[gi];
        I[gi] += I[gi]*add + a.x*a.x + a.y*a.y;
}}

// Divide A by B wherever B is not zero
kernel void divide_nonzero_inplace_real(global dsfloat *A, global const dsfloat *B, const int n){
    const int gi = get_global_id(0);
    if (gi < n){if (B[gi] != 0){A[gi] /= B[gi];}}
}

// Sum the amplitudes from a collection of atoms for given scattering vectors: SUM_i f_i * exp(i*q.r_i)
// This variant allows for an arbitrary collection of scattering vectors
kernel void phase_factor_qrf(
    global const dsfloat *q,  // Scattering vectors
    global const dsfloat *r,  // Atomic postion vectors
    global const dsfloat2 *f, // Atomic scattering factors
    const dsfloat16 R,        // Rotation matrix
    const dsfloat4 U,         // Translation vector acting on positions
    global dsfloat2 *a,       // The summed scattering amplitudes (output)
    const int n_atoms,        // Number of atoms
    const int n_pixels,       // Number of pixels
    const int add,            // Set to 1 if you wish to add to the existing amplitude (a) buffer; 0 will overwrite it
    const int twopi           // Multiply q by 2 pi
){
    const int gi = get_global_id(0); /* Global index */
    const int li = get_local_id(0);  /* Local group index */
    dsfloat4 qmod = (dsfloat4)(0.0f,0.0f,0.0f,0.0f);
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);
    local dsfloat4 rg[GROUP_SIZE];
    local dsfloat2 fg[GROUP_SIZE];
    // If the pixel index is not out of bounds, move the global scattering vector to private memory
    if (gi < n_pixels){qmod = (dsfloat4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);}
    if (twopi == 1){qmod *= PI2;}
    // Sum over atomic scattering amplitudes
    a_sum = phase_factor(qmod, R, U, r, f, n_atoms, rg, fg, li);
    // Again, check that this pixel index is not out of bounds
    if (gi < n_pixels){a[gi] = a[gi]*add + a_sum;}
}

// Sum the amplitudes from a collection of atoms for given scattering vectors: SUM_i f_i * exp(i*q.r_i)
// This variant allows for an arbitrary collection of scattering vectors
// This variant does not attempt to speed up the calculation by moving chunks to local memory.  The speedup
// due to using local memory appears to be small (~10% on Rick's laptop with onboard intel GPU).
kernel void phase_factor_qrf_global(
    global const dsfloat *q,  // Scattering vectors
    global const dsfloat *r,  // Atomic postion vectors
    global const dsfloat2 *f, // Atomic scattering factors
    const dsfloat16 R,        // Rotation matrix
    const dsfloat4 U,         // Translation vector acting on positions
    global dsfloat2 *a,       // The summed scattering amplitudes (output)
    const int n_atoms,        // Number of atoms
    const int n_pixels,       // Number of pixels
    const int add,            // Set to 1 if you wish to add to the existing amplitude (a) buffer; 0 will overwrite it
    const int twopi           // Multiply q by 2 pi
){
    const int gi = get_global_id(0); /* Global index */
    const int li = get_local_id(0);  /* Local group index */
    dsfloat4 qmod = (dsfloat4)(0.0f,0.0f,0.0f,0.0f);
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);
    // If the pixel index is not out of bounds, move the global scattering vector to private memory
    if (gi < n_pixels){qmod = (dsfloat4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);}
    if (twopi == 1){qmod *= PI2;}
    // Sum over atomic scattering amplitudes
    a_sum = phase_factor_global(qmod, R, U, r, f, n_atoms); //, rg, fg, li);
    // Again, check that this pixel index is not out of bounds
    if (gi < n_pixels){a[gi] = a[gi]*add + a_sum;}
}

// Sum the amplitudes from a collection of atoms for given scattering vectors: SUM_i f_i * exp(i*q.r_i)
// This variant internally computes scattering vectors corresponding to a pixel-array detector
kernel void phase_factor_pad(
    global const dsfloat *r,  // Atomic postion vectors
    global const dsfloat2 *f, // Atomic scattering factors
    const dsfloat16 R,        // Rotation matrix
    const dsfloat4 U,         // Translation vector acting on positions
    global dsfloat2 *a,       // The summed scattering amplitudes (output)
    const int n_pixels,       // Number of pixels
    const int n_atoms,        // Number of atoms
    const int nF,             // Number of fast-scan pixels
    const int nS,             // Number of slow-scan pixels
    const dsfloat w,          // Photon wavelength
    const dsfloat4 T,         // Translation of detector
    const dsfloat4 F,         // Fast-scan basis vector
    const dsfloat4 S,         // Slow-scan basis vector
    const dsfloat4 B,         // Incident beam unit vector
    const int add            // Set to 1 if you wish to add to the existing amplitude (a) buffer; 0 will overwrite it
){
    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;           /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */
    const int li = get_local_id(0);  /* Local group index */
    // Compute the scattering vector
    dsfloat4 qmod = q_pad(i,j,w,T,F,S,B);
    // Sum over atomic scattering amplitudes
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);
    local dsfloat4 rg[GROUP_SIZE];
    local dsfloat2 fg[GROUP_SIZE];
    a_sum = phase_factor(qmod, R, U, r, f, n_atoms, rg, fg, li);
    // Check that this pixel index is not out of bounds
    if (gi < n_pixels){a[gi] = a[gi]*add + a_sum;}
}

// Sum the amplitudes from a collection of atoms for given scattering vectors: SUM_i f_i * exp(i*q.r_i)
// This variant internally computes scattering vectors corresponding to a regular 3D grid.
// Grid points are computed according to the formula:
//     q_n = n*dq + q_min
// where n is the array index (staring with zero) for a given axis (x, y, z).  In the usual reborn
// standard, we should compute dq according to the formula
//     dq = (q_max - q_min)/(N-1)
kernel void phase_factor_mesh(
    global const dsfloat *r,   // Atomic postion vectors
    global const dsfloat2 *f,  // Atomic scattering factors
    global dsfloat2 *a,        // The summed scattering amplitudes (output)
    const int n_pixels,        // Number of pixels
    const int n_atoms,         // Number of atoms
    const int4 N,              // Number of points on the 3D grid (3 numbers specified)
    const dsfloat4 deltaQ,     // Spacings betwen grid points (3 numbers specified)
    const dsfloat4 q_min,      // Starting positions (i.e. corner) of grid (3 numbers specified)
    const dsfloat16 R,         // Rotation matrix
    const dsfloat4 U,          // Translation vector acting on positions
    const int add,             // Set to 1 if you wish to add to the existing amplitude (a) buffer; 0 will overwrite it
    const int twopi            // Multiply q by 2 pi
){
    const int gi = get_global_id(0); /* Global index */
    const int i = gi/(N.z*N.y);      /* Voxel coordinate i (x) */
    const int j = (gi/N.z) % N.y;    /* Voxel coordinate j (y) */
    const int k = gi % N.z;          /* Voxel corrdinate k (z) */
    const int li = get_local_id(0);  /* Local group index */
    // Each global index corresponds to a particular q-vector
    dsfloat4 qmod = (dsfloat4)(i*deltaQ.x+q_min.x, j*deltaQ.y+q_min.y, k*deltaQ.z+q_min.z,0.0f);
    if (twopi == 1){qmod *= PI2;}
    // Sum over atomic scattering amplitudes
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);
    local dsfloat4 rg[GROUP_SIZE];
    local dsfloat2 fg[GROUP_SIZE];
    a_sum = phase_factor(qmod, R, U, r, f, n_atoms, rg, fg, li);
    // Check that this pixel index is not out of bounds
    if (gi < n_pixels){a[gi] = a[gi]*add + a_sum;}
}

// Interpolate scattering amplitudes from a lookup table.  This is meant to be used in conjunction with the output of
// phase_factor_mesh.
kernel void mesh_interpolation(
    global dsfloat2 *a_map,  // Lookup table generated by phase_factor_mesh
    global dsfloat *q,       // Scattering vectors
    global dsfloat2 *a_out,  // The summed scattering amplitudes (output)
    int n_pixels,            // Number of pixels
    int4 N,                  // See phase_factor_mesh
    dsfloat4 deltaQ,         // See phase_factor_mesh
    dsfloat4 q_min,          // See phase_factor_mesh
    const dsfloat16 R,       // Rotation matrix
    const dsfloat4 U,        // Translation vector acting on positions
    int do_translate,        // Set to 1 to apply translation
    int add,                 // Set to 1 if you wish to add to the existing amplitude (a) buffer; 0 will overwrite it
    const int twopi          // Multiply q by 2 pi
){
    const int gi = get_global_id(0);
    dsfloat4 qmod = (dsfloat4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
    qmod = rotate_vec(R,qmod);
    if (twopi == 1){qmod *= PI2;}
    // Floating point coordinates
    const dsfloat i_f = (qmod.x - q_min.x)/deltaQ.x;
    const dsfloat j_f = (qmod.y - q_min.y)/deltaQ.y;
    const dsfloat k_f = (qmod.z - q_min.z)/deltaQ.z;
    // Integer coordinates
    const int i = (int)(floor(i_f));
    const int j = (int)(floor(j_f));
    const int k = (int)(floor(k_f));
    // Trilinear interpolation formula specified in paulbourke.net/miscellaneous/interpolation
    const int i0 = (i % N.x)*N.y*N.z;
    const int j0 = (j % N.y)*N.z;
    const int k0 = (k % N.z);
    const int i1 = ((i+1) % N.x)*N.y*N.z;
    const int j1 = ((j+1) % N.y)*N.z;
    const int k1 = ((k+1) % N.z);
    const dsfloat x0 = i_f - floor(i_f);
    const dsfloat y0 = j_f - floor(j_f);
    const dsfloat z0 = k_f - floor(k_f);
    const dsfloat x1 = 1.0f - x0;
    const dsfloat y1 = 1.0f - y0;
    const dsfloat z1 = 1.0f - z0;
    dsfloat2 a_sum = 0;
    a_sum = a_map[i0 + j0 + k0] * x1 * y1 * z1 +
            a_map[i1 + j0 + k0] * x0 * y1 * z1 +
            a_map[i0 + j1 + k0] * x1 * y0 * z1 +
            a_map[i0 + j0 + k1] * x1 * y1 * z0 +
            a_map[i1 + j0 + k1] * x0 * y1 * z0 +
            a_map[i0 + j1 + k1] * x1 * y0 * z0 +
            a_map[i1 + j1 + k0] * x0 * y0 * z1 +
            a_map[i1 + j1 + k1] * x0 * y0 * z0;
    dsfloat ph, cosph, sinph;
    dsfloat2 a_temp;
    if (do_translate == 1){
        ph = -dot(qmod,U);
        cosph = native_cos(ph);
        sinph = native_sin(ph);
        a_temp.x = a_sum.x*cosph - a_sum.y*sinph;
        a_temp.y = a_sum.x*sinph + a_sum.y*cosph;
        a_sum = a_temp;
    }
    // Check that this pixel index is not out of bounds
    if (gi < n_pixels){a_out[gi] = a_out[gi]*add + a_sum;}
}

// Interpolate scattering amplitudes from a lookup table.  This is meant to be used in conjunction with the output of
// phase_factor_mesh.  This is for real-valued input.
kernel void mesh_interpolation_real(
    global dsfloat *a_map,   // Lookup table generated by phase_factor_mesh
    global dsfloat *q,       // Scattering vectors
    global dsfloat *a_out,   // The summed scattering amplitudes (output)
    int n_pixels,            // Number of pixels
    int4 N,                  // See phase_factor_mesh
    dsfloat4 deltaQ,         // See phase_factor_mesh
    dsfloat4 q_min,          // See phase_factor_mesh
    const dsfloat16 R,       // Rotation matrix
    const dsfloat4 U,        // Translation vector acting on positions
    int do_translate,        // Set to 1 to apply translation
    int add,                 // Set to 1 if you wish to add to the existing amplitude (a) buffer; 0 will overwrite it
    const int twopi          // Multiply q by 2 pi
){
    const int gi = get_global_id(0);
    dsfloat4 qmod = (dsfloat4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
    qmod = rotate_vec(R,qmod);
    if (twopi == 1){qmod *= PI2;}
    // Floating point coordinates
    const dsfloat i_f = (qmod.x - q_min.x)/deltaQ.x;
    const dsfloat j_f = (qmod.y - q_min.y)/deltaQ.y;
    const dsfloat k_f = (qmod.z - q_min.z)/deltaQ.z;
    // Integer coordinates
    const int i = (int)(floor(i_f));
    const int j = (int)(floor(j_f));
    const int k = (int)(floor(k_f));
    // Trilinear interpolation formula specified in paulbourke.net/miscellaneous/interpolation
    const int i0 = (i % N.x)*N.y*N.z;
    const int j0 = (j % N.y)*N.z;
    const int k0 = (k % N.z);
    const int i1 = ((i+1) % N.x)*N.y*N.z;
    const int j1 = ((j+1) % N.y)*N.z;
    const int k1 = ((k+1) % N.z);
    const dsfloat x0 = i_f - floor(i_f);
    const dsfloat y0 = j_f - floor(j_f);
    const dsfloat z0 = k_f - floor(k_f);
    const dsfloat x1 = 1.0f - x0;
    const dsfloat y1 = 1.0f - y0;
    const dsfloat z1 = 1.0f - z0;
    dsfloat a_sum = 0;
    a_sum = a_map[i0 + j0 + k0] * x1 * y1 * z1 +
            a_map[i1 + j0 + k0] * x0 * y1 * z1 +
            a_map[i0 + j1 + k0] * x1 * y0 * z1 +
            a_map[i0 + j0 + k1] * x1 * y1 * z0 +
            a_map[i1 + j0 + k1] * x0 * y1 * z0 +
            a_map[i0 + j1 + k1] * x1 * y0 * z0 +
            a_map[i1 + j1 + k0] * x0 * y0 * z1 +
            a_map[i1 + j1 + k1] * x0 * y0 * z0;

    if (gi < n_pixels){
        a_out[gi] = a_out[gi]*add + a_sum;
    }
}

// The counterpart to mesh_interpolation - this one inserts intensities into the 3D mesh rather than extracting
// intensities from an existing mesh.  Experimental.  Requires atomic add.
kernel void mesh_insertion(
    global dsfloat2 *densities,  // Lookup table akin to one made by phase_factor_mesh
    global dsfloat *weights,     // Weights for insertion
    global dsfloat *vecs,        // Scattering vectors
    global dsfloat2 *vals,       // The scattering amplitudes to be inserted
    int n_pixels,                // Number of pixels
    int4 shape,                  // See phase_factor_mesh
    dsfloat4 deltas,             // See phase_factor_mesh
    dsfloat4 corner,             // See phase_factor_mesh
    const dsfloat16 R            // Rotation matrix
){
    const int gi = get_global_id(0);
    dsfloat4 vecs4r = (dsfloat4)(vecs[gi*3],vecs[gi*3+1],vecs[gi*3+2],0.0f);
    vecs4r = rotate_vec(R,vecs4r);
    const dsfloat i_f = (vecs4r.x - corner.x)/deltas.x;
    const dsfloat j_f = (vecs4r.y - corner.y)/deltas.y;
    const dsfloat k_f = (vecs4r.z - corner.z)/deltas.z;
    const int i = (int)(floor(i_f));
    const int j = (int)(floor(j_f));
    const int k = (int)(floor(k_f));
    const int i0 = (i % shape.x)*shape.y*shape.z;
    const int j0 = (j % shape.y)*shape.z;
    const int k0 = (k % shape.z);
    const int i1 = ((i+1) % shape.x)*shape.y*shape.z;
    const int j1 = ((j+1) % shape.y)*shape.z;
    const int k1 = ((k+1) % shape.z);
    const dsfloat x0 = i_f - floor(i_f);
    const dsfloat y0 = j_f - floor(j_f);
    const dsfloat z0 = k_f - floor(k_f);
    const dsfloat x1 = 1.0f - x0;
    const dsfloat y1 = 1.0f - y0;
    const dsfloat z1 = 1.0f - z0;
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i0 + j0 + k0)]), vals[gi].x * x1 * y1 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i1 + j0 + k0)]), vals[gi].x * x0 * y1 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i0 + j1 + k0)]), vals[gi].x * x1 * y0 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i0 + j0 + k1)]), vals[gi].x * x1 * y1 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i1 + j0 + k1)]), vals[gi].x * x0 * y1 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i0 + j1 + k1)]), vals[gi].x * x1 * y0 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i1 + j1 + k0)]), vals[gi].x * x0 * y0 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i1 + j1 + k1)]), vals[gi].x * x0 * y0 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i0 + j0 + k0)+1]), vals[gi].y * x1 * y1 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i1 + j0 + k0)+1]), vals[gi].y * x0 * y1 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i0 + j1 + k0)+1]), vals[gi].y * x1 * y0 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i0 + j0 + k1)+1]), vals[gi].y * x1 * y1 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i1 + j0 + k1)+1]), vals[gi].y * x0 * y1 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i0 + j1 + k1)+1]), vals[gi].y * x1 * y0 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i1 + j1 + k0)+1]), vals[gi].y * x0 * y0 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[2*(i1 + j1 + k1)+1]), vals[gi].y * x0 * y0 * z0);
    atomic_add_real(&weights[i0 + j0 + k0], x1 * y1 * z1);
    atomic_add_real(&weights[i1 + j0 + k0], x0 * y1 * z1);
    atomic_add_real(&weights[i0 + j1 + k0], x1 * y0 * z1);
    atomic_add_real(&weights[i0 + j0 + k1], x1 * y1 * z0);
    atomic_add_real(&weights[i1 + j0 + k1], x0 * y1 * z0);
    atomic_add_real(&weights[i0 + j1 + k1], x1 * y0 * z0);
    atomic_add_real(&weights[i1 + j1 + k0], x0 * y0 * z1);
    atomic_add_real(&weights[i1 + j1 + k1], x0 * y0 * z0);
}

kernel void mesh_insertion_real(
    global dsfloat *densities,   // Lookup table akin to one made by phase_factor_mesh
    global dsfloat *weights,     // Weights for insertion
    global dsfloat *vecs,        // Scattering vectors
    global dsfloat *vals,        // The scattering amplitudes to be inserted
    int n_pixels,                // Number of pixels
    int4 shape,                  // See phase_factor_mesh
    dsfloat4 deltas,             // See phase_factor_mesh
    dsfloat4 corner,             // See phase_factor_mesh
    const dsfloat16 R            // Rotation matrix
){
    const int gi = get_global_id(0);
    dsfloat4 vecs4r = (dsfloat4)(vecs[gi*3],vecs[gi*3+1],vecs[gi*3+2],0.0f);
    vecs4r = rotate_vec(R,vecs4r);
    const dsfloat i_f = (vecs4r.x - corner.x)/deltas.x;
    const dsfloat j_f = (vecs4r.y - corner.y)/deltas.y;
    const dsfloat k_f = (vecs4r.z - corner.z)/deltas.z;
    const int i = (int)(floor(i_f));
    const int j = (int)(floor(j_f));
    const int k = (int)(floor(k_f));
    const int i0 = (i % shape.x)*shape.y*shape.z;
    const int j0 = (j % shape.y)*shape.z;
    const int k0 = (k % shape.z);
    const int i1 = ((i+1) % shape.x)*shape.y*shape.z;
    const int j1 = ((j+1) % shape.y)*shape.z;
    const int k1 = ((k+1) % shape.z);
    const dsfloat x0 = i_f - floor(i_f);
    const dsfloat y0 = j_f - floor(j_f);
    const dsfloat z0 = k_f - floor(k_f);
    const dsfloat x1 = 1.0f - x0;
    const dsfloat y1 = 1.0f - y0;
    const dsfloat z1 = 1.0f - z0;
    if (gi < n_pixels){
    atomic_add_real(&(((volatile global dsfloat *)densities)[i0 + j0 + k0]), vals[gi] * x1 * y1 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[i1 + j0 + k0]), vals[gi] * x0 * y1 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[i0 + j1 + k0]), vals[gi] * x1 * y0 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[i0 + j0 + k1]), vals[gi] * x1 * y1 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[i1 + j0 + k1]), vals[gi] * x0 * y1 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[i0 + j1 + k1]), vals[gi] * x1 * y0 * z0);
    atomic_add_real(&(((volatile global dsfloat *)densities)[i1 + j1 + k0]), vals[gi] * x0 * y0 * z1);
    atomic_add_real(&(((volatile global dsfloat *)densities)[i1 + j1 + k1]), vals[gi] * x0 * y0 * z0);
    atomic_add_real(&weights[i0 + j0 + k0], x1 * y1 * z1);
    atomic_add_real(&weights[i1 + j0 + k0], x0 * y1 * z1);
    atomic_add_real(&weights[i0 + j1 + k0], x1 * y0 * z1);
    atomic_add_real(&weights[i0 + j0 + k1], x1 * y1 * z0);
    atomic_add_real(&weights[i1 + j0 + k1], x0 * y1 * z0);
    atomic_add_real(&weights[i0 + j1 + k1], x1 * y0 * z0);
    atomic_add_real(&weights[i1 + j1 + k0], x0 * y0 * z1);
    atomic_add_real(&weights[i1 + j1 + k1], x0 * y0 * z0);
    }
}

// Compute ideal lattice transform for parallelepiped crystal: PROD_i  [ sin(N_i x_i)/sin(x_i) ]^2
// This variant internally computes scattering vectors for a pixel-array detector.
kernel void lattice_transform_intensities_pad(
    const dsfloat16 abc,   // Real-space lattice vectors a,b,c, each contiguous in memory
    const int4 N,          // Number of unit cells along each axis
    const dsfloat16 R,     // Rotation matrix
    global dsfloat *I,     // Lattice transform intensities (output)
    const int n_pixels,    // Number of pixels
    const int nF,          // Refer to phase_factor_pad
    const int nS,          // Refer to phase_factor_pad
    const dsfloat w,       // Refer to phase_factor_pad
    const dsfloat4 T,      // Refer to phase_factor_pad
    const dsfloat4 F,      // Refer to phase_factor_pad
    const dsfloat4 S,      // Refer to phase_factor_pad
    const dsfloat4 B,      // Refer to phase_factor_pad
    const int add          // Refer to phase_factor_pad
){
    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;           /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */
    // Get the q vector
    dsfloat4 qmod = q_pad( i,j,w,T,F,S,B);
    // Rotate the q vector
    qmod = rotate_vec(R, qmod);
    // Compute lattice transform at this q vector
    dsfloat sn;
    dsfloat s;
    dsfloat sns;
    dsfloat x;
    dsfloat n;
    dsfloat4 a;
    dsfloat It = 1.0;
    // First crystal axis (this could be put in a loop over three axes...)
    n = (dsfloat)N.x;
    a = (dsfloat4)(abc.s0,abc.s1,abc.s2,0.0);
    x = dot(qmod,a) / 2.0;
    if (x != 0){
        // This does [ sin(Nx)/sin(x) ]^2
        sn = sin(n*x);
        s = sin(x);
        sns = sn/s;
        It *= sns*sns;
    } else {
        // This handles potential divide by zero
        It *= n*n;
    }
    // Second crystal axis
    n = (dsfloat)N.y;
    a = (dsfloat4)(abc.s3,abc.s4,abc.s5,0.0);
    x = dot(qmod,a) / 2.0;
    if (x != 0){
        // This does [ sin(Nx)/sin(x) ]^2
        sn = sin(n*x);
        s = sin(x);
        sns = sn/s;
        It *= sns*sns;
    } else {
        // This handles potential divide by zero
        It *= n*n;
    }
    // Third crystal axis
    n = (dsfloat)N.z;
    a = (dsfloat4)(abc.s6,abc.s7,abc.s8,0.0);
    x = dot(qmod,a) / 2.0;
    if (x != 0){
        // This does [ sin(Nx)/sin(x) ]^2
        sn = sin(n*x);
        s = sin(x);
        sns = sn/s;
        It *= sns*sns;
    } else {
        // This handles potential divide by zero
        It *= n*n;
    }
    if (gi < n_pixels ){
        if (add == 1){
            I[gi] += It;
        } else {
            I[gi] = It;
        }
    }
}

// Compute approximate gaussian-shaped lattice transform for a crystal: PROD_i  N_i^2 exp(- N_i^2 x_i^2 / 4pi)
// This variant internally computes scattering vectors for a pixel-array detector
kernel void gaussian_lattice_transform_intensities_pad(
    const dsfloat16 abc,   // Real-space lattice vectors a,b,c, each contiguous in memory
    const int4 N,          // Number of unit cells along each axis
    const dsfloat16 R,     // Rotation matrix
    global dsfloat *I,     // Lattice transform intensities (output)
    const int n_pixels,    // Number of pixels
    const int nF,          // Refer to phase_factor_pad
    const int nS,          // Refer to phase_factor_pad
    const dsfloat w,       // Refer to phase_factor_pad
    const dsfloat4 T,      // Refer to phase_factor_pad
    const dsfloat4 F,      // Refer to phase_factor_pad
    const dsfloat4 S,      // Refer to phase_factor_pad
    const dsfloat4 B,      // Refer to phase_factor_pad
    const int add          // Refer to phase_factor_pad
){
    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;           /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */
    // Get the q vector
    dsfloat4 qmod = q_pad( i,j,w,T,F,S,B);
    // Rotate the q vector
    qmod = rotate_vec(R, qmod);
    // Compute lattice transform at this q vector
    dsfloat x;
    dsfloat n;
    dsfloat4 a;
    dsfloat It = 1.0;
    // First crystal axis (this could be put in a loop over three axes...)
    n = (dsfloat)N.x;
    a = (dsfloat4)(abc.s0,abc.s1,abc.s2,0.0);
    x = dot(qmod,a);
    x = x - round(x/PI2)*PI2;
    It *= n*n*exp(-n*n*x*x/(4*PI));
    // Second crystal axis
    n = (dsfloat)N.y;
    a = (dsfloat4)(abc.s3,abc.s4,abc.s5,0.0);
    x = dot(qmod,a);
    x = x - round(x/PI2)*PI2;
    It *= n*n*exp(-n*n*x*x/(4*PI));
    // Third crystal axis
    n = (dsfloat)N.z;
    a = (dsfloat4)(abc.s6,abc.s7,abc.s8,0.0);
    x = dot(qmod,a);
    x = x - round(x/PI2)*PI2;
    It *= n*n*exp(-n*n*x*x/(4*PI));
    if (gi < n_pixels ){
        if (add == 1){
            I[gi] += It;
        } else {
            I[gi] = It;
        }
    }
}

// Compute form factor for a sphere.  Assumes complex amplitude array.
kernel void sphere_form_factor(
    global const dsfloat *q,       // Scattering magnitudes
    global dsfloat2 *amps,   // Amplitude array (input/output)
    const dsfloat r,               // The radius of the sphere
    const int n_q,                 // Number of q magnitudes
    const int add                  // Add to the amplitude buffer
){
    const int gi = get_global_id(0); /* Global index */
    dsfloat qr = q[gi]*r;
    dsfloat a;
    if (qr == 0){
        a = (4*PI*r*r*r)/3;
    } else {
        a = 4*PI*r*r*r*(native_sin(qr)-qr*native_cos(qr))/(qr*qr*qr);
    }
    if (gi < n_q ){
        if (add == 1){
            amps[gi].x += a;
        } else {
            amps[gi].x = a;
            amps[gi].y = 0;
        }
    }
}

//// Compute approximate gaussian-shaped lattice transform for a crystal convolved
//// with a normal distribution for mosaicity: PROD_i  N_i^2 exp(- N_i^2 x_i^2 / 4pi)
//// This variant internallly computes scattering vectors for a pixel-array detector
//
//kernel void mosaic_gaussian_lattice_transform_intensities_pad(
//    const dsfloat16 abc,   // Real-space lattice vectors a,b,c, each contiguous in memory
//    const dsfloat16 R,     // Rotation matrix
//    global dsfloat *S_eff,     // Effective Lattice transform  (output)
//    const dsfloat D,       // Crystal size N*a
//    const dsfloat sigm,   // Crystal mosaicity standard deviation
//    const dsfloat n,     // Number of sigmas to tolerate being within for Bragg peaks
//    const int n_pixels,  // Number of pixels
//    const int nF,        // Refer to phase_factor_pad
//    const int nS,        // Refer to phase_factor_pad
//    const dsfloat w,       // Refer to phase_factor_pad
//    const dsfloat4 T,      // Refer to phase_factor_pad
//    const dsfloat4 F,      // Refer to phase_factor_pad
//    const dsfloat4 S,      // Refer to phase_factor_pad
//    const dsfloat4 B,      // Refer to phase_factor_pad
//    const int add        // Refer to phase_factor_pad
//){
//
//    const int gi = get_global_id(0); /* Global index */
//    const int i = gi % nF;           /* Pixel coordinate i */
//    const int j = gi/nF;             /* Pixel coordinate j */
//
//    // Get the q vector
//    dsfloat4 q4r = q_pad( i,j,w,T,F,S,B);
//
//    // Rotate the q vector
//    q4r = rotate_vec(R, q4r);
//
//    // Calculate reciprocal lattice vectors
//    dsfloat4 a;
//    dsfloat4 b;
//    dsfloat4 c;
//    dsfloat4 astar;
//    dsfloat4 bstar;
//    dsfloat4 cstar;
//    dsfloat denom;
//
//    a = (dsfloat4)(abc.s0,abc.s1,abc.s2,0.0);
//    b = (dsfloat4)(abc.s3,abc.s4,abc.s5,0.0);
//    c = (dsfloat4)(abc.s6,abc.s7,abc.s5,0.0);
//    (dsfloat)denom = dot(a, cross(b, c))
//    astar = cross(b, c)/denom;
//    bstar = cross(c, a)/denom;
//    cstar = cross(a, b)/denom;
//
//    // Find hkl of q vector
//    // TODO: Decompose q vectors into astar,bstar,cstar basis vectors
//
//    // Find nearest 8 neighboring Bragg peaks
//    dsfloat4 hkl1;
//    dsfloat4 hkl2;
//    dsfloat4 hkl3;
//    dsfloat4 hkl4;
//    dsfloat4 hkl5;
//    dsfloat4 hkl6;
//    dsfloat4 hkl7;
//    dsfloat4 hkl8;
//    dsfloat8 hkls;
//
//    hkl1 = (floor(h), floor(k), floor(l), 0.0);
//    hkl2 = (floor(h), floor(k), ceil(l), 0.0);
//    hkl3 = (floor(h), ceil(k), floor(l), 0.0);
//    hkl4 = (floor(h), ceil(k), ceil(l), 0.0);
//    hkl5 = (ceil(h), floor(k), floor(l), 0.0);
//    hkl6 = (ceil(h), floor(k), ceil(l), 0.0);
//    hkl7 = (ceil(h), ceil(k), floor(l), 0.0);
//    hkl8 = (ceil(h), ceil(k), ceil(l), 0.0);
//
//    hkls = (hkl1,hkl2,hkl3,hkl4,hkl5,hkl6,hkl7,hkl8);
//
//    // Allocate memory
//    dsfloat dq;
//    dsfloat sigq;
//    dsfloat dtheta;
//    dsfloat sigtheta;
//    bool valid;
//    S = 0.0;
//
//    for(int i=0, i < 8, i++)
//    {
//        // Calculate Gaussian parameters
//        dq = length(q - hkl);
//        sigq = sqrt(PI2)/D;
//        dtheta = acos(normalize(q), normalize(hkl));
//        sigtheta = atan(sigq/length(q));
//        valid = (dq / sigq) < (n*sigq);
//
//        // Calculate S_eff
//        if(valid)
//        {
//            S += 1.0; // TODO: PUT FORMULA HERE
//        }
//    }
//
//    if (gi < n_pixels ){
//        if (add == 1){
//            S_eff[gi] += S;
//        } else {
//            S_eff[gi] = S;
//        }
//    }
//}

__kernel void qrf_cromer_mann(
    __global dsfloat16 *q_vecs, // reciprocal space vectors, followed by cromer mann lookups, see clcore.py for details
    __global dsfloat4 *r_vecs,  // atom vectors and atomic number
    __constant dsfloat *R,      // rotation matrix acting on atom coordinates
    __constant dsfloat *T,      // translation vector of molecule moving its center of mass
    __global dsfloat2 *A,       // amplitudes output vector
    const int n_atoms,
    const int twopi             // Multiply q by 2 pi
){
    int q_idx = get_global_id(0);
    int l_idx = get_local_id(0);
    //dsfloat Areal=0.0f;
    //dsfloat Aimag=0.0f;
    dsfloat Areal;
    dsfloat Aimag;
    dsfloat ff[16];
    // multiply trans vector by inverse rotation matrix  
    dsfloat Tx = R[0]*T[0] + R[3]*T[1] + R[6]*T[2];
    dsfloat Ty = R[1]*T[0] + R[4]*T[1] + R[7]*T[2];
    dsfloat Tz = R[2]*T[0] + R[5]*T[1] + R[8]*T[2];
    Areal=A[q_idx].x;
    Aimag=A[q_idx].y;
    dsfloat qx = q_vecs[q_idx].s0;
    dsfloat qy = q_vecs[q_idx].s1;
    dsfloat qz = q_vecs[q_idx].s2;
    dsfloat qRx = R[0]*qx + R[3]*qy + R[6]*qz;
    dsfloat qRy = R[1]*qx + R[4]*qy + R[7]*qz;
    dsfloat qRz = R[2]*qx + R[5]*qy + R[8]*qz;
    ff[0] = q_vecs[q_idx].s3;
    ff[1] = q_vecs[q_idx].s4;
    ff[2] = q_vecs[q_idx].s5;
    ff[3] = q_vecs[q_idx].s6;
    ff[4] = q_vecs[q_idx].s7;
    ff[5] = q_vecs[q_idx].s8;
    ff[6] = q_vecs[q_idx].s9;
    ff[7] = q_vecs[q_idx].sA;
    ff[8] = q_vecs[q_idx].sB;
    ff[9] = q_vecs[q_idx].sC;
    ff[10] = q_vecs[q_idx].sD;
    ff[11] = q_vecs[q_idx].sE;
    ff[12] = q_vecs[q_idx].sF;
    ff[13] = 0.0f;
    ff[14] = 0.0f;
    ff[15] = 0.0f;
    __local dsfloat4 LOC_ATOMS[GROUP_SIZE];
    for (int g=0; g<n_atoms; g+=GROUP_SIZE){
        int ai = g + l_idx;
        if (ai < n_atoms)
            LOC_ATOMS[l_idx] = r_vecs[ai];
        if( !(ai < n_atoms))
            LOC_ATOMS[l_idx] = (dsfloat4)(1.0f, 1.0f, 1.0f, 15.0f); // make atom ID 15, s.t. ff=0
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i=0; i< GROUP_SIZE; i++){
            dsfloat phase = qRx*(LOC_ATOMS[i].x+Tx) + qRy*(LOC_ATOMS[i].y+Ty) + qRz*(LOC_ATOMS[i].z+Tz);
            int species_id = (int) (LOC_ATOMS[i].w);
            Areal += native_cos(-phase)*ff[species_id];
            Aimag += native_sin(-phase)*ff[species_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    A[q_idx].x = Areal;
    A[q_idx].y = Aimag;
}

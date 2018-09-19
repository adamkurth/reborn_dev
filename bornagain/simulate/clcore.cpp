
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
#endif

// Note that dsfloat* (i.e. double/single) types will become either double or single as configured above
// Please be mindful of this in the code below

#if defined(DOUBLE_SUPPORT_AVAILABLE)
    typedef double dsfloat;
    typedef double2 dsfloat2;
    typedef double4 dsfloat4;
    typedef double16 dsfloat16;
    #define PI 3.14159265358979323846
    #define PI2 6.28318530717958647693
#else
    typedef float dsfloat;
    typedef float2 dsfloat2;
    typedef float4 dsfloat4;
    typedef float16 dsfloat16;
    #define PI 3.14159265359f
    #define PI2 6.28318530718f
#endif


static int linear_congruential(int n)
{
// Experimental... random number using linear congruential and constants from Numerical Recipes
    n = (1664525*n + 1013904223) % 4294967296;
    return n;
}


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


// Calculate the scattering vectors for a pixel-array detector

static dsfloat4 q_pad(
    const int i,     // Pixel fast-scan index
    const int j,     // Pixel slow-scan index
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


// Sum the amplitudes from a collection of atoms for a given scattering vector: SUM_i f_i * exp(i*q.r_i)
// ** There is one complication: we attempt to speed up the summation by making workers move atomic coordinates
// and scattering factors to a local memory buffer in parallele, in hopes of faster computation (is it really faster?)

static dsfloat2 phase_factor(
    const dsfloat4 q,         // Scattering vector
    global const dsfloat *r,  // Atomic coordinates
    global const dsfloat2 *f, // Atomic scattering factors
    const int n_atoms,        // Number of atoms
    local dsfloat4 *rg,       // Local storage for chunk of atom positions          (local dsfloat4 rg[GROUP_SIZE];)
    local dsfloat2 *fg,       // Local storage for chunk of atom scattering factors (local dsfloat2 fg[GROUP_SIZE];)
    const int li              // Local index of this worker (i.e. group member ID)
)
{

    int ai;
    dsfloat ph, sinph, cosph;
    dsfloat2 a_temp;
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);

    for (int g=0; g<n_atoms; g+=GROUP_SIZE){

        // Here we will move a chunk of atoms to local memory.  Each worker in a
        // group moves one atom.

        ai = g+li; // Index of the global array of atoms that this worker will move in this particular iteration

        if (ai < n_atoms ){
            rg[li] = (dsfloat4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
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


kernel void get_group_size(
    global int *g){
    const int gi = get_global_id(0);
    if (gi == 0){
        g[gi] = GROUP_SIZE;
    }
}


kernel void mod_squared_complex_to_real(
        global const dsfloat2 *A,
        global dsfloat *I,
        const int n)
{

// Take the mod square of an array of complex numbers.  This is supposed to be
// done with the pyopencl.array.Array class, but I only get seg. faults...

    const int gi = get_global_id(0);
    if (gi < n){
        dsfloat2 a = A[gi];
        I[gi] += a.x*a.x + a.y*a.y;
    }
}


// Sum the amplitudes from a collection of atoms for given scattering vectors: SUM_i f_i * exp(i*q.r_i)
// This variant allows for an arbitrary collection of scattering vectors

kernel void phase_factor_qrf(
    global const dsfloat *q,  // Scattering vectors
    global const dsfloat *r,  // Atomic postion vectors
    global const dsfloat2 *f, // Atomic scattering factors
    const dsfloat16 R,        // Rotation matrix
    global dsfloat2 *a,       // The summed scattering amplitudes (output)
    const int n_atoms,      // Number of atoms
    const int n_pixels,     // Number of pixels
    const int add)          // Set to 1 if you wish to add to the existing amplitude (a) buffer; 0 will overwrite it
{
    const int gi = get_global_id(0); /* Global index */
    const int li = get_local_id(0);  /* Local group index */

    dsfloat4 q4r;

    // If the pixel index is not out of bounds...
    if (gi < n_pixels){

        // Move the global scattering vector to private memory
        q4r = (dsfloat4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);

        // Rotate the scattering vector
        q4r = rotate_vec(R,q4r);

    } else { q4r = (dsfloat4)(0.0f,0.0f,0.0f,0.0f); }

    // Sum over atomic scattering amplitudes
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);
    local dsfloat4 rg[GROUP_SIZE];
    local dsfloat2 fg[GROUP_SIZE];
    a_sum = phase_factor(q4r, r, f, n_atoms, rg, fg, li);

    // Again, check that this pixel index is not out of bounds
    if (gi < n_pixels){
        if ( add == 1 ){
            a[gi] += a_sum;
        } else {
            a[gi] = a_sum;
        }
    }

}


// Sum the amplitudes from a collection of atoms for given scattering vectors: SUM_i f_i * exp(i*q.r_i)
// This variant internally computes scattering vectors corresponding to a pixel-array detector

kernel void phase_factor_pad(
    global const dsfloat *r,  // Atomic postion vectors
    global const dsfloat2 *f, // Atomic scattering factors
    const dsfloat16 R,        // Rotation matrix
    global dsfloat2 *a,       // The summed scattering amplitudes (output)
    const int n_pixels,     // Number of pixels
    const int n_atoms,      // Number of atoms
    const int nF,           // Number of fast-scan pixels
    const int nS,           // Number of slow-scan pixels
    const dsfloat w,          // Photon wavelength
    const dsfloat4 T,         // Translation of detector
    const dsfloat4 F,         // Fast-scan basis vector
    const dsfloat4 S,         // Slow-scan basis vector
    const dsfloat4 B,         // Incident beam unit vector
    const int add          // Set to 1 if you wish to add to the existing amplitude (a) buffer; 0 will overwrite it
){

    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;           /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */
    const int li = get_local_id(0);  /* Local group index */

    // Compute the scattering vector
    dsfloat4 q4r = q_pad(i,j,w,T,F,S,B);

    // Rotate the scattering vector
    q4r = rotate_vec(R, q4r);

    // Sum over atomic scattering amplitudes
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);
    local dsfloat4 rg[GROUP_SIZE];
    local dsfloat2 fg[GROUP_SIZE];
    a_sum = phase_factor(q4r, r, f, n_atoms, rg, fg, li);

    // Check that this pixel index is not out of bounds
    if (gi < n_pixels){
        if ( add == 1 ){
            a[gi] += a_sum;
        } else {
            a[gi] = a_sum;
        }
    }
}


// Sum the amplitudes from a collection of atoms for given scattering vectors: SUM_i f_i * exp(i*q.r_i)
// This variant internally computes scattering vectors corresponding to a regular 3D grid

kernel void phase_factor_mesh(
    global const dsfloat *r,   // Atomic postion vectors
    global const dsfloat2 *f,  // Atomic scattering factors
    global dsfloat2 *a,        // The summed scattering amplitudes (output)
    const int n_pixels,      // Number of pixels
    const int n_atoms,       // Number of atoms
    const int4 N,            // Number of points on the 3D grid (3 numbers specified)
    const dsfloat4 deltaQ,     // Spacings betwen grid points (3 numbers specified)
    const dsfloat4 q_min       // Starting positions (i.e. corner) of grid (3 numbers specified)
){

    const int Nxy = N.x*N.y;
    const int gi = get_global_id(0); /* Global index */
    const int i = gi % N.x;          /* Voxel coordinate i (x) */
    const int j = (gi/N.x) % N.y;    /* Voxel coordinate j (y) */
    const int k = gi/Nxy;            /* Voxel corrdinate k (z) */
    const int li = get_local_id(0);  /* Local group index */

    // Each global index corresponds to a particular q-vector
    dsfloat4 q4r = (dsfloat4)(i*deltaQ.x+q_min.x,
            j*deltaQ.y+q_min.y,
            k*deltaQ.z+q_min.z,0.0f);

    // Sum over atomic scattering amplitudes
    dsfloat2 a_sum = (dsfloat2)(0.0f,0.0f);
    local dsfloat4 rg[GROUP_SIZE];
    local dsfloat2 fg[GROUP_SIZE];
    a_sum = phase_factor(q4r, r, f, n_atoms, rg, fg, li);

    if (gi < n_pixels){
        a[gi] = a_sum;
    }
}


// Interpolate scattering amplitudes from a lookup table.  This is meant to be used in conjunction with the output of
// phase_factor_mesh.

kernel void buffer_mesh_lookup(
    global dsfloat2 *a_map,  // Lookup table generated by phase_factor_mesh
    global dsfloat *q,       // Scattering vectors
    global dsfloat2 *a_out,  // The summed scattering amplitudes (output)
    int n_pixels,          // Number of pixels
    int4 N,                // See phase_factor_mesh
    dsfloat4 deltaQ,         // See phase_factor_mesh
    dsfloat4 q_min,          // See phase_factor_mesh
    const dsfloat16 R        // Rotation matrix
){

    const int gi = get_global_id(0);

    dsfloat4 q4r = (dsfloat4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
    q4r = rotate_vec(R,q4r);

    // Floating point coordinates
    const dsfloat i_f = (q4r.x - q_min.x)/deltaQ.x;
    const dsfloat j_f = (q4r.y - q_min.y)/deltaQ.y;
    const dsfloat k_f = (q4r.z - q_min.z)/deltaQ.z;

    // Integer coordinates
    const int i = (int)(floor(i_f));
    const int j = (int)(floor(j_f));
    const int k = (int)(floor(k_f));

    // Trilinear interpolation formula specified in
    //     paulbourke.net/miscellaneous/interpolation
    const int k0 = k*N.x*N.y;
    const int j0 = j*N.x;
    const int i0 = i;
    const int k1 = (k+1)*N.x*N.y;
    const int j1 = (j+1)*N.x;
    const int i1 = i+1;
    const dsfloat x0 = i_f - floor(i_f);
    const dsfloat y0 = j_f - floor(j_f);
    const dsfloat z0 = k_f - floor(k_f);
    const dsfloat x1 = 1.0f - x0;
    const dsfloat y1 = 1.0f - y0;
    const dsfloat z1 = 1.0f - z0;

    if (i >= 0 && i < N.x && j >= 0 && j < N.y && k >= 0 && k < N.z){

        a_out[gi] = a_map[i0 + j0 + k0] * x1 * y1 * z1 +

                    a_map[i1 + j0 + k0] * x0 * y1 * z1 +
                    a_map[i0 + j1 + k0] * x1 * y0 * z1 +
                    a_map[i0 + j0 + k1] * x1 * y1 * z0 +

                    a_map[i0 + j1 + k1] * x1 * y0 * z0 +
                    a_map[i1 + j0 + k1] * x0 * y1 * z0 +
                    a_map[i1 + j1 + k0] * x0 * y0 * z1 +

                    a_map[i1 + j1 + k1] * x0 * y0 * z0   ;

    } else {
        a_out[gi] = (dsfloat2)(0.0f,0.0f);
    }

}


// Compute ideal lattice transform for parallelepiped crystal: PROD_i  [ sin(N_i x_i)/sin(x_i) ]^2
// This variant internallly computes scattering vectors for a pixel-array detector

kernel void lattice_transform_intensities_pad(
    const dsfloat16 abc,   // Real-space lattice vectors a,b,c, each contiguous in memory
    const int4 N,        // Number of unit cells along each axis
    const dsfloat16 R,     // Rotation matrix
    global dsfloat *I,     // Lattice transform intensities (output)
    const int n_pixels,  // Number of pixels
    const int nF,        // Refer to phase_factor_pad
    const int nS,        // Refer to phase_factor_pad
    const dsfloat w,       // Refer to phase_factor_pad
    const dsfloat4 T,      // Refer to phase_factor_pad
    const dsfloat4 F,      // Refer to phase_factor_pad
    const dsfloat4 S,      // Refer to phase_factor_pad
    const dsfloat4 B,      // Refer to phase_factor_pad
    const int add        // Refer to phase_factor_pad
){

    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;           /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */

    // Get the q vector
    dsfloat4 q4r = q_pad( i,j,w,T,F,S,B);

    // Rotate the q vector
    q4r = rotate_vec(R, q4r);

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
    x = dot(q4r,a) / 2.0;
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
    x = dot(q4r,a) / 2.0;
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
    x = dot(q4r,a) / 2.0;
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
// This variant internallly computes scattering vectors for a pixel-array detector

kernel void gaussian_lattice_transform_intensities_pad(
    const dsfloat16 abc,   // Real-space lattice vectors a,b,c, each contiguous in memory
    const int4 N,        // Number of unit cells along each axis
    const dsfloat16 R,     // Rotation matrix
    global dsfloat *I,     // Lattice transform intensities (output)
    const int n_pixels,  // Number of pixels
    const int nF,        // Refer to phase_factor_pad
    const int nS,        // Refer to phase_factor_pad
    const dsfloat w,       // Refer to phase_factor_pad
    const dsfloat4 T,      // Refer to phase_factor_pad
    const dsfloat4 F,      // Refer to phase_factor_pad
    const dsfloat4 S,      // Refer to phase_factor_pad
    const dsfloat4 B,      // Refer to phase_factor_pad
    const int add        // Refer to phase_factor_pad
){

    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;           /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */

    // Get the q vector
    dsfloat4 q4r = q_pad( i,j,w,T,F,S,B);

    // Rotate the q vector
    q4r = rotate_vec(R, q4r);

    // Compute lattice transform at this q vector
    dsfloat x;
    dsfloat n;
    dsfloat4 a;
    dsfloat It = 1.0;

    // First crystal axis (this could be put in a loop over three axes...)
    n = (dsfloat)N.x;
    a = (dsfloat4)(abc.s0,abc.s1,abc.s2,0.0);
    x = dot(q4r,a);
    x = x - round(x/PI2)*PI2;
    It *= n*n*exp(-n*n*x*x/(4*PI));

    // Second crystal axis
    n = (dsfloat)N.y;
    a = (dsfloat4)(abc.s3,abc.s4,abc.s5,0.0);
    x = dot(q4r,a);
    x = x - round(x/PI2)*PI2;
    It *= n*n*exp(-n*n*x*x/(4*PI));

    // Third crystal axis
    n = (dsfloat)N.z;
    a = (dsfloat4)(abc.s6,abc.s7,abc.s8,0.0);
    x = dot(q4r,a);
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

__kernel void qrf_default(
    __global dsfloat16 *q_vecs,
    __global dsfloat4 *r_vecs,
    __constant dsfloat *R,
    __global dsfloat2 *A,
    const int n_atoms)
// TODO: Derek will add documentation to this one.
{

    int q_idx = get_global_id(0);
    int l_idx = get_local_id(0);

    dsfloat Areal;
    dsfloat Aimag;

    dsfloat ff[16];
    
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

            dsfloat phase = qRx*LOC_ATOMS[i].x + qRy*LOC_ATOMS[i].y + qRz*LOC_ATOMS[i].z;
            int species_id = (int) (LOC_ATOMS[i].w);
            
            Areal += native_cos(-phase)*ff[species_id];
            Aimag += native_sin(-phase)*ff[species_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        }
   
    A[q_idx].x = Areal;
    A[q_idx].y = Aimag;
}


__kernel void qrf_kam(
    __global dsfloat16 *q_vecs,
    __global dsfloat4 *r_vecs,
    __constant dsfloat *R,
    __constant dsfloat *T,
    __global dsfloat2 *A,
    const int n_atoms)
// TODO: Derek will add documentation to this one.
{

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

            dsfloat phase = qRx*(LOC_ATOMS[i].x+Tx) + qRy*(LOC_ATOMS[i].y+Ty) +
                qRz*(LOC_ATOMS[i].z+Tz);
            int species_id = (int) (LOC_ATOMS[i].w);
            
            Areal += native_cos(-phase)*ff[species_id];
            Aimag += native_sin(-phase)*ff[species_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    A[q_idx].x = Areal;
    A[q_idx].y = Aimag;
}

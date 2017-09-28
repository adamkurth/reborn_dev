
// Group size may be set with a flag a compile time
#ifndef GROUP_SIZE
    #define GROUP_SIZE 1
#endif

// We allow for either double or single precision calculations.  As of now,
// this only affects floating-point numbers, not integers.  Do note that the
// Python layer will search-and-replace all "float" strings with "double"
// before compiling this.
#if CONFIG_USE_DOUBLE
    #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #define DOUBLE_SUPPORT_AVAILABLE
    #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #define DOUBLE_SUPPORT_AVAILABLE
    #endif
#endif // CONFIG_USE_DOUBLE

//#if defined(DOUBLE_SUPPORT_AVAILABLE)
    // double
//    typedef double float;
//    typedef double2 float2;
//    typedef double4 float4;
//    typedef double16 float16;
//    #define PI 3.14159265358979323846
//    #define PI2 6.28318530717958647693
//#else
    // float
//    typedef float float;
//    typedef float2 float2;
//    typedef float4 float4;
//    typedef float16 float16;
#define PI 3.14159265359f
#define PI2 6.28318530718f
//#endif


static float4 rotate_vec(
    __constant float *R, //alternatively.. 
    const float4 q)
// Please, let's always rotate vectors in this way... this is meant to act
// on the q vectors, so that it is done once per global id rather than being
// done on every atom, which would take more compute time
{
    float4 qout = (float4)(0.0,0.0,0.0,0.0);

    qout.x = R[0]*q.x + R[1]*q.y + R[2]*q.z;
    qout.y = R[3]*q.x + R[4]*q.y + R[5]*q.z;
    qout.z = R[6]*q.x + R[7]*q.y + R[8]*q.z;
    
    return qout;
}

static float4 rotate_vec2(
    const float16 R, //alternatively..
    const float4 q4)
// Please, let's always rotate vectors in this way... this is meant to act
// on the q vectors, so that it is done once per global id rather than being
// done on every atom, which would take more compute time
{
    float4 q4r = (float4)(0.0,0.0,0.0,0.0);

    // Rotate the q vector
    q4r.x = R.s0*q4.x + R.s1*q4.y + R.s2*q4.z;
    q4r.y = R.s3*q4.x + R.s4*q4.y + R.s5*q4.z;
    q4r.z = R.s6*q4.x + R.s7*q4.y + R.s8*q4.z;

    return q4r;
}

static float4 q_pad(
    const int i,
    const int j,
    const float w,
    __constant float *T,
    __constant float *F,
    __constant float *S,
    __constant float *B
    )
// Calculate the q vectors for a pixel-array detector
//
// Input:
// i, j are the pixel indices
// w is the photon wavelength
// T is the translation vector from origin to center of corner pixel
// F, S are the fast/slow-scan basis vectors (pointing alont rows/columns)
//      the length of these vectors is the pixel size
// B is the direction of the incident beam
//
// Output: A single q vector
{
    float4 q = (float4)(0.0f,0.0f,0.0f,0.0f);

    float Vx,Vy,Vz,Vnorm;
    
    Vx = T[0] + i*F[0] + j*S[0];
    Vy = T[1] + i*F[1] + j*S[1];
    Vz = T[2] + i*F[2] + j*S[2];
    Vnorm = sqrt(Vx*Vx + Vy*Vy + Vz*Vz);
    Vx = Vx/Vnorm; 
    Vy = Vy/Vnorm; 
    Vz = Vz/Vnorm; 
    
    q.x = (Vx-B[0])*PI2/w;
    q.y = (Vy-B[1])*PI2/w;
    q.z = (Vz-B[2])*PI2/w;

    return q;

}


//static float4 q_pad_mc(
//    const int i,
//    const int j,
//    const float w,
//    const float w_sig,
//    const float div_sig,
//    __constant float *T,
//    __constant float *F,
//    __constant float *S,
//    __constant float *B,
//    const float lc_x,
//    const float lc_a,
//    const float lc_c,
//    const float lc_m
//    )
//// Calculate the q vectors for a pixel-array detector
////
//// Input:
//// i, j are the pixel indices
//// w is the photon wavelength
//// T is the translation vector from origin to center of corner pixel
//// F, S are the fast/slow-scan basis vectors (pointing alont rows/columns)
////      the length of these vectors is the pixel size
//// B is the direction of the incident beam
////
//// Output: A single q vector
//{
//    float4 q = (float4)(0.0f,0.0f,0.0f,0.0f);
//
//    // Random number from linear congruential recurrence relation
//    lc_x1 = (lc_a*lc_x + lc_c) % m
//    float del_w;
//
//    float Vx,Vy,Vz,Vnorm;
//
//    Vx = T[0] + i*F[0] + j*S[0];
//    Vy = T[1] + i*F[1] + j*S[1];
//    Vz = T[2] + i*F[2] + j*S[2];
//    Vnorm = sqrt(Vx*Vx + Vy*Vy + Vz*Vz);
//    Vx = Vx/Vnorm;
//    Vy = Vy/Vnorm;
//    Vz = Vz/Vnorm;
//
//
//
//    q.x = (Vx-B[0])*PI2/w;
//    q.y = (Vy-B[1])*PI2/w;
//    q.z = (Vz-B[2])*PI2/w;
//
//    return q;
//
//}


kernel void get_group_size(
    global int *g){
    const int gi = get_global_id(0);
    if (gi == 0){
        g[gi] = GROUP_SIZE;
    }
}


kernel void mod_squared_complex_to_real(
        global const float2 *A,
        global float *I,
        const int n)
{

// Take the mod square of an array of complex numbers.  This is supposed to be
// done with the pyopencl.array.Array class, but I only get seg. faults...

    const int gi = get_global_id(0);
    if (gi < n){
        float2 a = A[gi];
        I[gi] += a.x*a.x + a.y*a.y;
    }
}

kernel void phase_factor_qrf(
    global const float *q,
    global const float *r,
    global const float2 *f,
    __constant float *R,
    global float2 *a,
    const int n_atoms,
    const int n_pixels)
{

// Calculate the the scattering amplitude according to a set of point
// scatterers:  A = sum{ f*exp(i r.q ) }.  This variant accepts as input a set
// of q vectors (computed as you wish), and then the atomic positions and
// scattering factors.  A rotation matrix acts on the q vectors.
//
// Input:
// q: scattering vectors
// r: atomic coordinates
// f: scattering factors
// R: rotation matrix acting on q vectors
// a: scattering amplitudes
// n_atoms: number of atoms
// n_pixels: number of q vectors

    const int gi = get_global_id(0); /* Global index */
    const int li = get_local_id(0);  /* Local group index */

    float ph, sinph, cosph;

    // Each global index corresponds to a particular q-vector.  Note that the
    // global index could be larger than the number of pixels because it must be a
    // multiple of the group size.  We must check if it is larger...
    float2 a_sum = (float2)(0.0f,0.0f);
    float4 qr;
    if (gi < n_pixels){

        a_sum.x = a[gi].x;
        a_sum.y = a[gi].y;
        
        // Move original q vector to private memory
        qr = (float4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);

        // Rotate the q vector
        qr = rotate_vec(R, qr);

    } else {
        // Dummy values; doesn't really matter what they are.
        qr = (float4)(0.0f,0.0f,0.0f,0.0f);
    }
    local float4 rg[GROUP_SIZE];
    local float2 fg[GROUP_SIZE];

    for (int g=0; g<n_atoms; g+=GROUP_SIZE){

        // Here we will move a chunk of atoms to local memory.  Each worker in a
        // group moves one atom.
        int ai = g+li;

        if (ai < n_atoms ){
            rg[li] = (float4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
            fg[li] = f[ai];
        } else {
            rg[li] = (float4)(0.0f,0.0f,0.0f,0.0f);
            fg[li] = (float2)(0.0f,0.0f);
        }

        // Don't proceed until **all** members of the group have finished moving
        // atom information into local memory.
        barrier(CLK_LOCAL_MEM_FENCE);

        // We use a local real and imaginary part to avoid floatint point overflow
        float2 a_temp = (float2)(0.0f,0.0f);

        // Now sum up the amplitudes from this subset of atoms
        for (int n=0; n < GROUP_SIZE; n++){
            ph = -dot(qr,rg[n]);
            sinph = native_sin(ph);
            cosph = native_cos(ph);
            a_temp.x += fg[n].x*cosph - fg[n].y*sinph;
            a_temp.y += fg[n].x*sinph + fg[n].y*cosph;
        }
        a_sum += a_temp;

        // Don't proceed until this subset of atoms are completed.
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (gi < n_pixels){
        a[gi] = a_sum;
    }
}

kernel void phase_factor_pad(
    global const float *r,
    global const float2 *f,
    const float16 R,
    global float2 *a,
    const int n_pixels,
    const int n_atoms,
    const int nF,
    const int nS,
    const float w,
    __constant float *T,
    __constant float *F,
    __constant float *S,
    __constant float *B)
{

// Calculate the the scattering amplitude according to a set of point
// scatterers:  A = sum{ f*exp(i r.q ) }.  This variant generates the q vectors
// for a pixel-array detector on the fly.  The atomic positions and
// scattering factors are specified.  A rotation matrix acts on the q vectors.
//
// Input:
// r: atomic coordinates
// f: complex atomic scattering factors
// R: rotation matrix acting on the q vectors
// a: output amplitudes
// n_pixels: number of pixels
// n_atoms: number of atoms
// nF,nS: number of detector pixels in fast/slow-scan direction
// w: photon wavelength
// T: translation vector from origin to corner detector pixel
// F,S: basis vectors pointing along fast/slow-scan directions (length is equal
//    to the pixel size)
// B: incident beam direction

    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;          /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */
    const int li = get_local_id(0);  /* Local group index */

    float ph, sinph, cosph;
    float re = 0;
    float im = 0;

    // Each global index corresponds to a particular q-vector
    //float4 V;
    float4 q; 
    q = q_pad( i,j,w,T,F,S,B);

    // Rotate the q vector
    q = rotate_vec2(R, q);

    local float4 rg[GROUP_SIZE];
    local float2 fg[GROUP_SIZE];

    for (int g=0; g<n_atoms; g+=GROUP_SIZE){

        // Here we will move a chunk of atoms to local memory.  Each worker in
        // a group moves one atom.
        int ai = g+li;

        if (ai < n_atoms){
            rg[li] = (float4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
            fg[li] = f[ai];
        } else {
            rg[li] = (float4)(0.0f,0.0f,0.0f,0.0f);
            fg[li] = (float2)(0.0f,0.0f);
        }
        // Don't proceed until **all** members of the group have finished
        // moving atom information into local memory.
        barrier(CLK_LOCAL_MEM_FENCE);

        // We use a local real and imaginary part to avoid floatint point
        // overflow
        float lre=0;
        float lim=0;

        // Now sum up the amplitudes from this subset of atoms
        for (int n=0; n < GROUP_SIZE; n++){
            ph = -dot(q,rg[n]);
            sinph = native_sin(ph);
            cosph = native_cos(ph);
            lre += fg[n].x*cosph - fg[n].y*sinph;
            lim += fg[n].x*sinph + fg[n].y*cosph;
        }
        re += lre;
        im += lim;

        // Don't proceed until this subset of atoms are completed.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gi < n_pixels ){
        a[gi].x = re;
        a[gi].y = im;
    }
}


kernel void phase_factor_mesh(
    global const float *r,
    global const float2 *f,
    global float2 *a,
    const int n_pixels,
    const int n_atoms,
    const int4 N,
    const float4 deltaQ,
    const float4 q_min)

// Calculate the the scattering amplitude according to a set of point
// scatterers:  A = sum{ f*exp(i r.q ) }.  This variant generates the q vectors
// corresponding to a 3D lattice.  The atomic positions and
// scattering factors are specified.
//
// Input:
// r: atomic coordinates
// f: scattering factors
// a: scattering amplitudes
// R: rotation matrix acting on q vectors
// n_pixels: number of q vectors
// n_atoms: number of atoms
// N: the number of points on the 3D grid (3 numbers specified)
// deltaQ: the spacing betwen points (3 numbers specified)
// q_min: the start position; edge of grid (3 numbers specified)

{
    const int Nxy = N.x*N.y;
    const int gi = get_global_id(0); /* Global index */
    const int i = gi % N.x;          /* Voxel coordinate i (x) */
    const int j = (gi/N.x) % N.y;    /* Voxel coordinate j (y) */
    const int k = gi/Nxy;            /* Voxel corrdinate k (z) */
    const int li = get_local_id(0);  /* Local group index */

    float ph, sinph, cosph;
    float re = 0;
    float im = 0;
    int ai;
    float4 qr;

    // Each global index corresponds to a particular q-vector
    qr = (float4)(i*deltaQ.x+q_min.x,
            j*deltaQ.y+q_min.y,
            k*deltaQ.z+q_min.z,0.0f);

    local float4 rg[GROUP_SIZE];
    local float2 fg[GROUP_SIZE];

    for (int g=0; g<n_atoms; g+=GROUP_SIZE){

        // Here we will move a chunk of atoms to local memory.  Each worker in a
        // group moves one atom.
        ai = g+li;
        if (ai < n_atoms){
            rg[li] = (float4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
            fg[li] = f[ai];
        } else {
            rg[li] = (float4)(0.0f,0.0f,0.0f,0.0f);
            fg[li] = (float2)(0.0f,0.0f);
        }
        // Don't proceed until **all** members of the group have finished moving
        // atom information into local memory.
        barrier(CLK_LOCAL_MEM_FENCE);

        // We use a local real and imaginary part to avoid floating point overflow
        float lre=0;
        float lim=0;

        // Now sum up the amplitudes from this subset of atoms
        for (int n=0; n < GROUP_SIZE; n++){
            ph = -dot(qr,rg[n]);
            sinph = native_sin(ph);
            cosph = native_cos(ph);
            lre += fg[n].x*cosph - fg[n].y*sinph;
            lim += fg[n].x*sinph + fg[n].y*cosph;
        }
        re += lre;
        im += lim;

        // Don't proceed until this subset of atoms are completed.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gi < n_pixels){
        a[gi].x = re;
        a[gi].y = im;
    }
}


kernel void buffer_mesh_lookup(
    global float2 *a_map,
    global float *q,
    global float2 *a_out,
    int n_pixels,
    int4 N,
    float4 deltaQ,
    float4 q_min,
    __constant float *R)
// This is meant to be used in conjunction with phase_factor_mesh.  This is
// an interpolation routine that uses the output of phase_factor_mesh as the
// "lookup table".  It does trilinear interpolation.
{
    const int gi = get_global_id(0);

    const float4 q4 = (float4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
    float4 q4r;

    q4r = rotate_vec(R,q4);

    // Floating point coordinates
    const float i_f = (q4r.x - q_min.x)/deltaQ.x;
    const float j_f = (q4r.y - q_min.y)/deltaQ.y;
    const float k_f = (q4r.z - q_min.z)/deltaQ.z;

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
    const float x0 = i_f - floor(i_f);
    const float y0 = j_f - floor(j_f);
    const float z0 = k_f - floor(k_f);
    const float x1 = 1.0f - x0;
    const float y1 = 1.0f - y0;
    const float z1 = 1.0f - z0;

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
        a_out[gi] = (float2)(0.0f,0.0f);
    }

}


kernel void lattice_transform_intensities_pad(
    __constant float *abc,
    __constant int *N,
    __constant float *R,
    global float *I,
    const int n_pixels,
    const int nF,
    const int nS,
    const float w,
    __constant float *T,
    __constant float *F,
    __constant float *S,
    __constant float *B,
    const int add)
{

// TODO: add documentation Rick

    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;           /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */

    // Each global index corresponds to a particular q-vector
    float4 q;
    q = q_pad( i,j,w,T,F,S,B);

    // Rotate the q vector
    q = rotate_vec(R, q);

    float sn;
    float s;
    float It = 1.0;
    for (int k=0; k<3; k++){
        float4 v = (float4)(abc[k*3+0],abc[k*3+1],abc[k*3+2],0.0);
        float x = dot(q,v) / 2.0;
        float n = (float)N[k];
        if (x == 0){
            It *= native_powr(n,2);
        } else {
            // This does [ sin(Nx)/sin(x) ]^2
            sn = native_sin(n*x);
            s = native_sin(x);
            It *= native_powr(sn/s,2);
        }
    }

    if (gi < n_pixels ){
        if (add == 1){
            I[gi] += It;
        } else {
            I[gi] = It;
        }
    }
}



__kernel void qrf_default(
    __global float16 *q_vecs,
    __global float4 *r_vecs,
    __constant float *R,
    __global float2 *A,
    const int n_atoms)
// Derek will add documentation to this one.
{

    int q_idx = get_global_id(0);
    int l_idx = get_local_id(0);

    float Areal;
    float Aimag;

    float ff[16];
    
    Areal=A[q_idx].x;
    Aimag=A[q_idx].y;

    float qx = q_vecs[q_idx].s0;
    float qy = q_vecs[q_idx].s1;
    float qz = q_vecs[q_idx].s2;

    float qRx = R[0]*qx + R[3]*qy + R[6]*qz;
    float qRy = R[1]*qx + R[4]*qy + R[7]*qz;
    float qRz = R[2]*qx + R[5]*qy + R[8]*qz;
    
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

    
    __local float4 LOC_ATOMS[GROUP_SIZE];
    for (int g=0; g<n_atoms; g+=GROUP_SIZE){
        int ai = g + l_idx;
        if (ai < n_atoms)
            LOC_ATOMS[l_idx] = r_vecs[ai];
        if( !(ai < n_atoms))
            LOC_ATOMS[l_idx] = (float4)(1.0f, 1.0f, 1.0f, 15.0f); // make atom ID 15, s.t. ff=0

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int i=0; i< GROUP_SIZE; i++){

            float phase = qRx*LOC_ATOMS[i].x + qRy*LOC_ATOMS[i].y + qRz*LOC_ATOMS[i].z;
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
    __global float16 *q_vecs,
    __global float4 *r_vecs,
    __constant float *R,
    __constant float *T,
    __global float2 *A,
    const int n_atoms)
// Derek will add documentation to this one.
{

    int q_idx = get_global_id(0);
    int l_idx = get_local_id(0);

    //float Areal=0.0f;
    //float Aimag=0.0f;
    float Areal;
    float Aimag;

    float ff[16];
    

    // multiply trans vector by inverse rotation matrix  
    float Tx = R[0]*T[0] + R[3]*T[1] + R[6]*T[2];
    float Ty = R[1]*T[0] + R[4]*T[1] + R[7]*T[2];
    float Tz = R[2]*T[0] + R[5]*T[1] + R[8]*T[2];

    Areal=A[q_idx].x;
    Aimag=A[q_idx].y;

    float qx = q_vecs[q_idx].s0;
    float qy = q_vecs[q_idx].s1;
    float qz = q_vecs[q_idx].s2;

    float qRx = R[0]*qx + R[3]*qy + R[6]*qz;
    float qRy = R[1]*qx + R[4]*qy + R[7]*qz;
    float qRz = R[2]*qx + R[5]*qy + R[8]*qz;
    
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

    __local float4 LOC_ATOMS[GROUP_SIZE];
    for (int g=0; g<n_atoms; g+=GROUP_SIZE){
        int ai = g + l_idx;
        if (ai < n_atoms)
            LOC_ATOMS[l_idx] = r_vecs[ai];
        if( !(ai < n_atoms))
            LOC_ATOMS[l_idx] = (float4)(1.0f, 1.0f, 1.0f, 15.0f); // make atom ID 15, s.t. ff=0

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int i=0; i< GROUP_SIZE; i++){

            float phase = qRx*(LOC_ATOMS[i].x+Tx) + qRy*(LOC_ATOMS[i].y+Ty) +
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




//__kernel void ideal_lattice_transform_intensity_qrf(
//    __global float4 *q,
//    __constant float *R,
//    __constant float *U,
//    __constant float *N,
//    __global float *I,
//    const int n_atoms)
//// Calculate the lattice transform intensity for a parallelepiped crystal.
////
//// Input:
//// q:
//// U is the orthogonalization matrix defined in Rupp's textbook "Biomolecular
//// Crystallography" as "O" (renamed to avoid confusion with zero).  The columns
//// of this matrix are the a, b, and c lattice vectors.
//{
//
//}

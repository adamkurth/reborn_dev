#ifndef GROUP_SIZE
    #define GROUP_SIZE 1
#endif

#if CONFIG_USE_DOUBLE
    #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #define DOUBLE_SUPPORT_AVAILABLE
    #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #define DOUBLE_SUPPORT_AVAILABLE
    #endif
#endif // CONFIG_USE_DOUBLE

#if defined(DOUBLE_SUPPORT_AVAILABLE)
    // double
    typedef double gfloat;
    typedef double2 gfloat2;
    typedef double4 gfloat4;
    typedef double16 gfloat16;
    #define PI 3.14159265358979323846
    #define PI2 6.28318530717958647693
#else
    // float
    typedef float gfloat;
    typedef float2 gfloat2;
    typedef float4 gfloat4;
    typedef float16 gfloat16;
    #define PI 3.14159265359f
    #define PI2 6.28318530718f
#endif

kernel void get_group_size(
    global int *g){
    const int gi = get_global_id(0);
    if (gi == 0){
        g[gi] = GROUP_SIZE;
    }
}

kernel void phase_factor_qrf2(
    __global gfloat4 *q,
    __global gfloat4 *r,
    __global gfloat2 *f,
    const gfloat16 R,
    __global gfloat2 *a,
    const int n_atoms)

{
    const int gi = get_global_id(0); /* Global index */
    const int li = get_local_id(0);  /* Local group index */

    gfloat ph, sinph, cosph;

    // Each global index corresponds to a particular q-vector.  Note that the
    // global index could be larger than the number of pixels because it must be a
    // multiple of the group size.  We must check if it is larger...
    gfloat2 a_sum = (gfloat2)(0.0f,0.0f);
    gfloat4 q4, q4r;

    a_sum.x = a[gi].x;
    a_sum.y = a[gi].y;
    
    // Move original q vector to private memory
    q4 = q[gi];

    // Rotate the q vector
    q4r.x = R.s0*q4.x + R.s1*q4.y + R.s2*q4.z;
    q4r.y = R.s3*q4.x + R.s4*q4.y + R.s5*q4.z;
    q4r.z = R.s6*q4.x + R.s7*q4.y + R.s8*q4.z;

    local gfloat4 rg[GROUP_SIZE];
    local gfloat2 fg[GROUP_SIZE];

    for (int g=0; g<n_atoms; g+=GROUP_SIZE){

        // Here we will move a chunk of atoms to local memory.  Each worker in a
        // group moves one atom.
        int ai = g+li;

        rg[li] = r[ai];
        fg[li] = f[ai];

        // Don't proceed until **all** members of the group have finished moving
        // atom information into local memory.
        barrier(CLK_LOCAL_MEM_FENCE);

        // We use a local real and imaginary part to avoid floatint point overflow
        gfloat2 a_temp = (gfloat2)(0.0f,0.0f);

        // Now sum up the amplitudes from this subset of atoms
        for (int n=0; n < GROUP_SIZE; n++){
            ph = -dot(q4r,rg[n]);
            sinph = native_sin(ph);
            cosph = native_cos(ph);
            a_temp.x += fg[n].x*cosph - fg[n].y*sinph;
            a_temp.y += fg[n].x*sinph + fg[n].y*cosph;
        }
        a_sum += a_temp;

        // Don't proceed until this subset of atoms are completed.
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    a[gi] = a_sum;
}


kernel void phase_factor_qrf(
    global const gfloat *q,
    global const gfloat *r,
    global const gfloat2 *f,
    const gfloat16 R,
    global gfloat2 *a,
    const int n_atoms,
    const int n_pixels)
{
    const int gi = get_global_id(0); /* Global index */
    const int li = get_local_id(0);  /* Local group index */

    gfloat ph, sinph, cosph;

    // Each global index corresponds to a particular q-vector.  Note that the
    // global index could be larger than the number of pixels because it must be a
    // multiple of the group size.  We must check if it is larger...
    gfloat2 a_sum = (gfloat2)(0.0f,0.0f);
    gfloat4 q4, q4r;
    if (gi < n_pixels){

        a_sum.x = a[gi].x;
        a_sum.y = a[gi].y;
        
        // Move original q vector to private memory
        q4 = (gfloat4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);

        // Rotate the q vector
        q4r.x = R.s0*q4.x + R.s1*q4.y + R.s2*q4.z;
        q4r.y = R.s3*q4.x + R.s4*q4.y + R.s5*q4.z;
        q4r.z = R.s6*q4.x + R.s7*q4.y + R.s8*q4.z;

    } else {
        // Dummy values; doesn't really matter what they are.
        q4r = (gfloat4)(0.0f,0.0f,0.0f,0.0f);
    }
    local gfloat4 rg[GROUP_SIZE];
    local gfloat2 fg[GROUP_SIZE];

    for (int g=0; g<n_atoms; g+=GROUP_SIZE){

        // Here we will move a chunk of atoms to local memory.  Each worker in a
        // group moves one atom.
        int ai = g+li;

        if (ai < n_atoms ){
            rg[li] = (gfloat4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
            fg[li] = f[ai];
        } else {
            rg[li] = (gfloat4)(0.0f,0.0f,0.0f,0.0f);
            fg[li] = (gfloat2)(0.0f,0.0f);
        }

        // Don't proceed until **all** members of the group have finished moving
        // atom information into local memory.
        barrier(CLK_LOCAL_MEM_FENCE);

        // We use a local real and imaginary part to avoid floatint point overflow
        gfloat2 a_temp = (gfloat2)(0.0f,0.0f);

        // Now sum up the amplitudes from this subset of atoms
        for (int n=0; n < GROUP_SIZE; n++){
            ph = -dot(q4r,rg[n]);
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
    global const gfloat *r,
    global const gfloat2 *f,
    const gfloat16 R,
    global gfloat2 *a,
    const int n_pixels,
    const int n_atoms,
    const int nF,
    const int nS,
    const gfloat w,
    const gfloat4 T,
    const gfloat4 F,
    const gfloat4 S,
    const gfloat4 B)
{
    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;          /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */
    const int li = get_local_id(0);  /* Local group index */


    gfloat ph, sinph, cosph;
    gfloat re = 0;
    gfloat im = 0;

    // Each global index corresponds to a particular q-vector
    gfloat4 V;
    gfloat4 q;

    V = T + i*F + j*S;
    V /= length(V);
    q = (V-B)*PI2/w;

    local gfloat4 rg[GROUP_SIZE];
    local gfloat2 fg[GROUP_SIZE];

    for (int g=0; g<n_atoms; g+=GROUP_SIZE){

        // Here we will move a chunk of atoms to local memory.  Each worker in a
        // group moves one atom.
        int ai = g+li;

        if (ai < n_atoms){
            rg[li] = (gfloat4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
            fg[li] = f[ai];
        } else {
            rg[li] = (gfloat4)(0.0f,0.0f,0.0f,0.0f);
            fg[li] = (gfloat2)(0.0f,0.0f);
        }
        // Don't proceed until **all** members of the group have finished moving
        // atom information into local memory.
        barrier(CLK_LOCAL_MEM_FENCE);

        // We use a local real and imaginary part to avoid floatint point overflow
        gfloat lre=0;
        gfloat lim=0;

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
    global const gfloat *r,
    global const gfloat2 *f,
    global gfloat2 *a,
    const int n_pixels,
    const int n_atoms,
    const int4 N,
    const gfloat4 deltaQ,
    const gfloat4 q_min)
{

    const int Nxy = N.x*N.y;
    const int gi = get_global_id(0); /* Global index */
    const int i = gi % N.x;          /* Voxel coordinate i (x) */
    const int j = (gi/N.x) % N.y;    /* Voxel coordinate j (y) */
    const int k = gi/Nxy;            /* Voxel corrdinate k (z) */
    const int li = get_local_id(0);  /* Local group index */

    gfloat ph, sinph, cosph;
    gfloat re = 0;
    gfloat im = 0;
    int ai;

    // Each global index corresponds to a particular q-vector
    const gfloat4 q4 = (gfloat4)(i*deltaQ.x+q_min.x,
            j*deltaQ.y+q_min.y,
            k*deltaQ.z+q_min.z,0.0f);

    local gfloat4 rg[GROUP_SIZE];
    local gfloat2 fg[GROUP_SIZE];

    for (int g=0; g<n_atoms; g+=GROUP_SIZE){

        // Here we will move a chunk of atoms to local memory.  Each worker in a
        // group moves one atom.
        ai = g+li;
        if (ai < n_atoms){
            rg[li] = (gfloat4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
            fg[li] = f[ai];
        } else {
            rg[li] = (gfloat4)(0.0f,0.0f,0.0f,0.0f);
            fg[li] = (gfloat2)(0.0f,0.0f);
        }
        // Don't proceed until **all** members of the group have finished moving
        // atom information into local memory.
        barrier(CLK_LOCAL_MEM_FENCE);

        // We use a local real and imaginary part to avoid floating point overflow
        gfloat lre=0;
        gfloat lim=0;

        // Now sum up the amplitudes from this subset of atoms
        for (int n=0; n < GROUP_SIZE; n++){
            ph = -dot(q4,rg[n]);
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
    global gfloat2 *a_map,
    global gfloat *q,
    global gfloat2 *a_out,
    int n_pixels,
    int4 N,
    gfloat4 deltaQ,
    gfloat4 q_min,
    gfloat16 R)
{
    const int gi = get_global_id(0);

    const gfloat4 q4 = (gfloat4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
    const gfloat4 q4r = (gfloat4)(0.0f,0.0f,0.0f,0.0f);

    q4r.x = R.s0*q4.x + R.s1*q4.y + R.s2*q4.z;
    q4r.y = R.s3*q4.x + R.s4*q4.y + R.s5*q4.z;
    q4r.z = R.s6*q4.x + R.s7*q4.y + R.s8*q4.z;

    // Floating point coordinates
    const gfloat i_f = (q4r.x - q_min.x)/deltaQ.x;
    const gfloat j_f = (q4r.y - q_min.y)/deltaQ.y;
    const gfloat k_f = (q4r.z - q_min.z)/deltaQ.z;

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
    const gfloat x0 = i_f - floor(i_f);
    const gfloat y0 = j_f - floor(j_f);
    const gfloat z0 = k_f - floor(k_f);
    const gfloat x1 = 1.0f - x0;
    const gfloat y1 = 1.0f - y0;
    const gfloat z1 = 1.0f - z0;

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
        a_out[gi] = (gfloat2)(0.0f,0.0f);
    }

}


__kernel void qrf_default(
    __global gfloat16 *q_vecs,
    __global gfloat4 *r_vecs,
    __constant gfloat *R,
    __global gfloat2 *A,
    const int n_atoms){

    int q_idx = get_global_id(0);
    int l_idx = get_local_id(0);

    gfloat Areal;
    gfloat Aimag;

    gfloat ff[16];
    
    Areal=A[q_idx].x;
    Aimag=A[q_idx].y;

    gfloat qx = q_vecs[q_idx].s0;
    gfloat qy = q_vecs[q_idx].s1;
    gfloat qz = q_vecs[q_idx].s2;

    gfloat qRx = R[0]*qx + R[3]*qy + R[6]*qz;
    gfloat qRy = R[1]*qx + R[4]*qy + R[7]*qz;
    gfloat qRz = R[2]*qx + R[5]*qy + R[8]*qz;
    
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

    
    __local gfloat4 LOC_ATOMS[GROUP_SIZE];
    for (int g=0; g<n_atoms; g+=GROUP_SIZE){
        int ai = g + l_idx;
        if (ai < n_atoms)
            LOC_ATOMS[l_idx] = r_vecs[ai];
        if( !(ai < n_atoms))
            LOC_ATOMS[l_idx] = (gfloat4)(1.0f, 1.0f, 1.0f, 15.0f); // make atom ID 15, s.t. ff=0

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int i=0; i< GROUP_SIZE; i++){

            gfloat phase = qRx*LOC_ATOMS[i].x + qRy*LOC_ATOMS[i].y + qRz*LOC_ATOMS[i].z;
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
    __global gfloat16 *q_vecs,
    __global gfloat4 *r_vecs,
    __constant gfloat *R,
    __constant gfloat *T,
    __global gfloat2 *A,
    const int n_atoms){

    int q_idx = get_global_id(0);
    int l_idx = get_local_id(0);

    //gfloat Areal=0.0f;
    //gfloat Aimag=0.0f;
    gfloat Areal;
    gfloat Aimag;

    gfloat ff[16];
    

    // multiply trans vector by inverse rotation matrix  
    gfloat Tx = R[0]*T[0] + R[3]*T[1] + R[6]*T[2];
    gfloat Ty = R[1]*T[0] + R[4]*T[1] + R[7]*T[2];
    gfloat Tz = R[2]*T[0] + R[5]*T[1] + R[8]*T[2];

    Areal=A[q_idx].x;
    Aimag=A[q_idx].y;

    gfloat qx = q_vecs[q_idx].s0;
    gfloat qy = q_vecs[q_idx].s1;
    gfloat qz = q_vecs[q_idx].s2;

    gfloat qRx = R[0]*qx + R[3]*qy + R[6]*qz;
    gfloat qRy = R[1]*qx + R[4]*qy + R[7]*qz;
    gfloat qRz = R[2]*qx + R[5]*qy + R[8]*qz;
    
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

    __local gfloat4 LOC_ATOMS[GROUP_SIZE];
    for (int g=0; g<n_atoms; g+=GROUP_SIZE){
        int ai = g + l_idx;
        if (ai < n_atoms)
            LOC_ATOMS[l_idx] = r_vecs[ai];
        if( !(ai < n_atoms))
            LOC_ATOMS[l_idx] = (gfloat4)(1.0f, 1.0f, 1.0f, 15.0f); // make atom ID 15, s.t. ff=0

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int i=0; i< GROUP_SIZE; i++){

            gfloat phase = qRx*(LOC_ATOMS[i].x+Tx) + qRy*(LOC_ATOMS[i].y+Ty) +
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


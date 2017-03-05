#define GROUP_SIZE 32
#define PI2 6.28318530718f

kernel void phase_factor_qrf(
    global const float *q,
    global const float *r,
    global const float2 *f,
    global float2 *a,
    int n_atoms,
    int n_pixels)
{
    const int gi = get_global_id(0); /* Global index */
    const int li = get_local_id(0);  /* Local group index */

    float ph, sinph, cosph;
    float re = 0;
    float im = 0;

    // Each global index corresponds to a particular q-vector.  Note that the
    // global index could be larger than the number of pixels because it must be a
    // multiple of the group size.
    float4 q4;
    if (gi < n_pixels){
        q4 = (float4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
    } else {
        q4 = (float4)(0.0f,0.0f,0.0f,0.0f);
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
        float lre=0;
        float lim=0;

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


kernel void phase_factor_pad(
    global const float *r,
    global const float2 *f,
    global float2 *a,
    int n_pixels,
    int n_atoms,
    int nF,
    int nS,
    float w,
    float4 T,
    float4 F,
    float4 S,
    float4 B)
{
    const int gi = get_global_id(0); /* Global index */
    const int i = gi % nF;          /* Pixel coordinate i */
    const int j = gi/nF;             /* Pixel coordinate j */
    const int li = get_local_id(0);  /* Local group index */


    float ph, sinph, cosph;
    float re = 0;
    float im = 0;

    // Each global index corresponds to a particular q-vector
    float4 V;
    float4 q;

    V = T + i*F + j*S;
    V /= length(V);
    q = (V-B)*PI2/w;

    local float4 rg[GROUP_SIZE];
    local float2 fg[GROUP_SIZE];

    for (int g=0; g<n_atoms; g+=GROUP_SIZE){

        // Here we will move a chunk of atoms to local memory.  Each worker in a
        // group moves one atom.
        int ai = g+li;

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

        // We use a local real and imaginary part to avoid floatint point overflow
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
    int n_pixels,
    int n_atoms,
    int4 N,
    float4 deltaQ,
    float4 q_min)
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

    // Each global index corresponds to a particular q-vector
    const float4 q4 = (float4)(i*deltaQ.x+q_min.x,
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
    global float2 *a_map,
    global float *q,
    global float2 *a_out,
    int n_pixels,
    int4 N,
    float4 deltaQ,
    float4 q_min)
{
    const int gi = get_global_id(0); /* Global index is q-vector index */
    const float4 q4 = (float4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);

    // Floating point coordinates
    const float i_f = (q4.x - q_min.x)/deltaQ.x;
    const float j_f = (q4.y - q_min.y)/deltaQ.y;
    const float k_f = (q4.z - q_min.z)/deltaQ.z;

    // Nearest integer coordinates
    const int i = (int)(floor(i_f));
    const int j = (int)(floor(j_f));
    const int k = (int)(floor(k_f));
    const int kk0 = k*N.x*N.y;
    const int jj0 = j*N.x;
    const int ii0 = i;
    const int kk1 = (k+1)*N.x*N.y;
    const int jj1 = (j+1)*N.x;
    const int ii1 = i+1;

    // Coordinates specified in paulbourke.net/miscellaneous/interpolation
    const float x = i_f - floor(i_f);
    const float y = j_f - floor(j_f);
    const float z = k_f - floor(k_f);
    const float x1 = 1.0f - x;
    const float y1 = 1.0f - y;
    const float z1 = 1.0f - z;

    if (i >= 0 && i < N.x && j >= 0 && j < N.y && k >= 0 && k < N.z){

        a_out[gi] = a_map[ii0 + jj0 + kk0] * x1 * y1 * z1 +

                a_map[ii1 + jj0 + kk0] * x  * y1 * z1 +
                a_map[ii0 + jj1 + kk0] * x1 * y  * z1 +
                a_map[ii0 + jj0 + kk1] * x1 * y1 * z  +

                a_map[ii0 + jj1 + kk1] * x1 * y  * z  +
                a_map[ii1 + jj0 + kk1] * x  * y1 * z  +
                a_map[ii1 + jj1 + kk0] * x  * y  * z1 +

                a_map[ii1 + jj1 + kk1] * x  * y  * z    ;

        // Nearest neighbor
        //const int idx = k*N.x*N.y + j*N.x + i;
        //a_out[gi].x = a_map[idx].x;
        //a_out[gi].y = a_map[idx].y;

    } else {
        a_out[gi].x = 0.0f;
        a_out[gi].y = 0.0f;
    }

}

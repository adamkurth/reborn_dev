#define GROUP_SIZE 32
#define PI2 6.28318530718f

kernel void phase_factor_qrf(
    global const float *q,
    global const float *r,
    global const float2 *f,
    const float16 R,
    global float2 *a,
    const int n_atoms,
    const int n_pixels)
{
    const int gi = get_global_id(0); /* Global index */
    const int li = get_local_id(0);  /* Local group index */

    float ph, sinph, cosph;
    float2 a_sum = (float2)(0.0f,0.0f);

    // Each global index corresponds to a particular q-vector.  Note that the
    // global index could be larger than the number of pixels because it must be a
    // multiple of the group size.  We must check if it is larger...
    float4 q4, q4r;
    if (gi < n_pixels){

        // Move original q vector to private memory
        q4 = (float4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);

        // Rotate the q vector
        q4r.x = R.s0*q4.x + R.s1*q4.y + R.s2*q4.z;
        q4r.y = R.s3*q4.x + R.s4*q4.y + R.s5*q4.z;
        q4r.z = R.s6*q4.x + R.s7*q4.y + R.s8*q4.z;

    } else {
        // Dummy values; doesn't really matter what they are.
        q4r = (float4)(0.0f,0.0f,0.0f,0.0f);
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
    global const float *r,
    global const float2 *f,
    const float16 R,
    global float2 *a,
    const int n_pixels,
    const int n_atoms,
    const int nF,
    const int nS,
    const float w,
    const float4 T,
    const float4 F,
    const float4 S,
    const float4 B)
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
    const int n_pixels,
    const int n_atoms,
    const int4 N,
    const float4 deltaQ,
    const float4 q_min)
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
    float4 q_min,
    float16 R)
{
    const int gi = get_global_id(0);

    const float4 q4 = (float4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
    const float4 q4r = (float4)(0.0f,0.0f,0.0f,0.0f);

    q4r.x = R.s0*q4.x + R.s1*q4.y + R.s2*q4.z;
    q4r.y = R.s3*q4.x + R.s4*q4.y + R.s5*q4.z;
    q4r.z = R.s6*q4.x + R.s7*q4.y + R.s8*q4.z;

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


__kernel void qrf_default2(
    __global float *q_vecs,
    __global float *r_vecs,
    __global float *form_facts,
    __global int *atomID,
    __global float *R,
    __global float2 *A,
    const int n_pixels,
    const int n_atoms){
    
    int q_idx = get_global_id(0);

    A[q_idx].x =0;
    A[q_idx].y =0;

    if ( q_idx < n_pixels) {

        float qx = q_vecs[q_idx*3];
        float qy = q_vecs[ q_idx*3+1];
        float qz = q_vecs[ q_idx*3+2];

        float qRx = R[0]*qx + R[1]*qy + R[2]*qz;
        float qRy = R[3]*qx + R[4]*qy + R[5]*qz;
        float qRz = R[6]*qx + R[7]*qy + R[8]*qz;
        
        for (int r_idx=0; r_idx< n_atoms; r_idx++){
        
            float rx = r_vecs[ r_idx*3];
            float ry = r_vecs[ r_idx*3+1];
            float rz = r_vecs[ r_idx*3+2];
            
            float frm_fct = form_facts[ atomID[r_idx]*n_pixels + q_idx  ];

            float phase = qRx*rx + qRy*ry + qRz*rz;
            
            float cosph = native_cos(-phase);
            float sinph = native_sin(-phase);
            
            A[q_idx].x += frm_fct*cosph;
            A[q_idx].y += frm_fct*sinph;
            }
        }
   
    // each processing unit should do the same thing? 
    if ( !(q_idx < n_pixels)) {
        
        float qx = q_vecs[0];//dummies
        float qy = q_vecs[1];
        float qz = q_vecs[2];
        
        float qRx = R[0]*qx + R[1]*qy + R[2]*qz;
        float qRy = R[3]*qx + R[4]*qy + R[5]*qz;
        float qRz = R[6]*qx + R[7]*qy + R[8]*qz;
        
        for (int r_idx=0; r_idx< n_atoms; r_idx++){
        
            float rx = r_vecs[0];//dummies
            float ry = r_vecs[1];
            float rz = r_vecs[2];
            
            float frm_fct = form_facts[  0  ]; //dummie
            
            float phase = qRx*rx + qRy*ry + qRz*rz;
            
            float sinph = native_sin(-phase);
            float cosph = native_cos(-phase);
            
            A[q_idx].x += 0;
            A[q_idx].y += 0;
            }
        }
    }

__kernel void qrf_default(
    __global float *q_vecs,
    __global float *r_vecs,
    __global float *form_facts,
    __global int *atomID,
    __global float *R,
    __global float2 *A,
    const int n_pixels,
    const int n_atoms){
    
    int q_idx = get_global_id(0);

    A[q_idx].x =0;
    A[q_idx].y =0;

    if ( q_idx < n_pixels) {

        float qx = q_vecs[q_idx*3];
        float qy = q_vecs[ q_idx*3+1];
        float qz = q_vecs[ q_idx*3+2];

        float qRx = R[0]*qx + R[3]*qy + R[6]*qz;
        float qRy = R[1]*qx + R[4]*qy + R[7]*qz;
        float qRz = R[2]*qx + R[5]*qy + R[8]*qz;
        
        for (int r_idx=0; r_idx< n_atoms; r_idx++){
        
            float rx = r_vecs[ r_idx*3];
            float ry = r_vecs[ r_idx*3+1];
            float rz = r_vecs[ r_idx*3+2];
            
            float frm_fct = form_facts[ atomID[r_idx]*n_pixels + q_idx  ];

            float phase = qRx*rx + qRy*ry + qRz*rz;
            
            float cosph = native_cos(-phase);
            float sinph = native_sin(-phase);
            
            A[q_idx].x += frm_fct*cosph;
            A[q_idx].y += frm_fct*sinph;
            }
        }
   
    // each processing unit should do the same thing? 
    if ( !(q_idx < n_pixels)) {
        
        float qx = q_vecs[0];//dummies
        float qy = q_vecs[1];
        float qz = q_vecs[2];
        
        float qRx = R[0]*qx + R[1]*qy + R[2]*qz;
        float qRy = R[3]*qx + R[4]*qy + R[5]*qz;
        float qRz = R[6]*qx + R[7]*qy + R[8]*qz;
        
        for (int r_idx=0; r_idx< n_atoms; r_idx++){
        
            float rx = r_vecs[0];//dummies
            float ry = r_vecs[1];
            float rz = r_vecs[2];
            
            float frm_fct = form_facts[  0  ]; //dummie
            
            float phase = qRx*rx + qRy*ry + qRz*rz;
            
            float sinph = native_sin(-phase);
            float cosph = native_cos(-phase);
            
            A[q_idx].x += 0;
            A[q_idx].y += 0;
            }
        }
    }


extern "C"{
__device__ __forceinline__ double PI(){return 3.141592653589793238462643383279502884;}

// deinterleave (Morton decode) for 64-bit
__device__ __forceinline__ unsigned long long compact1by1_64(unsigned long long x){
    x &= 0x5555555555555555ull;
    x = (x ^ (x >> 1)) & 0x3333333333333333ull;
    x = (x ^ (x >> 2)) & 0x0F0F0F0F0F0F0F0Full;
    x = (x ^ (x >> 4)) & 0x00FF00FF00FF00FFull;
    x = (x ^ (x >> 8)) & 0x0000FFFF0000FFFFull;
    x = (x ^ (x >> 16))& 0x00000000FFFFFFFFull;
    return x;
}

__global__ void pix2ang_nest_batch(
    const unsigned long long* __restrict__ ipnest, // [N]
    int N,
    int nside,
    double* __restrict__ out_theta,
    double* __restrict__ out_phi
){
    // jrll and jpll lookup as in HEALPix: face parameters
    __shared__ int jrll[12];
    __shared__ int jpll[12];
    if(threadIdx.x < 12){
        const int jrll_h[12] = {2,2,2,2,3,3,3,3,4,4,4,4};
        const int jpll_h[12] = {1,3,5,7,0,2,4,6,1,3,5,7};
        jrll[threadIdx.x] = jrll_h[threadIdx.x];
        jpll[threadIdx.x] = jpll_h[threadIdx.x];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;

    unsigned long long p = ipnest[i];
    long long pix_per_face = (long long)nside * (long long)nside;
    int face = (int)(p / pix_per_face);                // 0..11
    unsigned long long ipf = (unsigned long long)(p % pix_per_face); // index within face

    // decode Morton index within face to (ix,iy)
    unsigned long long ix = compact1by1_64(ipf);
    unsigned long long iy = compact1by1_64(ipf >> 1);

    // jrt (vertical), jpt (horizontal) in face coordinates
    long long jrt = (long long)ix + (long long)iy;             // 0 .. 2*(nside-1)
    long long jpt = (long long)ix - (long long)iy;             // -(nside-1) .. +(nside-1)

    // ring index jr (1..4*nside -1) per face
    long long jr = (long long)jrll[face] * (long long)nside - jrt - 1;

    double z;
    long long nr = (long long)nside;
    int kshift = (int)(((jr - (long long)nside) & 1LL) != 0); // (jr - nside) mod 2

    if(jr < (long long)nside){
        // North cap
        nr = jr;
        double fact1 = 1.0 / (3.0 * (double)nside * (double)nside);
        z = 1.0 - (double)(nr*nr) * fact1;
        kshift = 0;
    }else if(jr > 3LL*(long long)nside){
        // South cap
        nr = 4LL*(long long)nside - jr;
        double fact1 = 1.0 / (3.0 * (double)nside * (double)nside);
        z = -1.0 + (double)(nr*nr) * fact1;
        kshift = 0;
    }else{
        // Equatorial
        double fact2 = 2.0 / (3.0 * (double)nside);
        z = (double)(2*(long long)nside - jr) * fact2;
    }

    long long jp = ( (long long)jpll[face] * nr + jpt + 1 + kshift ) / 2;
    long long ns4 = 4LL*(long long)nside;
    if(jp > ns4) jp -= ns4;
    if(jp < 1)   jp += ns4;

    double theta = acos( fmax(-1.0, fmin(1.0, z)) );
    double phi = ( (double)jp - 0.5*(double)(kshift + 1) ) * (PI() / (2.0 * (double)nr));

    out_theta[i] = theta;
    out_phi[i] = phi;
}
} // extern "C"

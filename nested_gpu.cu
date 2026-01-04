
extern "C"{
__device__ __forceinline__ double PI(){return 3.141592653589793238462643383279502884;}
__device__ __forceinline__ double TWO_PI(){return 6.28318530717958647692528676655900577;}

// Helper: wrap to [0,2pi)
__device__ __forceinline__ double wrap_2pi(double a){
    double t = fmod(a, TWO_PI());
    if(t < 0.0) t += TWO_PI();
    return t;
}
// Morton helpers (64-bit)
__device__ __forceinline__ unsigned long long part1by1_u64(unsigned long long x){
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFull;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x << 2))  & 0x3333333333333333ull;
    x = (x | (x << 1))  & 0x5555555555555555ull;
    return x;
}
__device__ __forceinline__ unsigned long long morton2d_64(unsigned long long ix, unsigned long long iy){
    return (part1by1_u64(iy) << 1) | part1by1_u64(ix);
}

// Convert (theta,phi) to NESTED ipix (HEALPix exact algorithm).
// Follows Gorski et al. zone logic (equatorial vs polar), using z=cos(theta) and phi phase.
__global__ void vec2pix_nest_batch(
    const double* __restrict__ theta,
    const double* __restrict__ phi,
    int N,
    int nside,
    unsigned long long* __restrict__ out_ipix
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    double th = theta[i];
    double ph = wrap_2pi(phi[i]);
    double z = cos(th);
    double za = fabs(z);
    double tt = ph / (0.5*PI()); // 0..4

    int face;
    double jr, jp, jm;
    if(za <= 2.0/3.0){
        // Equatorial region
        double temp1 = nside * (0.5 + tt);
        double temp2 = nside * (z * 0.75);
        jp = floor(temp1 - temp2); // in [0,4*nside-1]
        jm = floor(temp1 + temp2);
        int ifp = (int)jp / nside;  // face number modulo 4
        int ifm = (int)jm / nside;
        if(ifp == ifm){
            face = (ifp == 4) ? 3 : ifp; // wrap
        }else if(ifp < ifm){
            face = (ifp == 0) ? 4 : ifp - 1;
        }else{
            face = (ifm == 0) ? 7 : ifm + 3;
        }
        int ix = (int)jm % nside;
        int iy = (int)jp % nside;
        // local (ix,iy) to pixel-in-face Morton code
        unsigned long long m = morton2d_64((unsigned long long)ix, (unsigned long long)iy);
        out_ipix[i] = (unsigned long long)face * (unsigned long long)(nside*nside) + m;
    }else{
        // Polar caps
        int ntt = (int)floor(tt);
        if(ntt >= 4) ntt = 3;
        double tp = tt - ntt;
        double tmp = nside * (2.0 - za*1.5);
        int jp_i = (int)floor( tp * tmp );
        int jm_i = (int)floor( (1.0 - tp) * tmp );
        int ir = (int)floor( nside - (jp_i + jm_i + 1)/2 ); // ring within the cap
        int ip;
        if(ir < 0) ir = 0;
        if(z >= 0.0){
            face = ntt;
            ip = (int)(jp_i - jm_i + nside + ((ir & 1) ? 1 : 0)) / 2;
            if(ip >= nside) ip -= nside;
            if(ip < 0) ip += nside;
            int ix = nside - 1 - ir;
            int iy = ip;
            unsigned long long m = morton2d_64((unsigned long long)ix, (unsigned long long)iy);
            out_ipix[i] = (unsigned long long)face * (unsigned long long)(nside*nside) + m;
        }else{
            face = ntt + 8;
            ip = (int)(jp_i - jm_i + nside + ((ir & 1) ? 1 : 0)) / 2;
            if(ip >= nside) ip -= nside;
            if(ip < 0) ip += nside;
            int ix = ir;
            int iy = ip;
            unsigned long long m = morton2d_64((unsigned long long)ix, (unsigned long long)iy);
            out_ipix[i] = (unsigned long long)face * (unsigned long long)(nside*nside) + m;
        }
    }
}
} // extern "C"


extern "C"{
__device__ __forceinline__ double TWO_PI(){return 6.28318530717958647692528676655900577;}
__device__ __forceinline__ double wrap_2pi(double a){
    double t = fmod(a, TWO_PI());
    if(t < 0.0) t += TWO_PI();
    return t;
}
__global__ void ang2pix_ring_with_rings(
    const double* __restrict__ theta,
    const double* __restrict__ phi,
    const int* __restrict__ ring_idx,
    int N,
    const long long* __restrict__ ring_start,
    const int* __restrict__ nphi,
    const double* __restrict__ phi_center0,
    long long* __restrict__ out_ipix
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    int r = ring_idx[i];
    int npr = nphi[r];
    double dphi = TWO_PI()/ (double)npr;
    double pc0 = phi_center0[r];
    double d = wrap_2pi(phi[i] - pc0);
    long long j = (long long)llround(d / dphi);
    if(j >= npr) j -= npr;
    out_ipix[i] = ring_start[r] + j;
}
__global__ void pix2ang_ring_batch(
    const long long* __restrict__ ipix,
    int N,
    const double* __restrict__ theta_centers,
    const long long* __restrict__ ring_start,
    const int* __restrict__ nphi,
    const double* __restrict__ phi_center0,
    double* __restrict__ out_theta,
    double* __restrict__ out_phi
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    long long p = ipix[i];
    // binary search ring
    // naive linear scan fallback for simplicity: advance until start > p
    int r=0;
    while(ring_start[r+1] <= p) r++;
    int npr = nphi[r];
    long long j = p - ring_start[r];
    double dphi = TWO_PI()/ (double)npr;
    double phi = phi_center0[r] + (double)j * dphi;
    if(phi >= TWO_PI()) phi -= TWO_PI()*floor(phi/TWO_PI());
    out_theta[i] = theta_centers[r];
    out_phi[i] = phi;
}
} // extern "C"

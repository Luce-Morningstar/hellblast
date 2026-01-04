
extern "C"{
__device__ __forceinline__ double PI(){return 3.141592653589793238462643383279502884;}

__global__ void build_ring_tables(long long nside,
                                  double* __restrict__ z_centers,
                                  double* __restrict__ theta_centers,
                                  int*    __restrict__ nphi,
                                  double* __restrict__ phi_center0,
                                  long long* __restrict__ ring_start,
                                  double* __restrict__ z_edges){
    long long Nr = 4*nside - 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(long long i = tid+1; i <= Nr; i += gridDim.x * blockDim.x){
        double z; int npr; int s; double dphi;
        if(i < nside){
            z = 1.0 - (double)(i*i) / (3.0*(double)(nside*nside));
            npr = 4 * (int)i;
            s = 1;
            dphi = 2.0*PI() / (double)npr;
            phi_center0[i-1] = (1.0 - 0.5*(double)s) * dphi;
        }else if(i <= 3*nside){
            z = (double)(2*nside - (long long)i) * (2.0/(3.0*(double)nside));
            npr = 4 * (int)nside;
            s = (int)((i - nside) & 1LL);
            dphi = 2.0*PI() / (double)npr;
            phi_center0[i-1] = (1.0 - 0.5*(double)s) * dphi;
        }else{
            long long ip = 4*nside - i;
            z = - (1.0 - (double)(ip*ip) / (3.0*(double)(nside*nside)));
            npr = 4 * (int)ip;
            s = 1;
            dphi = 2.0*PI() / (double)npr;
            phi_center0[i-1] = (1.0 - 0.5*(double)s) * dphi;
        }
        z_centers[i-1] = z;
        theta_centers[i-1] = acos(fmax(-1.0, fmin(1.0, z)));
        nphi[i-1] = npr;
    }
    if(tid==0){
        long long NrL = 4*nside - 1;
        ring_start[0] = 0;
        for(long long i=0;i<NrL;i++){
            ring_start[i+1] = ring_start[i] + (long long)nphi[i];
        }
        z_edges[0] = 1.0;
        for(long long i=0;i<NrL-1;i++){
            z_edges[i+1] = 0.5*(z_centers[i] + z_centers[i+1]);
        }
        z_edges[NrL] = -1.0;
    }
}
} // extern "C"

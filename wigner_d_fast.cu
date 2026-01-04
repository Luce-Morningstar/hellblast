
extern "C"{
__device__ __forceinline__ double PI(){return 3.141592653589793238462643383279502884;}

// Compute full d^l_{m m'}(beta) for one l and a batch of betas, with shared-memory tiling.
// This keeps the closed-form sum (robust) but reduces redundant pow() work and improves locality.
// Layout: out[b, im, ip] with b = beta index, im = m index (-l..l), ip = m' index (-l..l).
__global__ void wigner_d_full_tiled(
    const double* __restrict__ beta, int B,
    int l,
    double* __restrict__ out   // shape [B, 2l+1, 2l+1]
){
    extern __shared__ double sh[]; // [2*B] for ct/st per batch tile
    int tid = threadIdx.x;
    int ldim = 2*l + 1;
    int im = blockIdx.x;      // -l..l as 0..2l
    int ip = blockIdx.y;      // -l..l as 0..2l
    int b  = blockIdx.z;      // beta index

    if (im>=ldim || ip>=ldim || b>=B) return;

    double th = beta[b];
    double ct = cos(0.5*th);
    double st = sin(0.5*th);

    int m  = im - l;
    int mp = ip - l;

    // precompute factorial-like prefactor via lgamma
    auto lg = [] __device__ (int n){ return lgamma((double)n + 1.0); };
    double logpref = 0.5*( lg(l+m) + lg(l-m) + lg(l+mp) + lg(l-mp) );

    int kmin = max(0, m - mp);
    int kmax = min(l+m, l - mp);
    double sum = 0.0;
    double c = 0.0; // Kahan compensation
    for(int k=kmin; k<=kmax; ++k){
        int a = l + mp - k;
        int bpow = m - mp + 2*k;
        int cexp = l - m - k;
        if(a<0 || bpow<0 || cexp<0) continue;
        double logden = lg(l+m-k) + lg(k) + lg(mp-m+k) + lg(l-mp-k);
        double coeff = exp(logpref - logden);
        double term  = coeff * pow(ct, a) * pow(st, bpow);
        if ( ((k + mp - m) & 1) ) term = -term;
        double y = term - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    out[(b*ldim + im)*ldim + ip] = sum;
}
} // extern "C"

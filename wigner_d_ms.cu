
extern "C"{
__device__ __forceinline__ double PI(){return 3.141592653589793238462643383279502884;}

__device__ double logfac(int n){
    return lgamma((double)n + 1.0);
}
__device__ double choose_coeff_log(int l, int m, int s, int k){
    // log of coeff numerator sqrt((l+m)!(l-m)!(l+s)!(l-s)!)
    double num = 0.5*( logfac(l+m) + logfac(l-m) + logfac(l+s) + logfac(l-s) );
    // denom (l+m-k)! k! (s-m+k)! (l-s-k)!
    double den = logfac(l+m-k) + logfac(k) + logfac(s-m+k) + logfac(l-s-k);
    return num - den;
}
__device__ double pow_nonneg(double a, int p){
    // a>=0
    return (p<=0)?1.0:pow(a,(double)p);
}
// Compute d^l_{m,s}(theta) for l from max(|m|,|s|)..lmax for a single theta, fixed m,s
__global__ void wigner_d_ms_sum(
    const double* __restrict__ theta, int Ntheta,
    int lmax, int m, int s,
    double* __restrict__ out // [Ntheta, lmax+1], undefined for l<max(|m|,|s|)
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid>=Ntheta) return;
    double th = theta[tid];
    double ct = cos(0.5*th);
    double st = sin(0.5*th);
    int lmin = (abs(m)>abs(s)?abs(m):abs(s));
    for(int l=lmin; l<=lmax; ++l){
        // sum over k with valid factorial arguments
        int kmin = max(0, m - s);
        int kmax = min(l+m, l - s);
        double sum = 0.0;
        for(int k=kmin; k<=kmax; ++k){
            int a = l + s - k;
            int b = m - s + 2*k;
            int c = l - m - k;
            // powers must be non-negative
            if(a<0 || b<0 || c<0) continue;
            double sign = ((k & 1)? -1.0 : 1.0);
            double logc = choose_coeff_log(l, m, s, k);
            double coeff = exp(logc);
            double term = sign * coeff * pow_nonneg(ct, a) * pow_nonneg(st, b);
            sum += term;
        }
        out[tid*(lmax+1) + l] = sum;
    }
}
} // extern "C"

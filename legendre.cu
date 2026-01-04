
extern "C" __global__ void plm_recurrence(
    const double* __restrict__ x,   // [Ntheta]
    int Ntheta,
    int lmax,
    int m,
    double* __restrict__ Plm        // [Ntheta, lmax+1] row-major: theta-major
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Ntheta) return;
    double xi = x[tid];
    double somx2 = sqrt(fmax(0.0, (1.0 - xi) * (1.0 + xi))); // sqrt(1 - x^2)
    double pmm = 1.0;
    if (m > 0) {
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    Plm[tid*(lmax+1) + m] = pmm;
    if (m == lmax) return;
    double pmmp1 = xi * (2.0*m + 1.0) * pmm;
    Plm[tid*(lmax+1) + (m+1)] = pmmp1;
    double p_lm2 = pmm;
    double p_lm1 = pmmp1;
    for (int l = m+2; l <= lmax; ++l) {
        double a = (2.0*l - 1.0) * xi * p_lm1;
        double b = (l + m - 1.0) * p_lm2;
        double pl = (a - b) / (double)(l - m);
        Plm[tid*(lmax+1) + l] = pl;
        p_lm2 = p_lm1;
        p_lm1 = pl;
    }
}

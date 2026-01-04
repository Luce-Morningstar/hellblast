# Planned CUDA kernels for HEALPix geometry (legacy note from M1):
# - RING ang2pix/pix2ang: ring_info(nside) -> theta[r], nphi[r], phi0[r], weight[r]; kernels map angles<->pixels.
# - NESTED ang2pix/pix2ang: face mapping with bit-interleave (Morton).
# This file remains as a design stub for reference.

\
    #pragma once
    #include <stdint.h>

    __device__ __forceinline__ uint32_t part1by1_u32(uint32_t x) {
        x = (x | (x << 8)) & 0x00FF00FFu;
        x = (x | (x << 4)) & 0x0F0F0F0Fu;
        x = (x | (x << 2)) & 0x33333333u;
        x = (x | (x << 1)) & 0x55555555u;
        return x;
    }
    __device__ __forceinline__ uint64_t part1by1_u64(uint64_t x) {
        x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
        x = (x | (x << 8))  & 0x00FF00FF00FF00FFull;
        x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0Full;
        x = (x | (x << 2))  & 0x3333333333333333ull;
        x = (x | (x << 1))  & 0x5555555555555555ull;
        return x;
    }
    __device__ __forceinline__ uint64_t interleave2_u64(uint32_t x, uint32_t y) {
        return (uint64_t(part1by1_u64(x)) << 1) | uint64_t(part1by1_u64(y));
    }

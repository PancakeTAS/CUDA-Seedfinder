#include "javarand.hu"
#define MULTIPLIER 25214903917L
#define ADDEND 11L
#define MODULOS 281474976710655L

// Updates the seed
__device__ void scramble(int64_t* seed, int64_t new_seed) {
    *seed = new_seed ^ MULTIPLIER;
}

// Updates the seed with a weaker hash of the world seed
__device__ void scrambleWeakSeed(int64_t* seed, int64_t worldseed, int32_t chunkX, int32_t chunkZ) {
    int32_t sX = chunkX >> 4;
    int32_t sZ = chunkZ >> 4;
    *seed = ((sX ^ sZ << 4L) ^ worldseed) ^ MULTIPLIER;
}

// Recreation of java.util.Random#next(bits)
__device__ int32_t next(int64_t* seed, int32_t bits) {
    // Calculate next seed
    *seed = (*seed * MULTIPLIER + ADDEND) & MODULOS;
    // Return required bits
    return (int) (*seed >> (48 - bits));
}

// Recreation of java.util.Random#nextInt(bound)
__device__ int32_t nextInt(int64_t* seed, int32_t bound) {
    int bits, value;

    do {
        // Generate new int
        bits = next(seed, 31);
        // Fit new int within bounds
        value = bits % bound;
    } while (bits - value + (bound - 1) < 0); // Rerun until value fits
    
    return value;
}

__device__ int32_t nextIntPower(int64_t* seed, int32_t bound) {
    return (int32_t) ((bound * (int64_t) next(seed, 31)) >> 31);
}

// Recreation of java.util.Random#nextDouble()
__device__ double nextDouble(int64_t *seed) {
    int64_t x = (int64_t) next(seed, 26);
    x <<= 27;
    x += next(seed, 27);
    return (int64_t) x / (double) (1ULL << 53);
}

// Skips next bytes
__device__ void skipNextN(int64_t *seed, int64_t n) {
    int64_t m = 1;
    int64_t a = 0;
    int64_t im = 0x5deece66dLL;
    int64_t ia = 0xb;
    int64_t k;

    for (k = n; k; k >>= 1) {
        if (k & 1) {
            m *= im;
            a = im * a + ia;
        }
        ia = (im + 1) * ia;
        im *= im;
    }

    *seed = *seed * m + a;
    *seed &= 0xffffffffffffLL;
}

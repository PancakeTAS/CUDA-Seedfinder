#include "javarand.hu"
#define MULTIPLIER 25214903917L
#define ADDEND 11L
#define MODULOS 281474976710655L

// Updates the seed
__device__ void scramble(uint64_t* seed, uint64_t new_seed) {
    *seed = new_seed ^ MULTIPLIER;
}

// Updates the seed with a weaker hash of the world seed
__device__ void scrambleWeakSeed(uint64_t* seed, uint64_t worldseed, uint32_t chunkX, uint32_t chunkZ) {
    uint32_t sX = chunkX >> 4;
    uint32_t sZ = chunkZ >> 4;
    *seed = ((sX ^ sZ << 4L) ^ worldseed) ^ MULTIPLIER;
}

// Recreation of java.util.Random#next(bits)
__device__ uint32_t next(uint64_t* seed, uint32_t bits) {
    // Calculate next seed
    *seed = (*seed * MULTIPLIER + ADDEND) & MODULOS;
    // Return required bits
    return (int) (*seed >> (48 - bits));
}

// Recreation of java.util.Random#nextInt(bound)
__device__ uint32_t nextInt(uint64_t* seed, uint32_t bound) {
    int bits, value;

    /* This code was removed because it seemed unnecessary
    do {
    */
        // Generate new unsigned int
        bits = next(seed, 31);
        // Fit new unsigned int within bounds
        value = bits % bound;
    /*
    } while (bits - value + (bound - 1) < 0); // Rerun until value fits
    */
    return value;
}
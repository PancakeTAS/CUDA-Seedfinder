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

// Updates the seed with a stronger hash of the world seed
__device__ void scrambleCarverSeed(int64_t* seed, int64_t worldseed, int32_t chunkX, int32_t chunkZ) {
    scramble(seed, worldseed);
    scramble(seed, (int64_t) chunkX * nextLong(seed) ^ (int64_t) chunkZ * nextLong(seed) ^ worldseed);
}

// Updates the seed with a hash of a block position
__device__ void scramblePositionSeed(int64_t* seed, int32_t x, int32_t y, int32_t z) {
    int64_t i = ((int64_t) x * 3129871L) ^ ((int64_t) z * 116129781L) ^ y;
    scramble(seed, (i * i * 42317861L + i * 11L) >> 16);
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

// Recreation of java.util.Random#nextInt(bound)
__device__ int32_t nextIntPower(int64_t* seed, int32_t bound) {
    return (int32_t) ((bound * (int64_t) next(seed, 31)) >> 31);
}

// Recreation of java.util.Random#nextLong()
__device__ int64_t nextLong(int64_t* seed) {
    return ((int64_t) next(seed, 32) << 32) + next(seed, 32);
}

// Recreation of java.util.Random#nextFloat
__device__ float nextFloat(int64_t* seed) {
    return next(seed, 24) / ((float) (1 << 24));
}
#include "perlin.hu"

#define AMPLITUDE 1.111111
#define LACUNA 0.333333
#define PERSIST 0.015625
#define SKIPJRND 1572

__device__ void make_octave(int64_t* seed, perlin_noise* noise) {
    // Generate abc values
    noise->a = nextDouble(seed) * 256.0;
    noise->b = nextDouble(seed) * 256.0;
    noise->c = nextDouble(seed) * 256.0;

    // Fill d with base values
    int32_t i = 0;
    for (i = 0; i < 256; i++) {
        noise->d[i] = i;
    }

    // Add noise to d
    for (i = 0; i < 256; i++) {
		int32_t n3 = nextInt(seed, 256 - i) + i;
        int32_t n4 = noise->d[i];
        noise->d[i] = noise->d[n3];
        noise->d[n3] = n4;
        noise->d[i + 256] = noise->d[i];
    }
}

// Makes a nether noise layer
__device__ void make_nether_layer(nether_noise *noise, int64_t structure_seed) {
    int64_t seed;

    // Generate first 2 octaves of each perlin generator of the double perlin generator for temperature
    scramble(&seed, structure_seed);
	skipNextN(&seed, SKIPJRND);
	make_octave(&seed, &noise->temperature.octA.octave0);
	make_octave(&seed, &noise->temperature.octA.octave1);
	skipNextN(&seed, SKIPJRND);
	make_octave(&seed, &noise->temperature.octB.octave0);
	make_octave(&seed, &noise->temperature.octB.octave1);

    // Generate last 2 octaves of each perlin generator of the double perlin generator for humidity
    scramble(&seed, structure_seed+1);
	skipNextN(&seed, SKIPJRND);
	make_octave(&seed, &noise->humidity.octA.octave0);
	make_octave(&seed, &noise->humidity.octA.octave1);
	skipNextN(&seed, SKIPJRND);
	make_octave(&seed, &noise->humidity.octB.octave0);
	make_octave(&seed, &noise->humidity.octB.octave1);
}
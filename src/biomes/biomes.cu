#include "biomes.hu"

#define SKIPJRND 1572

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

// Obtains the nether biome at a given position
__device__ int get_nether_biome(NetherNoise *noise, int x, int z) {
    const float npoints[5][4] = {
        { 0,    0,      0,              0       },
        { 0,   -0.5,    0,              1     },
        { 0.4,  0,      0,              2     },
        { 0,    0.5,    0.375*0.375,    3     },
        {-0.5,  0,      0.175*0.175,    4     },
    };

    float temp = sample_double_perlin(&noise->temperature, x, z);
    float humidity = sample_double_perlin(&noise->humidity, x, z);

    int i, id = 0;
    float dmin = 0xfffffd00;
    float dmin2 = 0xfffffd00;
    for (i = 0; i < 5; i++) {
        float dx = npoints[i][0] - temp;
        float dy = npoints[i][1] - humidity;
        float dsq = dx*dx + dy*dy + npoints[i][2];
        if (dsq < dmin) {
            dmin2 = dmin;
            dmin = dsq;
            id = i;
        } else if (dsq < dmin2) {
            dmin2 = dsq;
        }
    }

    id = (int) npoints[id][3];
    return id;
}
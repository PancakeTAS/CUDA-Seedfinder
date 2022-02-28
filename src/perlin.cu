#include "perlin.hu"

#define AMPLITUDE 1.111111
#define LACUNA 0.333333
#define PERSIST 0.0156250
#define SKIPJRND 1572
#define MULTIP 1.01812688822

__device__ void make_octave(int64_t* seed, perlin_noise* noise) {
    // Generate abc values
    noise->a = nextDouble(seed) * 256.0;
    noise->b = nextDouble(seed) * 256.0;
    noise->c = nextDouble(seed) * 256.0;

    // Fill d with base values
    int i = 0;
    for (i = 0; i < 512; i++) {
        noise->d[i] = 0;
    }
    for (i = 0; i < 256; i++) {
        noise->d[i] = i;
    }

    // Add noise to d
    for (i = 0; i < 256; i++) {
        int bound = 256 - i;
        int m = bound - 1;
        int n3;
        if ((m & bound) == 0) {
            n3 = nextIntPower(seed, 256 - i) + i;
        } else {
            n3 = nextInt(seed, 256 - i) + i;
        }
        int n4 = noise->d[i];
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

__device__ double maintain_precision(double x) {
    return x - floor(x / 33554432.0 + 0.5) * 33554432.0;
}

__device__ double lerp(double part, double from, double to) {
    return from + part * (to - from);
}

__device__ __constant__ double cEdgeX[] = {1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0,-1.0, 0.0};
__device__ __constant__ double cEdgeY[] = {1.0, 1.0,-1.0,-1.0, 0.0, 0.0, 0.0, 0.0, 1.0,-1.0, 1.0,-1.0,  1.0,-1.0, 1.0,-1.0};
__device__ __constant__ double cEdgeZ[] = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0,-1.0,-1.0,  0.0, 1.0, 0.0,-1.0};

__device__ double indexedLerp(int idx, double d1, double d2, double d3) {
    idx &= 0xf;
    return cEdgeX[idx] * d1 + cEdgeY[idx] * d2 + cEdgeZ[idx] * d3;
}

// Samples a 3D point of a given noise with y 0
__device__ double sample_perlin(perlin_noise *rnd, double d1, double d2, double d3) {
    d1 += rnd->a;
    d2 += rnd->b;
    d3 += rnd->c;
    int i1 = (int)d1 - (int)(d1 < 0);
    int i2 = (int)d2 - (int)(d2 < 0);
    int i3 = (int)d3 - (int)(d3 < 0);
    d1 -= i1;
    d2 -= i2;
    d3 -= i3;
    double t1 = d1*d1*d1 * (d1 * (d1*6.0-15.0) + 10.0);
    double t2 = d2*d2*d2 * (d2 * (d2*6.0-15.0) + 10.0);
    double t3 = d3*d3*d3 * (d3 * (d3*6.0-15.0) + 10.0);

    i1 &= 0xff;
    i2 &= 0xff;
    i3 &= 0xff;

    int a1 = rnd->d[i1]   + i2;
    int a2 = rnd->d[a1]   + i3;
    int a3 = rnd->d[a1+1] + i3;
    int b1 = rnd->d[i1+1] + i2;
    int b2 = rnd->d[b1]   + i3;
    int b3 = rnd->d[b1+1] + i3;

    printf("%f, %f, %f, %d, %d\n", d1, d3, d3, a3, rnd->d[a3]);

    double l1 = indexedLerp(rnd->d[a2],   d1,   d2,   d3);
    double l2 = indexedLerp(rnd->d[b2],   d1-1, d2,   d3);
    double l3 = indexedLerp(rnd->d[a3],   d1,   d2-1, d3);
    double l4 = indexedLerp(rnd->d[b3],   d1-1, d2-1, d3);
    double l5 = indexedLerp(rnd->d[a2+1], d1,   d2,   d3-1);
    double l6 = indexedLerp(rnd->d[b2+1], d1-1, d2,   d3-1);
    double l7 = indexedLerp(rnd->d[a3+1], d1,   d2-1, d3-1);
    double l8 = indexedLerp(rnd->d[b3+1], d1-1, d2-1, d3-1);

    printf("%f, %f, %f, %f, %f, %f, %f, %f\n", l1, l2, l3, l4, l5, l6, l7, l8);

    l1 = lerp(t1, l1, l2);
    l3 = lerp(t1, l3, l4);
    l5 = lerp(t1, l5, l6);
    l7 = lerp(t1, l7, l8);

    l1 = lerp(t2, l1, l3);
    l5 = lerp(t2, l5, l7);

    return lerp(t3, l1, l5);
}

// Samples a 3D point of a given octave with y 0
__device__ double sample_octave(octave_noise *noise, double x, double z) {
    double persist = PERSIST;
    double lacuna = LACUNA;
    double v = 0;
    
    double ax = maintain_precision(x * persist);
    double ay = maintain_precision(0);
    double az = maintain_precision(z * persist);
    printf("%f, %f, %f\n", ax, ay, az);
    v += lacuna * sample_perlin(&noise->octave0, ax, ay, az);
    printf("%f\n", v);
    persist *= 0.5;
    lacuna *= 2.0;

    ax = maintain_precision(x * persist);
    az = maintain_precision(z * persist);
    printf("%f, %f, %f\n", ax, ay, az);
    v += lacuna * sample_perlin(&noise->octave1, ax, ay, az);
    printf("%f\n", v);

    return v;
}

// Samples a 3D point of a given double perlin layer with y 0
__device__ double sample_double_perlin(double_perlin_noise* noise, int32_t x, int32_t z) {
    return (sample_octave(&noise->octA, x, z) + sample_octave(&noise->octB, x*MULTIP, z*MULTIP)) * AMPLITUDE;
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


    printf("%f, %f\n", temp, humidity);

    int i, id = 0;
    float dmin = 0xfffffd00;
    float dmin2 = 0xfffffd00;
    for (i = 0; i < 5; i++) {
        float dx = npoints[i][0] - temp;
        float dy = npoints[i][1] - humidity;
        float dsq = dx*dx + dy*dy + npoints[i][2];
        printf("%f\n", dsq);
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
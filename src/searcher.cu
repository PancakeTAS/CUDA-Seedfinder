#include "searcher.hu"

#define A 341873128712LL
#define B 132897987541LL

__device__ double clampedLerp(double part, double from, double to) {
    if (part <= 0) return from;
    if (part >= 1) return to;
    return lerp(part, from, to);
}

__device__ double lerp2(
        double dx, double dy, double v00, double v10, double v01, double v11)
{
    return lerp(dy, lerp(dx, v00, v10), lerp(dx, v01, v11));
}

__device__ double lerp3(
        double dx, double dy, double dz,
        double v000, double v100, double v010, double v110,
        double v001, double v101, double v011, double v111)
{
    v000 = lerp2(dx, dy, v000, v100, v010, v110);
    v001 = lerp2(dx, dy, v001, v101, v011, v111);
    return lerp(dz, v000, v001);
}

__device__ double sampleSurfaceNoise(surface_noise *sn, int x, int y, int z) {
    double xzScale = 684.412 * sn->xzScale;
    double yScale = 684.412 * sn->yScale;
    double xzStep = xzScale / sn->xzFactor;
    double yStep = yScale / sn->yFactor;

    double minNoise = 0;
    double maxNoise = 0;
    double mainNoise = 0;
    double persist = 1.0;
    double dx, dy, dz, sy, ty;
    int i;

    for (i = 0; i < 16; i++)
    {
        dx = maintain_precision(x * xzScale * persist);
        dy = maintain_precision(y * yScale  * persist);
        dz = maintain_precision(z * xzScale * persist);
        sy = yScale * persist;
        ty = y * sy;

        minNoise += sample_perlin(&sn->octmin.octaves[i], dx, dy, dz, sy, ty) / persist;
        maxNoise += sample_perlin(&sn->octmax.octaves[i], dx, dy, dz, sy, ty) / persist;

        if (i < 8)
        {
            dx = maintain_precision(x * xzStep * persist);
            dy = maintain_precision(y * yStep  * persist);
            dz = maintain_precision(z * xzStep * persist);
            sy = yStep * persist;
            ty = y * sy;
            mainNoise += sample_perlin(&sn->octmain.octaves[i], dx, dy, dz, sy, ty) / persist;
        }
        persist /= 2.0;
    }

    return clampedLerp(0.5 + 0.05*mainNoise, minNoise/512.0, maxNoise/512.0);
}

__device__ double simplexGrad(int idx, double x, double y, double z, double d) {
    double con = d - x*x - y*y - z*z;
    if (con < 0)
        return 0;
    con *= con;
    return con * con * indexedLerp(idx, x, y, z);
}

__device__ double sampleSimplex2D(perlin_noise *noise, double x, double y) {
    const double SKEW = 0.5 * (sqrtf(3) - 1.0);
    const double UNSKEW = (3.0 - sqrtf(3)) / 6.0;

    double hf = (x + y) * SKEW;
    int hx = (int)floor(x + hf);
    int hz = (int)floor(y + hf);
    double mhxz = (hx + hz) * UNSKEW;
    double x0 = x - (hx - mhxz);
    double y0 = y - (hz - mhxz);
    int offx = (x0 > y0);
    int offz = !offx;
    double x1 = x0 - offx + UNSKEW;
    double y1 = y0 - offz + UNSKEW;
    double x2 = x0 - 1.0 + 2.0 * UNSKEW;
    double y2 = y0 - 1.0 + 2.0 * UNSKEW;
    int gi0 = noise->d[0xff & (hz)];
    int gi1 = noise->d[0xff & (hz + offz)];
    int gi2 = noise->d[0xff & (hz + 1)];
    gi0 = noise->d[0xff & (gi0 + hx)];
    gi1 = noise->d[0xff & (gi1 + hx + offx)];
    gi2 = noise->d[0xff & (gi2 + hx + 1)];
    double t = 0;
    t += simplexGrad(gi0 % 12, x0, y0, 0.0, 0.5);
    t += simplexGrad(gi1 % 12, x1, y1, 0.0, 0.5);
    t += simplexGrad(gi2 % 12, x2, y2, 0.0, 0.5);
    return 70.0 * t;
}

__device__ float getEndHeightNoise(perlin_noise *en, int x, int z, int range) {
    int hx = x / 2;
    int hz = z / 2;
    int oddx = x % 2;
    int oddz = z % 2;
    int i, j;

    int64_t h = 64 * (x*(int64_t)x + z*(int64_t)z);
    if (range == 0)
        range = 12;

    for (j = -range; j <= range; j++)
    {
        for (i = -range; i <= range; i++)
        {
            int64_t rx = hx + i;
            int64_t rz = hz + j;
            uint64_t rsq = rx*rx + rz*rz;
            uint16_t v = 0;
            if (rsq > 4096 && sampleSimplex2D(en, rx, rz) < -0.9f)
            {
                v = (llabs(rx) * 3439 + llabs(rz) * 147) % 13 + 9;
                rx = (oddx - i * 2);
                rz = (oddz - j * 2);
                rsq = rx*rx + rz*rz;
                int64_t noise = rsq * v*v;
                if (noise < h)
                    h = noise;
            }
        }
    }

    float ret = 100 - sqrtf((float) h);
    if (ret < -100) ret = -100;
    if (ret > 80) ret = 80;
    return ret;
}

__device__ void sampleNoiseColumnEnd(double column[], surface_noise *sn, perlin_noise *en, int x, int z, int colymin, int colymax) {
    double depth = getEndHeightNoise(en, x, z, 0) - 8.0f;
    int y;
    for (y = colymin; y <= colymax; y++) {
        double noise = sampleSurfaceNoise(sn, x, y, z);
        noise += depth; // falloff for the End is just the depth
        // clamp top and bottom slides from End settings
        noise = lerp((32 + 46 - y) / 64.0, -3000, noise);
        noise = lerp((y - 1) / 7.0, -30, noise);
        column[y - colymin] = noise;
    }
}

__device__ int getSurfaceHeight(
        const double ncol00[], const double ncol01[],
        const double ncol10[], const double ncol11[],
        int colymin, int colymax, int blockspercell, double dx, double dz) {
    int y, celly;
    for (celly = colymax-1; celly >= colymin; celly--) {
        int idx = celly - colymin;
        double v000 = ncol00[idx];
        double v001 = ncol01[idx];
        double v100 = ncol10[idx];
        double v101 = ncol11[idx];
        double v010 = ncol00[idx+1];
        double v011 = ncol01[idx+1];
        double v110 = ncol10[idx+1];
        double v111 = ncol11[idx+1];

        for (y = blockspercell - 1; y >= 0; y--)
        {
            double dy = y / (double) blockspercell;
            double noise = lerp3(dy, dx, dz, // Note: not x, y, z
                v000, v010, v100, v110,
                v001, v011, v101, v111);
            if (noise > 0)
                return celly * blockspercell + y;
        }
    }
    return 0;
}

__global__ void startSearch(int64_t structureSeedOffset) {
    // Figure out what seed to check
    int64_t index = ((structureSeedOffset + threadIdx.x) + (((int64_t) blockIdx.x) * 256));
    int64_t structureSeed = (index << 16) + structureSeedOffset;

    int x = 28;
    int z = -29;

    perlin_noise noise;
    int64_t seed;
    scramble(&seed, structureSeed);
    skipNextN(&seed, 17292);
    make_octave(&seed, &noise);

    surface_noise sn;
    scramble(&seed, structureSeed);
    make_octave(&seed, &sn.octmin, sn.oct+0, -15, 16);
    make_octave(&seed, &sn.octmax, sn.oct+16, -15, 16);
    make_octave(&seed, &sn.octmain, sn.oct+32, -7, 8);
    sn.xzScale = 2.0;
    sn.yScale = 1.0;
    sn.xzFactor = 80;
    sn.yFactor = 160;

    int cellx = (x >> 3);
    int cellz = (z >> 3);
    double dx = (x & 7) / 8.0;
    double dz = (z & 7) / 8.0;
    
    double ncol00[33];
    double ncol01[33];
    double ncol10[33];
    double ncol11[33];
    sampleNoiseColumnEnd(ncol00, &sn, &noise, cellx, cellz, 0, 32);
    sampleNoiseColumnEnd(ncol01, &sn, &noise, cellx, cellz+1, 0, 32);
    sampleNoiseColumnEnd(ncol10, &sn, &noise, cellx+1, cellz, 0, 32);
    sampleNoiseColumnEnd(ncol11, &sn, &noise, cellx+1, cellz+1, 0, 32);

    int i = getSurfaceHeight(ncol00, ncol01, ncol10, ncol11, 0, 32, 4, dx, dz);
    if (i > 80)
        printf("Node on seed %llu is %d blocks high\n", structureSeed, i+15);
}
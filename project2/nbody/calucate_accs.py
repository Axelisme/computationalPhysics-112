import cupy as cp

BLOCK_N = cp.cuda.Device().attributes["MaxThreadsPerBlock"]
BLOCK_X = int(BLOCK_N**0.5)
BLOCK_Y = BLOCK_X

gravity_kernel = cp.RawKernel(
    r"""
extern "C" __device__
void add_accs(
    float *pos1,
    float *pos2,
    int dim, float G, float rsoft2, float mass1, float mass2,
    float *acc1,
    float *acc2
) {
    // Calculate relative distance squared
    float rel_r2 = rsoft2;
    for (int k = 0; k < dim; k++) {
        float rel_x = pos1[k] - pos2[k];
        rel_r2 += rel_x * rel_x;
    }

    // pre-calculate some values
    float inv_r = rsqrt(rel_r2);
    float G_r3 = G * inv_r * inv_r * inv_r;

    // Calculate gravity force
    for (int k = 0; k < dim; k++) {
        float Gx_r3 = G_r3 * (pos1[k] - pos2[k]);
        atomicAdd(acc1 + k, -Gx_r3 * mass2);
        atomicAdd(acc2 + k, Gx_r3 * mass1);
    }
}


extern "C" __global__
void gravity_kernel(float *D, float *masses, int N, int dim, float G, float rsoft, float *accs) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int ntidx = gridDim.x * blockDim.x;

    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int ntidy = gridDim.y * blockDim.y;

    float rsoft2 = rsoft * rsoft;

    int odd = N % 2;
    int even = 1 - odd;
    for (int idx = tidx; idx < N + even; idx += ntidx) {
        for (int jdx = tidy; jdx < N/2; jdx += ntidy) {
            // calculate i and j
            // rectangular box (RB) strategy
            int i, j;
            if (jdx > idx || (jdx == idx && odd)) {
                i = N - 1 - idx;
                j = N - 1 - jdx - odd;
            } else {
                i = idx + even;
                j = jdx;
            }

            // calculate the force between i and j
            add_accs(
                D + i * dim,
                D + j * dim,
                dim, G, rsoft2, masses[i], masses[j],
                accs + i * dim,
                accs + j * dim
            );
        }
    }
}
""",
    "gravity_kernel",
)


def calculate_accs(G, rsoft, nparticles, masses, positions):
    poss = cp.array(positions, dtype=cp.float32)
    masses = cp.array(masses, dtype=cp.float32)
    accs = cp.zeros_like(poss, dtype=cp.float32)

    gridn = nparticles**2 // (4 * BLOCK_N)
    gravity_kernel(
        grid=(gridn,),
        block=(BLOCK_X, BLOCK_Y),
        args=(
            poss,
            masses,
            cp.int32(nparticles),
            cp.int32(positions.shape[1]),
            cp.float32(G),
            cp.float32(rsoft),
            accs,
        ),
    )
    return accs.get()

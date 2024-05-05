import cupy as cp

BLOCK_N = cp.cuda.Device().attributes["MaxThreadsPerBlock"]
BLOCK_X = int(BLOCK_N**0.5)
BLOCK_Y = BLOCK_X

energy_kernel = cp.RawKernel(
    r"""
extern "C" __device__
void fill_energy(
    float *pos1,
    float *pos2,
    int dim, float G, float rsoft2, float mass1, float mass2,
    float *energy
) {
    // Calculate relative distance squared
    float rel_r2 = rsoft2;
    for (int k = 0; k < dim; k++) {
        float rel_x = pos1[k] - pos2[k];
        rel_r2 += rel_x * rel_x;
    }

    // Calculate gravity potential energy
    atomicAdd(energy, -G * mass1 * mass2 * rsqrt(rel_r2));
}


extern "C" __global__
void gravity_kernel(float *D, float *masses, int N, int dim, float G, float rsoft, float *energy) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int ntidx = gridDim.x * blockDim.x;

    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int ntidy = gridDim.y * blockDim.y;

    float rsoft2 = rsoft * rsoft;

    int odd = N % 2;
    for (int idx = tidx + 1 - odd; idx < N; idx += ntidx) {
        for (int jdx = tidy; jdx < N/2; jdx += ntidy) {
            // calculate i and j
            // rectangular box (RB) strategy
            int i = (jdx < idx) ? idx : N - idx - odd;
            int j = (jdx < idx) ? jdx : N - 1 - jdx - odd;

            // calculate the energy between i and j
            fill_energy(
                D + i * dim,
                D + j * dim,
                dim, G, rsoft2, masses[i], masses[j],
                energy + idx
            );
        }
    }
}
""",
    "gravity_kernel",
)


def get_energy(G, rsoft, nparticles, masses, positions):
    poss = cp.array(positions, dtype=cp.float32)
    masses = cp.array(masses, dtype=cp.float32)
    energy = cp.zeros((nparticles + 1,), dtype=cp.float32)

    gridn = nparticles**2 // (4 * BLOCK_N)
    energy_kernel(
        grid=(gridn,),
        block=(BLOCK_X, BLOCK_Y),
        args=(
            poss,
            masses,
            cp.int32(nparticles),
            cp.int32(positions.shape[1]),
            cp.float32(G),
            cp.float32(rsoft),
            energy,
        ),
    )
    return energy.sum().get().item()


if __name__ == "__main__":
    import numpy as np

    num_particles = 5
    masses = np.ones((num_particles,))
    positions = np.random.randn(num_particles, 3)
    E = get_energy(1.0, 1e-6, num_particles, masses, positions)
    print(E)

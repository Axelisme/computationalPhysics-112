# %%
import matplotlib.pyplot as plt
import numpy as np
from nbody import NBodySimulator, Particles, load_files, save_movie

# %%
num_particles = 1000
seed = 0
particles = Particles(N=num_particles)

np.random.seed(seed)
masses = np.ones((num_particles, 1)) * 20 / num_particles
positions = np.random.randn(num_particles, 3)
velocities = np.random.randn(num_particles, 3)
accelerations = np.random.randn(num_particles, 3)

particles.set_particles(positions, velocities, accelerations, masses)

# %%
simulation = NBodySimulator(particles=particles)
simulation.setup(
    G=1.0,
    rsoft=0.01,
    # method='rk4',
    method="leapfrog",
    io_freq=30,
)
# simulation.evolve(dt=0.01, tmax=10.0)

# %%
fns = load_files("nbody")

au = 5.0
save_movie(fns, lengthscale=au, filename="nbody_earth_sun.mp4", fps=10)

snapshot_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
au = 5.0

# plot on x -y plane
fig = plt.figure()
fig.set_dpi(300)
plt.tight_layout()
plt.style.use("dark_background")
for i, time in enumerate(snapshot_times):
    file = fns[int((time + 1e-6 / 0.01))]
    _, _, x, y, *_ = np.loadtxt(file)
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xlim(-au, au)
    ax.set_ylim(-au, au)
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f"t={time}")
    ax.plot(x, y, ",")
plt.savefig("nbody_earth_sun.png", dpi=300)

# %%

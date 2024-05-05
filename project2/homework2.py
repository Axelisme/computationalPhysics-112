import os

import matplotlib.pyplot as plt
import numpy as np
from nbody import NBodySimulator, Particles, load_files

num_particles = 40000
seed = 0
M = 20
G = 1.0
RSOFT = 0.01
DT = 0.01
T = 10.0


def run_simulation(method):
    np.random.seed(seed)
    masses = np.ones((num_particles,)) * M / num_particles
    positions = np.random.randn(num_particles, 3)
    velocities = np.random.randn(num_particles, 3)
    accelerations = np.random.randn(num_particles, 3)

    particles = Particles(N=num_particles)
    particles.set_particles(positions, velocities, accelerations, masses)

    simulation = NBodySimulator(particles=particles)
    simulation.setup(
        G=G,
        rsoft=RSOFT,
        method=method,
        io_freq=30,
        io_header=method,
    )

    simulation.evolve(dt=DT, tmax=T)


for method in ["euler", "rk2", "rk4", "leapfrog"]:
    print(f"Running {method} method...", end="")
    run_simulation(method)
    print("Done!")


def plot_energy(method):
    kenetic_energy = []
    potential_energy = []
    times = []

    particles = Particles(N=num_particles)
    for file in load_files(method):
        kE, pE, time = particles.load(file)
        kenetic_energy.append(kE)
        potential_energy.append(pE)
        times.append(time)
    total_energy = [k + p for k, p in zip(kenetic_energy, potential_energy)]

    plt.style.use("classic")
    fig, ax = plt.subplots()

    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_xlim(0, T)
    ax.plot(times, kenetic_energy, label="Kenetic Energy")
    ax.plot(times, potential_energy, label="Potential Energy")
    ax.plot(times, total_energy, label="Total Energy")
    ax.legend()
    plt.savefig(f"fig/{method}_energy.png", dpi=300)
    plt.close()


os.makedirs("fig", exist_ok=True)
for method in ["euler", "rk2", "rk4", "leapfrog"]:
    print(f"Plot energy of {method} method...", end="")
    plot_energy(method)
    print("Done!")


def plot_snapshot(method):
    snapshot_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    au = 5.0

    fns = load_files(method)

    # plot on x -y plane
    plt.style.use("classic")
    fig = plt.figure()
    fig.set_dpi(300)
    fig.tight_layout()
    for i, time in enumerate(snapshot_times):
        file = fns[int((time + 1e-6) / (30 * DT))]
        _, _, x, y, *_ = np.loadtxt(file)
        ax = fig.add_subplot(2, 3, i + 1)
        ax.set_xlim(-au, au)
        ax.set_ylim(-au, au)
        ax.set_aspect("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"t={time}")
        ax.plot(x, y, ",", markersize=1)
    fig.savefig(f"fig/{method}.png", dpi=300)


for method in ["euler", "rk2", "rk4", "leapfrog"]:
    print(f"Plotting snapshot of {method} method...", end="")
    plot_snapshot(method)
    print("Done!")

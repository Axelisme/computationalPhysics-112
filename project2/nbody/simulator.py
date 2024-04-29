import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads

"""
The N-Body Simulator class is responsible for simulating the motion of N bodies



"""


@njit(
    "float64[:,:](float64, float64, int64, float64[:,:], float64[:,:])", parallel=True
)
def _calculate_acceleration(G, rsoft, nparticles, masses, positions):
    """
    Calculate the acceleration of the particles
    """
    accelerations = np.zeros_like(positions)

    for i in prange(nparticles):
        for j in prange(nparticles):
            if j > i:
                rij = positions[i] - positions[j]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                force = -G * masses[i, 0] * masses[j, 0] / r**3 * rij
                accelerations[i] = accelerations[i] + force / masses[i, 0]
                accelerations[j] = accelerations[j] - force / masses[j, 0]

    return accelerations


class NBodySimulator:
    def __init__(self, particles: Particles):
        self.particles = particles

        return

    def setup(
        self,
        G=1,
        rsoft=0.01,
        method="RK4",
        io_freq=10,
        io_header="nbody",
        io_screen=True,
        visualization=False,
    ):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output.
        :param io_header: the output header
        :param io_screen: print message on screen or not.
        :param visualization: on the fly visualization or not.
        """

        self.G = G
        self.rsoft = rsoft
        self.method = method.lower()
        self.io_freq = io_freq
        self.io_header = io_header
        self.io_screen = io_screen
        self.visualization = visualization

        return

    def evolve(self, dt: float, tmax: float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve

        """

        for i, t in enumerate(np.arange(0, tmax, dt)):
            self._advance_particles(dt, self.particles)
            self.particles.time = t
            if i % self.io_freq == 0:
                # print info to screen
                if self.io_screen:
                    print("Time: ", t, "dt: ", dt)

                # check output directroy
                folder = "data_" + self.io_header
                Path(folder).mkdir(parents=True, exist_ok=True)

                # output data
                fn = self.io_header + "_" + str(i).zfill(6) + ".dat"
                fn = folder + "/" + fn
                self.particles.output(fn)

                # visualization
                if self.visualization:
                    self.particles.draw()

        print("Simulation is done!")
        return

    def _calculate_acceleration(self, nparticles, masses, positions):
        """
        Calculate the acceleration of the particles
        """
        return _calculate_acceleration(
            self.G, self.rsoft, nparticles, masses, positions
        )

    def _advance_particles(self, dt, particles):
        match self.method:
            case "euler":
                particles = self._advance_particles_Euler(dt, particles)
            case "rk2":
                particles = self._advance_particles_RK2(dt, particles)
            case "rk4":
                particles = self._advance_particles_RK4(dt, particles)
            case _:
                raise ValueError(f"Invalid method: {self.method}")
        return

    def _advance_particles_Euler(self, dt, particles: Particles):
        nparticles = particles.N
        masses = particles.massA
        positions = particles.posA
        velocities = particles.velA
        accelerations = self._calculate_acceleration(nparticles, masses, positions)

        # do the Euler update
        positions += velocities * dt
        velocities += accelerations * dt

        # update the particles
        particles.set_particles(positions, velocities, accelerations)

        return particles

    def _advance_particles_RK2(self, dt, particles: Particles):
        nparticles = particles.nparticles
        mass = particles.massA

        pos = particles.posA
        vel = particles.velA
        acc = self._calculate_acceleration(nparticles, mass, pos)

        # do the RK2 update
        pos2 = pos + vel * dt
        vel2 = vel + acc * dt
        acc2 = self._calculate_acceleration(nparticles, mass, pos2)

        pos2 = pos2 + vel2 * dt
        vel2 = vel2 + acc2 * dt

        # average
        pos = 0.5 * (pos + pos2)
        vel = 0.5 * (vel + vel2)
        acc = self._calculate_acceleration(nparticles, mass, pos)

        # update the particles
        particles.set_particles(pos, vel, acc)

        return particles

    def _advance_particles_RK4(self, dt, particles: Particles):
        nparticles = particles.N
        mass = particles.massA

        # y0
        pos = particles.posA
        vel = particles.velA  # k1
        acc = self._calculate_acceleration(nparticles, mass, pos)  # k1

        dt2 = dt / 2
        # y1
        pos1 = pos + vel * dt2
        vel1 = vel + acc * dt2  # k2
        acc1 = self._calculate_acceleration(nparticles, mass, pos1)  # k2

        # y2
        pos2 = pos + vel1 * dt2
        vel2 = vel + acc1 * dt2  # k3
        acc2 = self._calculate_acceleration(nparticles, mass, pos2)  # k3

        # y3
        pos3 = pos + vel2 * dt
        vel3 = vel + acc2 * dt  # k4
        acc3 = self._calculate_acceleration(nparticles, mass, pos3)  # k4

        # rk4
        pos = pos + (vel + 2 * vel1 + 2 * vel2 + vel3) * dt / 6
        vel = vel + (acc + 2 * acc1 + 2 * acc2 + acc3) * dt / 6
        acc = self._calculate_acceleration(nparticles, mass, pos)

        # update the particles
        particles.set_particles(pos, vel, acc)

        return particles


if __name__ == "__main__":
    pass

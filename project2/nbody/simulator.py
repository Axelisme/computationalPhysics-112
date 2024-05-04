from pathlib import Path

import numpy as np

from .calucate_accs import calculate_accs
from .particles import Particles


class NBodySimulator:
    def __init__(self, particles: Particles):
        self.particles = particles
        self.time = particles.time

    def setup(
        self,
        G=1,
        rsoft=0.01,
        method="rk4",
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

    def evolve(self, dt: float, tmax: float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve

        """

        assert hasattr(self, "G"), "Please setup the simulation first!"

        for i, t in enumerate(np.arange(0, tmax, dt)):
            self._advance_particles(dt, self.particles)
            self.particles.time = t
            if i % self.io_freq == 0:
                # print info to screen
                if self.io_screen:
                    print(f"Time: {t: 2f}", "dt: ", dt)

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
        return calculate_accs(self.G, self.rsoft, nparticles, masses, positions)

    def _advance_particles(self, dt, particles):
        match self.method:
            case "euler":
                particles = self._advance_particles_Euler(dt, particles)
            case "rk2":
                particles = self._advance_particles_RK2(dt, particles)
            case "rk4":
                particles = self._advance_particles_RK4(dt, particles)
            case "leapfrog":
                particles = self._advance_particles_leapfrog(dt, particles)
            case _:
                raise ValueError(f"Invalid method: {self.method}")

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

    def _advance_particles_leapfrog(self, dt, particles: Particles):
        nparticles = particles.N
        mass = particles.masses

        pos = particles.posA
        vel = particles.velA
        acc = self._calculate_acceleration(nparticles, mass, pos)

        # do the leapfrog update
        vel += acc * dt / 2
        pos += vel * dt
        acc = self._calculate_acceleration(nparticles, mass, pos)

        vel += acc * dt / 2

        # update the particles
        particles.set_particles(pos, vel, acc)

        return particles


if __name__ == "__main__":
    pass

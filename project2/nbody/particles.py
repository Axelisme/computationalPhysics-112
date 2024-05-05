import numpy as np

from .get_energy import get_energy


class Particles:
    """
    Particle class to store particle properties
    """

    def __init__(self, N: int = 0):
        self.N = N
        self.massA = np.zeros(self.N)
        self.tagA = np.zeros(self.N, dtype=int)
        self.posA = np.zeros((self.N, 3))
        self.velA = np.zeros((self.N, 3))
        self.accA = np.zeros((self.N, 3))
        self.time = 0.0

    @property
    def nparticles(self):
        return self.N

    @property
    def masses(self):
        return self.massA

    @property
    def tags(self):
        return self.tagA

    @property
    def positions(self):
        return self.posA

    @property
    def velocities(self):
        return self.velA

    @property
    def accelerations(self):
        return self.accA

    @masses.setter
    def masses(self, m):
        if len(m) != self.N:
            raise ValueError("Number of particles does not match!")
        self.massA = m

    @tags.setter
    def tags(self, tag):
        if len(tag) != self.N:
            raise ValueError("Number of particles does not match!")
        self.tagA = tag

    @positions.setter
    def positions(self, pos):
        if len(pos) != self.N:
            raise ValueError("Number of particles does not match!")
        self.posA = pos

    @velocities.setter
    def velocities(self, vel):
        if len(vel) != self.N:
            raise ValueError("Number of particles does not match!")
        self.velA = vel

    @accelerations.setter
    def accelerations(self, acc):
        if len(acc) != self.N:
            raise ValueError("Number of particles does not match!")
        self.accA = acc

    def kenetic_energy(self):
        return 0.5 * np.sum(self.massA * np.sum(self.velA**2, axis=1))

    def potential_energy(self, G, rsoft):
        return get_energy(G, rsoft, self.N, self.massA, self.posA)

    def set_particles(self, pos, vel, acc, mass=None, time=None):
        assert len(pos) == len(vel) == len(acc) == self.N, "Inconsistent length"
        if mass is not None:
            assert len(mass) == self.N, "Inconsistent length"

        self.posA = pos
        self.velA = vel
        self.accA = acc
        if mass is not None:
            self.massA = mass
        if time is not None:
            self.time = time

    def add_particles(self, mass, pos, vel, acc):
        if (
            len(mass) != self.N
            or len(pos) != self.N
            or len(vel) != self.N
            or len(acc) != self.N
        ):
            raise ValueError("Inconsistent length of input arrays")
        self.N += len(mass)
        self.massA = np.vstack((self.massA, mass))
        self.tagA = np.arange(self.N)
        self.posA = np.vstack((self.posA, pos))
        self.velA = np.vstack((self.velA, vel))
        self.accA = np.vstack((self.accA, acc))

    def load(self, filename):
        with open(filename) as f:
            for _ in range(9):
                f.readline()
            time = float(f.readline().split("=")[-1])
            kE = float(f.readline().split("=")[-1])
            pE = float(f.readline().split("=")[-1])

        m, t, x, y, z, vx, vy, vz, ax, ay, az = np.loadtxt(filename)
        self.N = len(m)
        self.massA = m
        self.tagA = t
        self.posA = np.column_stack((x, y, z))
        self.velA = np.column_stack((vx, vy, vz))
        self.accA = np.column_stack((ax, ay, az))

        return kE, pE, time

    def output(self, filename, G, rsoft):
        massA = self.massA
        posA = self.posA
        velA = self.velA
        accA = self.accA
        tagA = self.tagA
        time = self.time

        header = """----------------------------------------------------
            Data from a 3D direct N-body simulation. 

            rows are i-particle; 
            coumns are :mass, tag, x ,y, z, vx, vy, vz, ax, ay, az

            NTHU, Computational Physics 

            ----------------------------------------------------
        """
        header += "Time = {}\n".format(time)
        header += f"Kenetic Energy = {self.kenetic_energy()}\n"
        header += f"Potential Energy = {self.potential_energy(G, rsoft)}\n"
        np.savetxt(
            filename,
            (
                massA,
                tagA,
                posA[:, 0],
                posA[:, 1],
                posA[:, 2],
                velA[:, 0],
                velA[:, 1],
                velA[:, 2],
                accA[:, 0],
                accA[:, 1],
                accA[:, 2],
            ),
            header=header,
        )

    def draw(self, dim=2):
        import matplotlib.pyplot as plt

        fig = plt.figure()

        if dim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(self.posA[:, 0], self.posA[:, 1], s=1)
            ax.set_xlabel("X [code unit]")
            ax.set_ylabel("Y [code unit]")
        elif dim == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(self.posA[:, 0], self.posA[:, 1], self.posA[:, 2], s=1)
            ax.set_xlabel("X [code unit]")
            ax.set_ylabel("Y [code unit]")
            ax.set_zlabel("Z [code unit]")
        else:
            raise ValueError("Invalid dimension")

        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
        return fig, ax

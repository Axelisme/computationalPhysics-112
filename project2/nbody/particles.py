import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from collections import namedtuple

@dataclass
class Particles:
    """
    Particle class to store particle properties
    """
    N: int
    massA: np.ndarray = field(init=False)
    tagA: np.ndarray  = field(init=False)
    time: float       = field(default=0.0)
    posA: np.ndarray  = field(init=False)
    velA: np.ndarray  = field(init=False)
    accA: np.ndarray  = field(init=False)

    def __post_init__(self):
        assert self.N >= 0, "Number of particles must be non-negative"
        self.massA = np.zeros(self.N)
        self.tagA = np.zeros(self.N, dtype=int)
        self.posA = np.zeros((self.N, 3))
        self.velA = np.zeros((self.N, 3))
        self.accA = np.zeros((self.N, 3))

    def _sub_particles(self, idx):
        sub_ps = Particles(N=len(idx), time=self.time)
        sub_ps.massA = self.massA[idx]
        sub_ps.tagA = self.tagA[idx]
        sub_ps.posA = self.posA[idx]
        sub_ps.velA = self.velA[idx]
        sub_ps.accA = self.accA[idx]
        return sub_ps

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return namedtuple('Particle', ['mass', 'tag', 'pos', 'vel', 'acc'])(
                self.massA[idx], self.tagA[idx], self.posA[idx], self.velA[idx], self.accA[idx]
                )
        elif isinstance(idx, slice):
            return self._sub_particles(idx)

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            self.massA[idx] = value.mass
            self.tagA[idx] = value.tag
            self.posA[idx] = value.pos
            self.velA[idx] = value.vel
            self.accA[idx] = value.acc
        elif isinstance(idx, slice):
            if not isinstance(value, Particles):
                raise ValueError("Cannot set with non-Particles object")
            if len(value) != len(idx):
                raise ValueError("Inconsistent length between slice and Particles object")
            self.massA[idx] = value.massA
            self.tagA[idx] = value.tagA
            self.posA[idx] = value.posA
            self.velA[idx] = value.velA
            self.accA[idx] = value.accA

    def __len__(self):
        return len(self.massA)

    def add_particles(self, mass, tag, pos, vel, acc):
        if len(mass) != self.N or \
              len(tag) != self.N or \
              len(pos) != self.N or \
              len(vel) != self.N or \
              len(acc) != self.N:
                raise ValueError("Inconsistent length of input arrays")
        self.massA = mass
        self.tagA = tag
        self.posA = pos
        self.velA = vel
        self.accA = acc

    @property
    def nparticles(self):
        return self.N

    def output(self, filename):
        with open(filename, 'w') as f:
            f.write(f"{self.N}\n")
            for i in range(self.N):
                f.write(f"{self.massA[i]:.6e} {self.tagA[i]} ")
                f.write(f"{self.posA[i,0]:.6e} {self.posA[i,1]:.6e} {self.posA[i,2]:.6e} ")
                f.write(f"{self.velA[i,0]:.6e} {self.velA[i,1]:.6e} {self.velA[i,2]:.6e} ")
                f.write(f"{self.accA[i,0]:.6e} {self.accA[i,1]:.6e} {self.accA[i,2]:.6e}\n")

    def draw(self, dim=2):
        fig = plt.figure()

        if dim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(self.posA[:,0], self.posA[:,1], s=1)
            ax.set_xlabel('X [code unit]')
            ax.set_ylabel('Y [code unit]')
        elif dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.posA[:,0], self.posA[:,1], self.posA[:,2], s=1)
            ax.set_xlabel('X [code unit]')
            ax.set_ylabel('Y [code unit]')
            ax.set_zlabel('Z [code unit]')
        else:
            raise ValueError("Invalid dimension")

        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        return fig, ax
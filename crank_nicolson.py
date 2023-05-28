import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Callable
from tqdm import tqdm

from utils import DEFAULT_INITIAL, DEFAULT_BOUNDARY, \
    Boundary, THERMAL_DICT


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNSolver:

    def __init__(
            self,
            initial: np.ndarray|Callable[[float, float], float] \
                = DEFAULT_INITIAL,
            boundaries: Boundary = DEFAULT_BOUNDARY,
            dx: float = 1., dt: float = 0.01, a: float | str = "iron",
            nx: int = 128, ny: int = 512
    ) -> None:
        
        if isinstance(initial, np.ndarray):
            self.u = torch.from_numpy(initial).to(device)
            self.ny, self.nx = initial.shape
        else:
            self.u = torch.zeros((ny, nx))
            for j, row in enumerate(self.u):
                for i, _ in enumerate(row):
                    self.u[j, i] = initial((i-60)*dx, (j-220)*dx)
            self.ny, self.nx = ny, nx
        self.bc = boundaries
        if isinstance(a, str):
            a: float = THERMAL_DICT.get(a, 20.)
        self.c: float = a*dt/dx**2
        self.dt = dt
        self.dx = dx
        self.up5 = torch.zeros_like(self.u).to(device)
        lhs1 = torch.diag_embed((1+self.c) * torch.ones(self.nx-2)).to(device) \
            - torch.diag_embed(self.c/2 * torch.ones(self.nx-3), offset=1).to(device) \
            - torch.diag_embed(self.c/2 * torch.ones(self.nx-3), offset=-1).to(device)
        self.lhs1_inv = torch.linalg.inv(lhs1)
        lhs2 = torch.diag_embed((1+self.c) * torch.ones(self.ny-2)).to(device) \
            - torch.diag_embed(self.c/2 * torch.ones(self.ny-3), offset=1).to(device) \
            - torch.diag_embed(self.c/2 * torch.ones(self.ny-3), offset=-1).to(device)
        self.lhs2_inv = torch.linalg.inv(lhs2)
        self.apply_bc(self.u)
        self.frames =[self.u.detach().cpu().numpy()]

    
    def plot(self, k):
        u = self.u.detach().cpu()
        plt.imshow(u, cmap="coolwarm")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.title(f"t = {self.dt*k} s")
        plt.colorbar()
        plt.show()
    

    def apply_bc(self, field):

        if self.bc.up.type == "value":
            field[0, :] = self.bc.up.expr
        elif self.bc.up.type == "neumann":
            field[0, :] = field[1, :] + self.dx * self.bc.up.expr
        else:
            raise NotImplementedError
        if self.bc.down.type == "value":
            field[-1, :] = self.bc.down.expr
        elif self.bc.down.type == "neumann":
            field[-1, :] = field[-2, :] + self.dx * self.bc.down.expr
        else:
            raise NotImplementedError
        if self.bc.left.type == "value":
            field[:, 0] = self.bc.left.expr
        elif self.bc.left.type == "neumann":
            field[:, 0] = field[:, 1] + self.dx * self.bc.left.expr
        else:
            raise NotImplementedError
        if self.bc.right.type == "value":
            field[:, -1] = self.bc.right.expr
        elif self.bc.right.type == "neumann":
            field[:, -1] = field[:, -2] + self.dx * self.bc.right.expr
        else:
            raise NotImplementedError
    
    
    def halfstep_one(self):
        """
        Iterate over y-direction, solve equation in x-direction
        """

        for j in range(1, self.ny-1):
            rhs = self.c/2*self.u[j-1, 1:-1] + self.c/2*self.u[j+1, 1:-1] \
                + (1-self.c)*self.u[j, 1:-1]
            rhs = rhs.float().to(device)
            # vec = torch.linalg.solve(self.lhs1, rhs)
            self.up5[j, 1:-1] = self.lhs1_inv @ rhs
        
        self.apply_bc(self.up5)
        
    
    def halfstep_two(self):
        """
        Iterate over x-direction, solve equation in y-direction
        """

        for i in range(1, self.nx-1):
            rhs = self.c/2 * self.up5[1:-1, i-1] \
                + self.c/2 * self.up5[1:-1, i+1] \
                + (1-self.c) * self.up5[1:-1, i]
            rhs = rhs.float().to(device)
            # vec = torch.linalg.solve(self.lhs2, rhs)
            self.u[1:-1, i] = self.lhs2_inv @ rhs
        
        self.apply_bc(self.u)

    
    def timestep(self):
        self.halfstep_one()
        self.halfstep_two()


    def animation(
            self, tf: float = 30.0, interval: int = 25,
            fps: int = 10, gif_title: str = None
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(self.frames[0], cmap="coolwarm")
        cb = fig.colorbar(im)
        cb.set_label("Temperature (normalized)")
        ax.set_xlabel("x/dx [mm]")
        ax.set_ylabel("y/dx [mm]")
        title = ax.set_title(f"t = 0.00 s")
        heat_list = []

        def animate(i):
            arr = self.frames[i]
            im.set_data(arr)
            title.set_text(f"t = {i*interval*self.dt:.2f} s")
            return [im]

        for k in tqdm(range(1, int(tf/self.dt)+1)):
            self.timestep()
            if k % interval == 0:
                arr = self.u.cpu().numpy()
                self.frames.append(arr)
                heat_list.append(np.sum(arr))
        
        anim = FuncAnimation(fig, func=animate, frames=len(self.frames), blit=True)
        if gif_title is None:
            gif_title = "anim"
        anim.save(gif_title+".gif", writer=PillowWriter(fps=fps))

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(range(len(heat_list)), heat_list)
        plt.show()
    

if __name__ == '__main__':
    cn = CNSolver()
    cn.animation(tf=3)
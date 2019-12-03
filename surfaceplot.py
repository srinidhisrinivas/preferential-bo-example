import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sys
import math
from scipy.optimize import minimize

def levy(x):
    if not isinstance(x, torch.Tensor): x = torch.tensor(x).float();	
    if len(x.shape) < 2: x = x.unsqueeze(0);
    assert x.shape[1] == 2, 'Input should be of dimension 2'

    w = lambda x: 1. + (x - 1.)/4.; 
    result = torch.sin(math.pi * w(x[:,0])) ** 2;
    result += torch.sum(((w(x[:,:-1])-1)**2) * (1. + 10*torch.sin(math.pi * w(x[:,:-1]) + 1.) ** 2), dim=1);
    result += ((w(x[:,-1])-1)**2) * (1. + torch.sin(2. * math.pi * w(x[:,-1])) ** 2)

    return result;

def dixon_price(x):
    if not isinstance(x, torch.Tensor): x = torch.tensor(x).float();    
    if len(x.shape) < 2: x = x.unsqueeze(0);
    assert x.shape[1] == 2, 'Input should be of dimension 2'

    result = (x[:,0] - 1.) ** 2;
    result += 2 * ((2 * (x[:,1] ** 2) - x[:,0]) ** 2);
    return result;

def create_hypercube_grid(bounds, num_points, n_dims):
    axpoints = torch.linspace(bounds[0], bounds[1], num_points);
    x = torch.zeros(num_points ** n_dims, n_dims);
    #print(axpoints.unsqueeze(1).repeat(num_points ** (n_dims - 0 - 1), num_points ** (0 + 1)).view(-1));
    for i in range(n_dims):
        x[:,i].copy_(axpoints.unsqueeze(1).repeat(num_points ** (n_dims - 1 - i), num_points ** i).view(-1))
    return torch.flip(x, dims=(1,));

f = dixon_price;

fig = plt.figure()
ax = fig.gca(projection='3d')
bounds = (-10.,10.)
# Make data.
X = torch.linspace(-10, 10, 200).unsqueeze(1).repeat(1,200);
Y = torch.linspace(-10, 10, 200).repeat(200,1);

grid = create_hypercube_grid((-10, 10), 200, 2)

Z = f(grid).reshape(200,200);

num_repeats = 10000;
min_y = float("inf");
min_x = float("inf");

init_guesses = torch.distributions.uniform.Uniform(torch.tensor([bounds[0]]), torch.tensor([bounds[1]])).sample(torch.Size([num_repeats,2]))
for x0 in init_guesses:
    res = minimize(levy, x0.numpy(), bounds=2*[bounds]);
    y = levy(res.x)
    if y < min_y:
        min_y = y;
        min_x = res.x;

print(min_y, min_x)

# Plot the surface.
surf = ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='jet',
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 100000)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
import torch
from Systems import Rossler
import numpy as np
from NN import NN
import matplotlib.pyplot as plt
from Observer import Observer
import pandas as pd

plt.rcParams['font.size'] = 15
plt.rcParams["legend.labelspacing"] = 0.2
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.serif'] = 'Times New Roman'

legend_fontsize = 13
fig_dim = (10,6)
title_ypos = 1.25
legend_ypos = 0.8
legend_xpos = 0.861

#------------- Load model for Rossler and initialize observer ------------- 
a0 = 0.2
b0 = 0.2
c0 = 5.7
limits = np.array([[-1,1], [-1,1], [-1,1]])    # Sample space

# A and B matricies 
M = np.array([[-2.12116939,  2.73877907, -0.75338041,  3.13947511,  1.00581224,
        -5.34548426, -5.34544528],
    [-5.91811463, -1.99160105, -0.26953991,  0.45654273, -0.56750459,
        -3.79023002, -3.00480715],
    [-1.27814843, -2.08375366, -2.60781906, -4.11186517,  0.33087248,
        0.08191241, -2.81998688],
    [ 0.69080313,  2.00687664,  3.02478991, -3.53231915, -0.89704804,
        -5.19153027,  3.16796666],
    [-1.16566857, -0.53979687, -0.30300461, -3.79753229, -2.7939018 ,
        -0.30176078, -2.61480264],
    [ 2.97899741,  3.97265572, -2.04951338, -1.08716448, -0.98741222,
        -4.7047881 , -1.38218644],
    [ 0.95887295,  0.43056828, -1.39564903, -3.18608794,  3.74388666,
        -3.88509855, -3.07468653]])

K = np.ones([7,1])

rossler = Rossler(a0, b0, c0, noise_measurement_std=0.1, noise_std=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 1000
a = 0
b = 50

model_data = torch.load('models/rossler_T_inverse_lr0.001_batch32_epoch15', map_location=device)
num_hidden = model_data['config']['num_hidden']
hidden_size = model_data['config']['hidden_size']
x_size = model_data['config']['x_size']
z_size = model_data['config']['z_size']
activation = model_data['config']['activation']
normalizer = model_data['config']['normalizer']
T_inv_net = NN(num_hidden, hidden_size, z_size, x_size, activation, normalizer=normalizer).to(device)
T_inv_net.load_state_dict(model_data['model'])
T_inv_net.eval()

observer = Observer(rossler, M, K, T_inv_net, a, b, N, init_z_zero=True)

# -------------------------------------------------------------------------------------------

# -------------------------- Simulate and plot the trajectories -------------------------
ic = np.array([[0.5, 0.5, 0.5]])

x, x_hat, t, error = observer.simulate(ic, add_noise = True)
fig = plt.figure(figsize = fig_dim, facecolor='w')

# -------------------------- Save Data -------------------------- 
df = pd.DataFrame(x[:,0])
df.to_excel("rossler_x1.xlsx", index=False, header=False)

df = pd.DataFrame(x[:,1])
df.to_excel("rossler_x2.xlsx", index=False, header=False)

df = pd.DataFrame(x[:,2])
df.to_excel("rossler_x3.xlsx", index=False, header=False)

df = pd.DataFrame(x_hat[:,0])
df.to_excel("rossler_x1_hat.xlsx", index=False, header=False)

df = pd.DataFrame(x_hat[:,1])
df.to_excel("rossler_x2_hat.xlsx", index=False, header=False)

df = pd.DataFrame(x_hat[:,2])
df.to_excel("rossler_x3_hat.xlsx", index=False, header=False)

# ------------------------------------------------------------------

# Plot x1
plt.subplot(3, 1, 1)
plt.plot(t, x[:,0], label = r'$x_1$', color = 'black')
plt.plot(t, x_hat[:,0], label = r'$\hat{x}_1$', color = 'magenta', linestyle='--')
plt.title('Rossler')
plt.xlim((0,50))
plt.ylim((-15,15))
plt.grid(linestyle = '--')
plt.tick_params(axis='x', labelbottom=False)  # Hide x-axis tick labels but keep ticks
plt.yticks(np.arange(-15, 15+15, 15))
plt.legend(prop = {'size':legend_fontsize}, loc = 'upper left')

# Plot x2
plt.subplot(3, 1, 2)
plt.plot(t, x[:,1], label = r'$x_2$', color = 'black')
plt.plot(t, x_hat[:,1], label = r'$\hat{x}_2$', color = 'magenta', linestyle='--')
plt.xlim((0,50))
plt.ylim((-15,15))
plt.tick_params(axis='x', labelbottom=False)  # Hide x-axis tick labels but keep ticks
plt.yticks(np.arange(-15, 15+15, 15))
plt.grid(True, linestyle = '--', which='both')
plt.legend(prop = {'size':legend_fontsize}, loc = 'best')

# Plot x3
plt.subplot(3, 1, 3)
plt.plot(t, x[:,2], label = r'$x_3$', color = 'black')
plt.plot(t, x_hat[:,2], label = r'$\hat{x}_3$', color = 'magenta', linestyle='--')
plt.xlim((0,50))
plt.ylim((-5,30))
plt.grid(linestyle = '--')
plt.xticks(np.arange(0, 50+10, 10))
plt.yticks(np.arange(0, 30+10, 10))
plt.xlabel('Time')
plt.legend(prop = {'size':legend_fontsize}, loc = 'best')

plt.savefig(f'figures/rossler/rossler.png', dpi=700, transparent = False, bbox_inches = 'tight', facecolor = 'w')


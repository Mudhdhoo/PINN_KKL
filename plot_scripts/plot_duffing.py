import torch
from Systems import RevDuff
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#------------- Load model for Duffing Oscillator and initialize observer ------------- 
M5_diag = np.diag([-1,-2,-3,-4,-5])    # Choose M matrix
K5_diag = np.array([[1,1,1,1,1]]).T    # Choose K matrix
revduff = RevDuff(5, noise_measurement_std=0.1, noise_std=0.0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 1000
a = 0
b = 50

model_data = torch.load('models/duffing_T_inverse_lr0.001_batch32_epoch15', map_location=device)
num_hidden = model_data['config']['num_hidden']
hidden_size = model_data['config']['hidden_size']
x_size = model_data['config']['x_size']
z_size = model_data['config']['z_size']
activation = model_data['config']['activation']
normalizer = model_data['config']['normalizer']
T_inv_net = NN(num_hidden, hidden_size, z_size, x_size, activation, normalizer=normalizer).to(device)
T_inv_net.load_state_dict(model_data['model'])
T_inv_net.eval()

observer = Observer(revduff, M5_diag, K5_diag, T_inv_net, a, b, N, init_z_zero=True)
# -------------------------------------------------------------------------------------------

# -------------------------- Simulate and plot the trajectories -------------------------
ic = np.array([[0.5, 0.5]])

x, x_hat, t, error = observer.simulate(ic, add_noise = True)
fig = plt.figure(figsize = fig_dim, facecolor='w')

# -------------------------- Save Data -------------------------- 
df = pd.DataFrame(x[:,0])
df.to_excel("data/duffing/duffing_x1.xlsx", index=False, header=False)

df = pd.DataFrame(x[:,1])
df.to_excel("data/duffing/duffing_x2.xlsx", index=False, header=False)

df = pd.DataFrame(x_hat[:,0])
df.to_excel("data/duffing/duffing_x1_hat.xlsx", index=False, header=False)

df = pd.DataFrame(x_hat[:,1])
df.to_excel("data/duffing/duffing_x2_hat.xlsx", index=False, header=False)
# ------------------------------------------------------------------

# Plot x1
plt.subplot(2, 1, 1)
plt.plot(t, x[:,0], label = r'$x_1$', color = 'black')
plt.plot(t, x_hat[:,0], label = r'$\hat{x}_1$', color = 'magenta', linestyle='--')
plt.title('Reverse Duffing Oscillator')
plt.xlim((0,50))
plt.ylim((-1,1))
plt.grid(linestyle = '--')
plt.tick_params(axis='x', labelbottom=False)  # Hide x-axis tick labels but keep ticks
plt.yticks(np.arange(-1, 1+1, 1))
plt.legend(prop = {'size':legend_fontsize}, loc = 'upper left')

# Plot x2
plt.subplot(2, 1, 2)
plt.plot(t, x[:,1], label = r'$x_2$', color = 'black')
plt.plot(t, x_hat[:,1], label = r'$\hat{x}_2$', color = 'magenta', linestyle='--')
plt.xlim((0,50))
plt.ylim((-1,1))
plt.xlabel('Time')
plt.xticks(np.arange(0, 50+10, 10))
plt.yticks(np.arange(-1, 1+1, 1))
plt.grid(True, linestyle = '--', which='both')
plt.legend(prop = {'size':legend_fontsize}, loc = 'upper left')
# -------------------------------------------------------------------------------------------

# Save figure
plt.savefig(f'figures/duffing/duffing.png', dpi=700, transparent = False, bbox_inches = 'tight', facecolor = 'w')


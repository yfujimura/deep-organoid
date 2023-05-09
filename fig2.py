import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize

days = [4.5,5,6,7,8]
cell = 64 
z = 6

cm = 1/2.54 
fig = plt.figure(figsize=(17*cm, 15*cm),dpi=100)
axes = fig.subplots(4,5)

for axis in axes.flatten():
    axis.set_axis_off()
    
for di, day in enumerate(days):
    img = Image.open("results/sample/img/{}_{}_{}.png".format(cell, day, z))
    axes[0][di].imshow(img, cmap="gray")
    
    gt = np.array(Image.open("results/sample/gt/{}_{}_{}.png".format(cell, day, z)))
    gt = gt.astype(np.float32) / 255.
    axes[1][di].imshow(gt)
    
    pred = np.array(Image.open("results/sample/pred/{}_{}_{}.png".format(cell, day, z)))
    pred = pred.astype(np.float32) / 255.
    axes[2][di].imshow(pred)
    
    error = np.abs(pred[:,:,1] - resize(gt, (256,256))[:,:,1])**2
    error_map = axes[3][di].imshow(error, cmap="jet", clim=(0,0.1)) 
    
    
plt.subplots_adjust(hspace=0.05, wspace=-0.5)
cbar = fig.colorbar(error_map, ax=axes[:4,:], shrink=0.3, pad=0.01, location="bottom")
cbar.set_label("Squared error", size=8)
cbar.ax.tick_params(labelsize=8)
plt.savefig("figures/fig2.png")


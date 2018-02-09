#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
# ------------------------------------------------------------------------------
# Patches
# ------------------------------------------------------------------------------
# verts = [
#        (0., 0.),  # left, bottom
#        (0., 1.),  # left, top
#        (1., 1.),  # right, top
#        (1., 0.),  # right, bottom
#        (0., 0.),  # ignored
#     ]
verts = [
       (2., 2.),  # left, bottom
       (2., 3.),  # left, top
       (3., 3.),  # right, top
       (3., 2.),  # right, bottom
       (2., 2.),  # ignored
    ]

codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

path = Path(verts, codes)
# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
patch = patches.PathPatch(path, facecolor='white', lw=1)
path = Path(verts, codes)
# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

ax.add_patch(patch)
ax.grid()
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.text(
    2.5*(1./8),
    (8 - 2.5)*(1./8),
    '0.35',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=18,
    color='red',
    transform=ax.transAxes
)


plt.show()

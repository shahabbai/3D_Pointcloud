from matplotlib.pyplot import figure, draw, pause, gca
from pathlib import Path
import numpy as np


def plotderiv(fx, fy, ft):

    fg = figure(figsize=(18, 5))
    ax = fg.subplots(1, 3)

    for f, a, t in zip((fx, fy, ft), ax, ('$f_x$', '$f_y$', '$f_t$')):
        h = a.imshow(f, cmap='bwr')
        a.set_title(t)
        fg.colorbar(h, ax=a)


def compareGraphs(u, v, Inew, scale: int = 3, quivstep: int = 5, fn: Path = None):
    """
    makes quiver
    """

    ax = figure().gca()
    ax.imshow(Inew, cmap='gray', origin='lower')
    # plt.scatter(POI[:,0,1],POI[:,0,0])
    for i in range(0, u.shape[0], quivstep):
        for j in range(0, v.shape[1], quivstep):
            ax.arrow(
                j,
                i,
                v[i, j] * scale,
                u[i, j] * scale,
                color='red',
                head_width=0.5,
                head_length=1,
            )

        # plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)
    if fn:
        ax.set_title(fn)

    draw()
    pause(0.01)


def compareGraphsLK(imgOld, imgNew, POI, V, scale=1.0, fn: Path = None):

    ax = gca()
    ax.imshow(imgNew, cmap='gray', origin='lower')
    # plt.scatter(POI[:,0,1],POI[:,0,0])
    for i in range(len(POI)):
        ax.arrow(
            POI[i, 0, 1], POI[i, 0, 0], V[i, 1] * scale, V[i, 0] * scale, color='red'
        )
    # plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)
    if fn:
        ax.set_title(fn)

    draw()
    pause(0.5)


def disparity(u, v, Iold, Inew, scale: int = 3, quivstep: int = 1):

    xy = np.zeros([2, Inew.shape[0]*Inew.shape[1]])
    dis = 0
    '''
     kp2 = [None] * (Inew.shape[0] * Inew.shape[1])

     for make in range(Inew.shape[0] * Inew.shape[1]):
        kp2[make] = temp

     '''
    it1 = 0
    it2 = 0
# x points
    for i in range(0, u.shape[0], quivstep):
        for j in range(0, v.shape[1], quivstep):
            xy[0, it1] = abs(i - v[i, j])
            it1 += 1

    for i in range(0, u.shape[0], quivstep):
        for j in range(0, v.shape[1], quivstep):
            xy[1, it2] = j - u[i, j]
            it2 += 1
    temp = 0
    disparity_img = np.zeros([Inew.shape[0], Inew.shape[1]], dtype=np.float)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            dis = i-xy[0, temp]
            disparity_img[i, j] = dis
            temp += 1

    return disparity_img
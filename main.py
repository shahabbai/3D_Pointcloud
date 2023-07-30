#!/usr/bin/env python

import imageio.v2 as imageio
from pathlib import Path
from matplotlib.pyplot import show
from argparse import ArgumentParser
import numpy as np
from opt import HornSchunck, getimgfiles
from opt.plots import compareGraphs
from opt.plots import disparity
import cv2 as cv
import open3d as o3d
import time

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

FILTER = 7

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

def main():
    p = ArgumentParser(description='3D Point Cloud')
    p.add_argument('stem', help='path/stem of files to analyze')
    p.add_argument('pat', help='glob pattern of files', default='*.bmp')
    p.add_argument('-p', '--plot', help='show plots', action='store_true')
    p.add_argument('-a', '--alpha', help='regularization parameter', type=float, default=0.001)
    p.add_argument('-N', help='number of iterations', type=int, default=10)
    p = p.parse_args()

    U, V = horn_schunck(p.stem, p.pat, alpha=p.alpha, Niter=p.N, verbose=p.plot)

    show()


def horn_schunck(stem: Path, pat: str, alpha: float, Niter: int, verbose: bool):
    start = time.time()
    flist = getimgfiles(stem, pat)
    # qmat = np.float32([[2, 0, 0, 0],
    #                    [0, 2, 0, 0],
    #                    [0, 0, 25 * 0.05, 0],
    #                    [0, 0, 0, 2]])
    # Baseline qmat 
    qmat = np.eye(4)
    # Translate
    qmat[:3,3] = [0, 0, 2]

    # Rotate
    angle = np.pi / 4 
    rotation_mat = [[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]]
    qmat[:2,:2] = rotation_mat

    # Scale 
    qmat *= 2

    for i in range(len(flist) - 1):
        fn1 = flist[i]
        im1 = imageio.imread(fn1, as_gray=True)

        fn2 = flist[i + 1]
        im2 = imageio.imread(fn2, as_gray=True)
        rgb1 = imageio.imread(fn1, as_gray=False)
        rgb2 = imageio.imread(fn2, as_gray=False)
        
        if rgb1.shape[2] == 4:
            rgb1 = rgb1[:,:,:3]
        U, V = HornSchunck(im1, im2, alpha=alpha, Niter=Niter, verbose=verbose)
        new_uv = U+V
        norm_image = cv.normalize(new_uv, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        norm_image = norm_image.astype(np.uint8)
        #cv.imshow('disparity_norm', norm_image)
        hsv = np.zeros_like(rgb1)
        # Sets image saturation to maximum
        hsv[..., 1] = 255
        magnitude, angle = cv.cartToPolar(U, V)
        # Sets image hue according to the optical flow direction
        hsv[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        points = cv.reprojectImageTo3D(norm_image, qmat)
        colors = cv.cvtColor(np.asarray(im1), cv.COLOR_GRAY2RGB)
        mask = norm_image > norm_image.min()
        out_points = points[mask]
        out_colors = colors[mask]
        
        write_ply('out.ply', out_points, out_colors)
        print('total time={0}'.format(time.time()-start))
        pcd = o3d.io.read_point_cloud("out.ply")
        o3d.visualization.draw_geometries([pcd])

        #cv.imshow('disparity', bgr)
        cv.waitKey(0)
        cv.destroyAllWindows()
        #cv.imwrite(f'./hs_results/optical_hs_{1}.png', bgr)

    return U, V


if __name__ == '__main__':
    main()
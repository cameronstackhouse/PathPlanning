# this is the three dimensional space
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yue qi
"""
import numpy as np

# from utils3D import OBB2AABB


def R_matrix(z_angle, y_angle, x_angle):
    # s angle: row; y angle: pitch; z angle: yaw
    # generate rotation matrix in SO3
    # RzRyRx = R, ZYX intrinsic rotation
    # also (r1,r2,r3) in R3*3 in {W} frame
    # used in obb.O
    # [[R p]
    # [0T 1]] gives transformation from body to world
    return (
        np.array(
            [
                [np.cos(z_angle), -np.sin(z_angle), 0.0],
                [np.sin(z_angle), np.cos(z_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        @ np.array(
            [
                [np.cos(y_angle), 0.0, np.sin(y_angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(y_angle), 0.0, np.cos(y_angle)],
            ]
        )
        @ np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(x_angle), -np.sin(x_angle)],
                [0.0, np.sin(x_angle), np.cos(x_angle)],
            ]
        )
    )


def getblocks():
    # AABBs
    block = [
        [4.00e00, 1.20e01, 0.00e00, 5.00e00, 2.00e01, 5.00e00],
        [5.5e00, 1.20e01, 0.00e00, 1.00e01, 1.30e01, 5.00e00],
        [1.00e01, 1.20e01, 0.00e00, 1.40e01, 1.30e01, 5.00e00],
        [1.00e01, 9.00e00, 0.00e00, 2.00e01, 1.00e01, 5.00e00],
        [9.00e00, 6.00e00, 0.00e00, 1.00e01, 1.00e01, 5.00e00],
    ]
    Obstacles = []
    for i in block:
        i = np.array(i)
        Obstacles.append([j for j in i])
    return np.array(Obstacles)


def getballs():
    spheres = [[2.0, 6.0, 2.5, 1.0], [14.0, 14.0, 2.5, 2]]
    Obstacles = []
    for i in spheres:
        Obstacles.append([j for j in i])
    return np.array(Obstacles)


def getAABB(blocks):
    # used for Pyrr package for detecting collision
    AABB = []
    for i in blocks:
        AABB.append(
            np.array([np.add(i[0:3], -0), np.add(i[3:6], 0)])
        )  # make AABBs alittle bit of larger
    return AABB


def getAABB2(blocks):
    # used in lineAABB
    AABB = []
    for i in blocks:
        AABB.append(aabb(i))
    return AABB


def add_block(block=[1.51e01, 0.00e00, 2.10e00, 1.59e01, 5.00e00, 6.00e00]):
    return block


class aabb(object):
    # make AABB out of blocks,
    # P: center point
    # E: extents
    # O: Rotation matrix in SO(3), in {w}
    def __init__(self, AABB):
        self.P = [
            (AABB[3] + AABB[0]) / 2,
            (AABB[4] + AABB[1]) / 2,
            (AABB[5] + AABB[2]) / 2,
        ]  # center point
        self.E = [
            (AABB[3] - AABB[0]) / 2,
            (AABB[4] - AABB[1]) / 2,
            (AABB[5] - AABB[2]) / 2,
        ]  # extents
        self.O = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def contains_point(self, point):
        min_x, min_y, min_z = (
            self.P[0] - self.E[0],
            self.P[1] - self.E[1],
            self.P[2] - self.E[2],
        )
        max_x, max_y, max_z = (
            self.P[0] + self.E[0],
            self.P[1] + self.E[1],
            self.P[2] + self.E[2],
        )

        return (
            min_x <= point[0] <= max_x
            and min_y <= point[1] <= max_y
            and min_z <= point[2] <= max_z
        )


class obb(object):
    # P: center point
    # E: extents
    # O: Rotation matrix in SO(3), in {w}
    def __init__(self, P, E, O):
        self.P = P
        self.E = E
        self.O = O
        self.T = np.vstack(
            [np.column_stack([self.O.T, -self.O.T @ self.P]), [0, 0, 0, 1]]
        )


class env:
    def __init__(self, xmin=0, ymin=0, zmin=0, xmax=20, ymax=20, zmax=5, resolution=1):
        # def __init__(self, xmin=-5, ymin=0, zmin=-5, xmax=10, ymax=5, zmax=10, resolution=1):
        self.x_range = xmax
        self.y_range = ymax
        self.z_range = zmax
        self.resolution = resolution
        self.boundary = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
        self.blocks = getblocks()
        self.AABB = getAABB2(self.blocks)
        self.AABB_pyrr = getAABB(self.blocks)
        self.balls = getballs()
        self.OBB = np.array(
            [
                obb([5.0, 7.0, 2.5], [0.5, 2.0, 2.5], R_matrix(135, 0, 0)),
                obb([12.0, 4.0, 2.5], [0.5, 2.0, 2.5], R_matrix(45, 0, 0)),
            ]
        )
        self.start = np.array([2.0, 2.0, 2.0])
        self.goal = np.array([6.0, 16.0, 0.0])
        self.t = 0  # time

    def new_block_corners(self, corners):
        self.blocks = np.vstack([self.blocks, corners])
        self.AABB = getAABB2(self.blocks)
        self.AABB_pyrr = getAABB(self.blocks)

    def New_block(self):
        newblock = add_block()
        self.blocks = np.vstack([self.blocks, newblock])
        self.AABB = getAABB2(self.blocks)
        self.AABB_pyrr = getAABB(self.blocks)

    def move_start(self, x):
        self.start = x

    def move_block(
        self, a=[0, 0, 0], s=0, v=[0.1, 0, 0], block_to_move=0, mode="translation"
    ):
        # t is time , v is velocity in R3, a is acceleration in R3, s is increment ini time,
        # R is an orthorgonal transform in R3*3, is the rotation matrix
        # (s',t') = (s + tv, t) is uniform transformation
        # (s',t') = (s + a, t + s) is a translation
        if mode == "translation":
            ori = np.array(self.blocks[block_to_move])
            self.blocks[block_to_move] = np.array(
                [
                    ori[0] + a[0],
                    ori[1] + a[1],
                    ori[2] + a[2],
                    ori[3] + a[0],
                    ori[4] + a[1],
                    ori[5] + a[2],
                ]
            )

            self.AABB[block_to_move].P = [
                self.AABB[block_to_move].P[0] + a[0],
                self.AABB[block_to_move].P[1] + a[1],
                self.AABB[block_to_move].P[2] + a[2],
            ]
            self.t += s
            # return a range of block that the block might moved
            a = self.blocks[block_to_move]
            return np.array(
                [
                    a[0] - self.resolution,
                    a[1] - self.resolution,
                    a[2] - self.resolution,
                    a[3] + self.resolution,
                    a[4] + self.resolution,
                    a[5] + self.resolution,
                ]
            ), np.array(
                [
                    ori[0] - self.resolution,
                    ori[1] - self.resolution,
                    ori[2] - self.resolution,
                    ori[3] + self.resolution,
                    ori[4] + self.resolution,
                    ori[5] + self.resolution,
                ]
            )
            # return a,ori
        # (s',t') = (Rx, t)

    def move_OBB(self, obb_to_move=0, theta=[0, 0, 0], translation=[0, 0, 0]):
        # theta stands for rotational angles around three principle axis in world frame
        # translation stands for translation in the world frame
        ori = [self.OBB[obb_to_move]]
        self.OBB[obb_to_move].P = [
            self.OBB[obb_to_move].P[0] + translation[0],
            self.OBB[obb_to_move].P[1] + translation[1],
            self.OBB[obb_to_move].P[2] + translation[2],
        ]
        # Calculate orientation
        self.OBB[obb_to_move].O = R_matrix(
            z_angle=theta[0], y_angle=theta[1], x_angle=theta[2]
        )
        # generating transformation matrix
        self.OBB[obb_to_move].T = np.vstack(
            [
                np.column_stack(
                    [
                        self.OBB[obb_to_move].O.T,
                        -self.OBB[obb_to_move].O.T @ self.OBB[obb_to_move].P,
                    ]
                ),
                [translation[0], translation[1], translation[2], 1],
            ]
        )
        return self.OBB[obb_to_move], ori[0]


class CustomEnv(env):
    def __init__(
        self, data, xmin=0, ymin=0, zmin=0, xmax=100, ymax=100, zmax=100, resolution=1
    ):
        super().__init__(xmin, ymin, zmin, xmax, ymax, zmax, resolution)
        self.data = data
        self.gen_obs()
        self.covered_blocks = set()

    def gen_obs(self):
        grid = self.data["grid"]

        self.start = np.array(self.data["agent"])
        self.goal = np.array(self.data["goal"])

        depths = len(grid)

        rows = len(grid[0])
        cols = len(grid[0][0])
        visited = np.zeros((depths, rows, cols), dtype=bool)
        cuboids = []

        for z in range(depths):
            for y in range(rows):
                for x in range(cols):
                    if grid[z][y][x] == 1 and not visited[z, y, x]:
                        # Find the extent of the cuboid
                        rect_x, rect_y, rect_z = x, y, z
                        while rect_x < cols and grid[z][y][rect_x] == 1:
                            rect_x += 1
                        while rect_y < rows and all(
                            grid[z][rect_y][i] == 1 for i in range(x, rect_x)
                        ):
                            rect_y += 1
                        while rect_z < depths and all(
                            grid[rect_z][j][i] == 1
                            for j in range(y, rect_y)
                            for i in range(x, rect_x)
                        ):
                            rect_z += 1

                        # xmin, ymin, zmin, xmax, ymax, zmax
                        cuboid = [x, y, z, rect_x, rect_y, rect_z]
                        cuboids.append(cuboid)

                        # Mark the found cuboid as visited
                        visited[z:rect_z, y:rect_y, x:rect_x] = True

        Obstacles = []
        for i in cuboids:
            Obstacles.append(np.array(i))

        self.balls = []
        self.blocks = np.array(Obstacles)

        self.AABB = getAABB2(self.blocks)
        self.AABB_pyrr = getAABB(self.blocks)


if __name__ == "__main__":
    newenv = env()

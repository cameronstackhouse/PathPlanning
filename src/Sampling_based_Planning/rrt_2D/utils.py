"""
utils for collision check
@author: huiming zhou
"""

import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy.polynomial import polynomial

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D import env
from rrt_2D.rrt import Node


class Utils:
    def __init__(self):
        self.env = env.Env()

        self.delta = 0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def update_obs(self, obs_cir, obs_bound, obs_rec):
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec

    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for ox, oy, w, h in self.obs_rectangle:
            vertex_list = [
                [ox - delta, oy - delta],
                [ox + w + delta, oy - delta],
                [ox + w + delta, oy + h + delta],
                [ox - delta, oy + h + delta],
            ]
            obs_list.append(vertex_list)

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False

    def is_intersect_circle(self, o, d, a, r):
        d2 = np.dot(d, d)
        delta = self.delta

        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r + delta:
                return True

        return False

    def is_collision(self, start, end):
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True

        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        for v1, v2, v3, v4 in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        for x, y, r in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y], r):
                return True

        return False

    def is_inside_obs(self, node):
        delta = self.delta

        for x, y, r in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        for x, y, w, h in self.obs_rectangle:
            if (
                0 <= node.x - (x - delta) <= w + 2 * delta
                and 0 <= node.y - (y - delta) <= h + 2 * delta
            ):
                return True

        for x, y, w, h in self.obs_boundary:
            if (
                0 <= node.x - (x - delta) <= w + 2 * delta
                and 0 <= node.y - (y - delta) <= h + 2 * delta
            ):
                return True

        return False

    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)

    @staticmethod
    def euclidian_distance(p_1, p_2):
        return math.sqrt((p_2[0] - p_1[0]) ** 2 + (p_2[1] - p_1[1]) ** 2)

    @staticmethod
    def path_cost(path):
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += Utils.euclidian_distance(path[i], path[i + 1])
        return total_distance

    @staticmethod
    def calculate_turn_angle(p_1, p_2, p_3):
        v1 = np.array(p_1) - np.array(p_2)
        v2 = np.array(p_3) - np.array(p_2)

        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)

        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)

        return angle

    @staticmethod
    def turn_energy(turn_power, degrees, speed):
        """
        Calculates the energy consumption of turning given
        the power required to turn per second, the number of degrees to turn,
        and the speed of turning of the UAV.
        """
        return turn_power * (degrees / speed)

    @staticmethod
    def path_energy(path) -> float:
        """
        Calculates the energy used to traverse a path based on a UAV energy model and values
        calculated by ding et al.

        https://www.researchgate.net/publication/331264690_Energy-Efficient_Min-Max_Planning_of_Heterogeneous_Tasks_with_Multiple_UAVs?enrichId=rgreq-fa09167b1c97a83f7ec66c8af7c0e062-XXX&enrichSource=Y292ZXJQYWdlOzMzMTI2NDY5MDtBUzo5NzEyMDAzNzMyMjM0MjVAMTYwODU2MzYyMTUzNA%3D%3D&el=1_x_3
        """
        x = np.array([0, 2, 4, 6])
        # Power required (W) for each at 0, 2, 4, 6 m/s
        P_acc = np.array([242, 235, 239, 249])
        P_dec = np.array([245, 232, 230, 239])
        P_v = [242, 245, 246, 268]
        TURN_POWER = 260
        TURN_SPEED = 2.07

        # Fits curve and defines integrand
        cubic_coeffs_acc = np.polyfit(x, P_acc, 3)
        cubic_poly_acc = np.poly1d(cubic_coeffs_acc)

        cubic_coeffs_dec = np.polyfit(x, P_dec, 3)
        cubic_poly_dec = np.poly1d(cubic_coeffs_dec)

        def integrand_acc(x):
            return cubic_poly_acc(x)

        def integrand_dec(x):
            return cubic_poly_dec(x)

        fixed_speed = 6
        total_energy = 0

        for i in range(1, len(path)):
            p_1 = path[i - 1]
            p_2 = path[i]

            distance = Utils.euclidian_distance(p_1, p_2)

            accel_time = 2
            decel_time = 2

            # Too short to reach full speed
            if distance <= fixed_speed * (accel_time + decel_time):
                time_accel = np.sqrt(distance / (0.5 * fixed_speed))
                time_decel = time_accel
                time_cruise = 0
            else:
                time_accel = accel_time
                time_decel = decel_time
                time_cruise = (
                    distance - (fixed_speed * (accel_time + decel_time))
                ) / fixed_speed

            # Energy for acceleration
            energy_accel, _ = quad(
                integrand_acc, 0, fixed_speed * (time_accel / accel_time)
            )
            energy_accel *= time_accel / accel_time

            # Energy for constant speed
            energy_cruise = time_cruise * np.interp(fixed_speed, x, P_v)

            # Energy for deceleration
            energy_decel, _ = quad(
                integrand_dec, 0, fixed_speed * (time_decel / decel_time)
            )
            energy_decel *= time_decel / decel_time

            total_energy += energy_accel + energy_cruise + energy_decel

            print(total_energy)
            if i < len(path) - 1:
                p_3 = path[i + 1]
                angle = Utils.calculate_turn_angle(p_1, p_2, p_3)
                total_energy += Utils.turn_energy(TURN_POWER, angle, TURN_SPEED)

        return total_energy

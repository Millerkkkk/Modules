# =========================
# 能耗模型
# =========================
import numpy as np


class EnergyModel:
    def __init__(self, phi_motor=1.184692, phi_battery=1.112434, g=9.8, theta_ij = 0,
                 C_r=0.012, L=3000, R=0.7, A=3.8, rho=1.2041):
        """
        初始化能耗模型参数
        :param phi_motor: 驱动电机输出效率系数 (φ^d)
        :param phi_battery: 电池输出效率系数 (φ^a)
        :param g: 重力加速度 (m/s^2)
        :param theta_ij: 路段坡度 (弧度)
        :param C_r: 滚动阻力系数
        :param L: 车辆自重 (kg)
        :param R: 空气阻力系数 (Cd)
        :param A: 车辆正面迎风面积 (m^2)
        :param rho: 空气密度 (kg/m^3)
        """
        self.phi_motor = phi_motor
        self.phi_battery = phi_battery
        self.g = g
        self.theta_ij = theta_ij
        self.C_r = C_r
        self.L = L
        self.R = R
        self.A = A
        self.rho = rho

    def calculate_energy_consumption(self, u_ijk, v_ijk_R, t_ijk_R):
        """
        计算电动车在路段 (i,j) 上的能量消耗
        Args:
            u_ijk (float): 电动车在路段 (i,j) 上的实时载荷, kg
            v_ijk_R (float): 电动车在路段 (i,j) 上的行驶速度, km/h
            t_ijk_R (float): 电动车在路段 (i,j) 上的行驶时间, h
        Returns:
            float: 消耗的能量
        """
        # The rolling and air resistance components of energy consumption 滚动阻力 + 坡度阻力
        resistance_energy = (self.g * np.sin(self.theta_ij) + self.C_r * self.g * np.cos(self.theta_ij)) * (self.L + u_ijk) / 3600.0
        # The aerodynamic drag component of energy consumption 空气阻力
        aerodynamic_drag_energy = (self.R * self.A * self.rho * (v_ijk_R ** 2)) / 76140.0
        # Total energy consumption for the EV 总能耗
        e_ijk_R = self.phi_motor * self.phi_battery * (resistance_energy + aerodynamic_drag_energy) * v_ijk_R * t_ijk_R
        return e_ijk_R
    
    def calculate_one_node_energy_time(self, distance, departure_time, u_ijk):
        max_minutes = 1440  # 一天 = 1440 分钟
        period_length = 60  # 每段 60 分钟（1 小时）
        total_energy = 0.0
        total_travel_time = 0.0
        remaining_distance = distance
        travel_speed = 0

        while remaining_distance > 0:
            # Converting time to 24-hour format
            current_time = departure_time % max_minutes
            period = int(current_time // period_length)
            period_end = (period + 1) * period_length

            current_hour = (departure_time / 60) % 24
            if 0 <= current_hour <= 2 or 10 <= current_hour <= 12:
                travel_speed = 30
            else:
                travel_speed = 60

            # Calculates the travel time in the current time period
            time_available = period_end - current_time
            max_travel_time = remaining_distance / travel_speed * 60.0  # → 以分钟计算
            t_ijk_R = min(time_available, max_travel_time)  # 本段行驶时间（分钟）

            # Calculate the energy consumption in the current time period
            energy_this_period = self.calculate_energy_consumption(u_ijk, travel_speed, t_ijk_R / 60.0)  # 转为小时

            # Update
            total_energy += energy_this_period
            total_travel_time += t_ijk_R
            remaining_distance -= travel_speed * (t_ijk_R / 60.0)
            departure_time += t_ijk_R

        return total_energy, total_travel_time, travel_speed

    def calculate_charging_time(self, q_ik):
        charging_time = max(q_ik, 0.0)     # 防止负值
        return charging_time * 60 / (0.9 * 60)  # 分钟

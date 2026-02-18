#  coding: UTF-8  #
'''
@Project     : RA model
@File        : RA.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/11/22 22:10
'''


# ra_model.py
import math

class RangeAnxietyModel:
    def __init__(self,
                 soc_max: float,
                 phi: float = 1.0,
                 theta_k: float = 2.0,
                 beta_T: float = 1.0,
                 k_sig: float = 0.1,
                 T0: float = 70.0,
                 kappa_peak: float = 1.2,
                 kappa_offpeak: float = 1.0,
                 peak_hours=(7, 9, 17, 19)):
        """
        soc_max: 电池最大电量（和 Decoder.Q_max 对齐）
        phi, theta_k: 能量维度参数
        beta_T, k_sig, T0: 时间 Sigmoid 参数
        kappa_peak, kappa_offpeak: 高峰/非高峰时段权重
        peak_hours: 一个简单表示高峰的区间，如 (7,9,17,19) 表示 7-9, 17-19 点
        """
        self.soc_max = soc_max
        self.phi = phi
        self.theta_k = theta_k
        self.beta_T = beta_T
        self.k_sig = k_sig
        self.T0 = T0
        self.kappa_peak = kappa_peak
        self.kappa_offpeak = kappa_offpeak
        self.peak_hours = peak_hours  # (start_morning, end_morning, start_evening, end_evening)

        # 预先算好 Sigmoid 归一化用的常数
        self._sig0 = 1.0 / (1.0 + math.exp(self.k_sig * self.T0))   # 1/(1+e^{kT0})
        self._den  = 1.0 - self._sig0                               # 分母 1-1/(1+e^{kT0})

    def _gQ(self, Qtr):
        x = max(0.0, min(1.0, Qtr / (self.soc_max + 1e-9)))
        R_energy = self.phi * (x ** self.theta_k)
        return R_energy
    
    def _gT(self, Ttr):
        if Ttr < 0:
            Ttr = 0.0
        sig = 1.0 / (1.0 + math.exp(-self.k_sig * (Ttr - self.T0)))
        # 归一化后的 Sigmoid，再乘 βT
        return self.beta_T * ( (sig - self._sig0) / (self._den + 1e-9) )

    def _gh(self, hour_of_day):
        """
        hour_of_day: [0,24) 小时；如果你没有真实时间，可以先固定传 None，直接用 off-peak
        """
        if hour_of_day is None:
            return self.kappa_offpeak

        h = hour_of_day % 24.0
        s_m, e_m, s_e, e_e = self.peak_hours
        in_peak = (s_m <= h < e_m) or (s_e <= h < e_e)
        return self.kappa_peak if in_peak else self.kappa_offpeak

    def R_instant(self, Qtr, Ttr, hour_of_day):
        """
        归一化到 0~1
        瞬时 RA 值 R(Qtr, Ttr, h)
        Qtr: 已消耗电量 = soc_max - 当前剩余电量
        Ttr: 行驶时间（分钟或者小时，只要和 T0 一致即可）
        """
        gQ = self._gQ(Qtr)
        gT = self._gT(Ttr)
        gh = self._gh(hour_of_day)

        R = gQ * gT * gh
        Rmax = self.phi * self.beta_T * max(self.kappa_peak, self.kappa_offpeak)
        return R / (Rmax + 1e-9)

    def SUMR(self, Qend, Ttr, hour_of_day):
        """
        累积 RA：SUMR(Qend)
        """
        x = max(0.0, Qend / (self.soc_max + 1e-9))
        gT = self._gT(Ttr)
        gh = self._gh(hour_of_day)
        return gT * gh * self.phi / (self.theta_k + 1.0) * (x ** (self.theta_k + 1.0))









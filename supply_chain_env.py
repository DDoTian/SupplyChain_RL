import numpy as np
import math


class SupplyChainEnv:
    def __init__(self, con_mat, price, k_pr, k_st, k_pe, k_tr, lead_time, st_max, de_hist_len, zeta):
        self.con_mat = con_mat
        self.price = price
        self.k_pr = k_pr
        self.k_st = k_st
        self.k_pe = k_pe
        self.k_tr = k_tr
        self.lead_time = lead_time
        self.st_max = st_max
        self.de_hist_len = de_hist_len
        self.zeta = zeta

        self.factory_num = len(self.con_mat)
        self.warehouse_num = len(self.con_mat[0])
        self.action_dim = np.sum(self.con_mat) + self.factory_num
        self.state_dim = self.factory_num + self.warehouse_num + self.de_hist_len*self.warehouse_num

        self.s_f = [[0] for _ in range(self.factory_num)]
        self.s_w = [[0] for _ in range(self.warehouse_num)]
        self.a_f = [[] for _ in range(self.factory_num)]
        self.a_w = [[[] for _ in range(self.warehouse_num)] for _ in range(self.factory_num)]
        self.d = [[] for _ in range(self.warehouse_num)]

    def reset(self, IC_s_f=None, IC_s_w=None):
        if IC_s_f:
            self.s_f = [[IC_s_f[i]] for i in range(self.factory_num)]
        else:
            self.s_f = [[0] for _ in range(self.factory_num)]
        if IC_s_w:
            self.s_w = [[IC_s_w[j]] for j in range(self.warehouse_num)]
        else:
            self.s_w = [[0] for _ in range(self.warehouse_num)]
        self.a_f = [[] for _ in range(self.factory_num)]
        self.a_w = [[[] for _ in range(self.warehouse_num)] for _ in range(self.factory_num)]
        self.d = [[] for _ in range(self.warehouse_num)]
        return np.concatenate((self.s_f, self.s_w, [[0]*self.de_hist_len]*self.warehouse_num), axis=None)

    def step(self, action_f, action_w, demand):
        for i in range(self.factory_num):
            self.a_f[i].append(action_f[i])
        for i in range(self.factory_num):
            for j in range(self.warehouse_num):
                self.a_w[i][j].append(action_w[i][j])
        for j in range(self.warehouse_num):
            self.d[j].append(demand[j])

        a_w_temp = action_w
        for i in range(self.factory_num):
            for j in range(self.warehouse_num):
                if len(self.a_w[i][j]) <= self.lead_time[i][j]:
                    a_w_temp[i][j] = 0
                else:
                    a_w_temp[i][j] = self.a_w[i][j][-self.lead_time[i][j]-1]

        revenue = 0
        for j in range(self.warehouse_num):
            revenue += self.price[j] * (min(self.d[j][-1], max(self.s_w[j][-1], 0)))
        pr_cost = 0
        for i in range(self.factory_num):
            pr_cost += self.k_pr[i] * self.a_f[i][-1]
        st_cost = 0
        for i in range(self.factory_num):
            st_cost += self.k_st[0][i] * max(self.s_f[i][-1], 0)
        for j in range(self.warehouse_num):
            st_cost += self.k_st[1][j] * max(self.s_w[j][-1], 0)
        pe_cost = 0
        for j in range(self.warehouse_num):
            pe_cost = self.k_pe[j] * min(self.s_w[j][-1], 0)
        tr_cost = 0
        for i in range(self.factory_num):
            for j in range(self.warehouse_num):
                tr_cost += self.k_tr[i][j] * math.ceil(a_w_temp[i][j] / self.zeta)
        one_step_r = revenue - pr_cost - st_cost + pe_cost - tr_cost

        s_f_next = [0] * self.factory_num
        s_w_next = [0] * self.warehouse_num
        for i in range(self.factory_num):
            sum_a_j = 0
            for j in range(self.warehouse_num):
                sum_a_j += self.a_w[i][j][-1]
            s_f_next[i] = min(max(self.s_f[i][-1], 0) + self.a_f[i][-1] - sum_a_j, self.st_max[0][i])
            self.s_f[i].append(s_f_next[i])

        for j in range(self.warehouse_num):
            sum_a_i = 0
            for i in range(self.factory_num):
                sum_a_i += a_w_temp[i][j]
            s_w_next[j] = min(max(self.s_w[j][-1], 0) + sum_a_i - demand[j], self.st_max[1][j])
            self.s_w[j].append(s_w_next[j])

        de_hist = [[0]*self.de_hist_len]*self.warehouse_num
        for j in range(self.warehouse_num):
            for k in range(self.de_hist_len):
                if len(self.d[j]) >= (self.de_hist_len - k):
                    de_hist[j][k] = self.d[j][-self.de_hist_len+k]
                else:
                    de_hist[j][k] = 0

        S_next = np.concatenate((s_f_next, s_w_next, de_hist), axis=None)
        terminal = 0
        return S_next, one_step_r, terminal

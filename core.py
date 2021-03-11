
import numpy as np
from functools import partial
from datetime import datetime, timedelta
from tqdm import tqdm

from helpers import pr_EI_long, pr_MO_long, pr_IM_long, get_T1_and_T2, R0
from const import *


class Simulator:
    def __init__(self, params, p0_time, total_days, bed_info, show_bar=False, verbose=0):
        self.params = params
        self.p0_time = p0_time
        self.total_days = total_days
        self.bed_info = bed_info
        self.show_bar = show_bar
        self.verbose = verbose
        
        # T=0 is the day before simulation
        self.pr_EI = partial(pr_EI_long, mu_ei=params.mu_ei, k=params.k_days)
        self.pr_MO = partial(pr_MO_long, mu_mo=params.mu_mo, k=params.k_days)
        self.pr_IM = partial(pr_IM_long, k=params.k_pt,  x0=params.x0_pt)
        
        self.num_stages = params.num_stages
        # used to calculate the R0 values for each stage
        self.E2I_by_days_by_stage = {s: np.zeros(params.k_days) for s in range(self.num_stages)}
        self.I2OM_by_days_by_stage = {s: np.zeros(params.k_days+1) for s in range(self.num_stages)}

        self.init()

    def plus_time_and_to_string(days):
        """return absoluate date, which is p0_time + days"""
        return (self.p0_time + timedelta(days=int(days))).strftime('%d/%m/%y')
        
    def init(self):
        self.create_total_array()
        self.create_delta_array()
        self.create_delta_plus_array()
        self.create_trans_array()
        self.populate_bed_info()

        self.create_I_array()
        
    def create_total_array(self):
        # the total number of each state at each day
        self.total_array = np.zeros((self.total_days+1, NUM_STATES), dtype=float)
        self.total_array[0, STATE.S] = self.params.total_population
        self.total_array[0, STATE.E] = self.params.initial_num_E
        self.total_array[0, STATE.I] = self.params.initial_num_I
        self.total_array[0, STATE.M] = self.params.initial_num_M

    def create_delta_array(self):
        self.delta_array = np.zeros((self.total_days+1, NUM_STATES), dtype=float)
        self.delta_array[0, STATE.S] = self.params.total_population
        self.delta_array[0, STATE.E] = self.params.initial_num_E
        self.delta_array[0, STATE.I] = self.params.initial_num_I
        self.delta_array[0, STATE.M] = self.params.initial_num_M

    def create_delta_plus_array(self):
        self.delta_plus_array = np.zeros((self.total_days+1, NUM_STATES), dtype=float)
        self.delta_plus_array[0, STATE.S] = self.params.total_population
        self.delta_plus_array[0, STATE.E] = self.params.initial_num_E
        self.delta_plus_array[0, STATE.I] = self.params.initial_num_I
        self.delta_plus_array[0, STATE.M] = self.params.initial_num_M

    def create_trans_array(self):
        self.trans_array = np.zeros((self.total_days+1, NUM_TRANS), dtype=float)
        self.trans_array[0, TRANS.S2E] = 0
        self.trans_array[0, TRANS.E2I] = 0
        self.trans_array[0, TRANS.I2M] = 0
        self.trans_array[0, TRANS.I2O] = 0
        self.trans_array[0, TRANS.M2O] = 0
        self.trans_array[0, TRANS.EbyE] = 0
        self.trans_array[0, TRANS.EbyI] = 0

    def populate_bed_info(self):
        for T, num in self.bed_info:
            self.delta_array[T, STATE.H] = num
            self.delta_plus_array[T, STATE.H] = num
            
        self.total_array[:, STATE.H] = np.cumsum(self.delta_plus_array[:, STATE.H])

    def create_I_array(self):
        # dynamic array recording the number of I at each day
        self.num_in_I = np.zeros((self.total_days+1), dtype=float)
        self.num_in_I[0] = self.params.initial_num_I

    def update_inf_probas(self, T):
        """get infection probability at time T"""
        self.inf_proba_E = min(1, self.total_array[T-1, STATE.E] * self.params.alpha_func(T-1))
        self.inf_proba_I = min(1, self.total_array[T-1, STATE.I] * self.params.beta_func(T-1))

        if np.isclose(self.inf_proba_E, 0):
            self.inf_proba_E = 0

        if np.isclose(self.inf_proba_I, 0):
            self.inf_proba_I = 0

        # infection by E or I
        inf_proba_sum = self.inf_proba_E + self.inf_proba_I
        if inf_proba_sum > 1:
            # bound it from above by 1
            self.inf_proba_E /= inf_proba_sum
            self.inf_proba_I /= inf_proba_sum

        self.inf_proba = self.inf_proba_E + self.inf_proba_I
        # self.inf_proba = min(1, self.inf_proba_E + self.inf_proba_I)  # bound it by 1

        assert self.inf_proba_E >= 0, self.inf_proba_E
        assert self.inf_proba_I >= 0, self.inf_proba_I
        assert self.inf_proba_E <= 1, \
            (self.total_array[T-1, STATE.E], self.params.alpha_func(T-1), self.inf_proba_E)
        assert self.inf_proba_I <= 1, \
            (self.total_array[T-1, STATE.I], self.params.beta_func(T-1),  self.inf_proba_I)
        assert self.inf_proba <= 1

    def update_day_offsets(self, T):
        # previous days to consider for E-I
        self.day_offsets = [t for t in range(1, self.params.k_days+1) if T - t >= 0]
        
    def get_S2E(self, T):
        S2E = (self.total_array[T-1, STATE.S] * self.inf_proba)
        return S2E
    
    def get_E2I(self, T):
        # what do they mean?
        self.E_by_E = self.inf_proba_E * self.total_array[T-1, STATE.S]
        self.E_by_I = self.inf_proba_I * self.total_array[T-1, STATE.S]

        # each element is the number of infections from E to I at a specific day in the past
        self.E2I_array = [self.pr_EI(t) * self.delta_plus_array[T-t, STATE.E] for t in self.day_offsets]
        
        E2I = np.sum(self.E2I_array)
        return E2I

    def get_I2O(self, T):
        # remaining I exceeding k_days go to O
        # (I -> O)
        if T - self.params.k_days - 1 >= 0:
            I2O = self.num_in_I[T - self.params.k_days - 1]
            self.num_in_I[T - self.params.k_days - 1] = 0
        else:
            I2O = 0

        return I2O

    def get_I2M(self, T):
        # I -> M: infected to hospitized
        self.I2M_array = np.array(
            [
                self.pr_IM(T-1, t, self.total_array) * self.delta_plus_array[T-t, STATE.I]
                for t in self.day_offsets
            ]
        )
        I2M = np.sum(self.I2M_array)

        # if hospital is full now
        # I -> M is not allowed (no I goes to hospital)
        if self.total_array[T-1,  STATE.M] == self.total_array[T-1, STATE.H]:
            assert I2M == 0
        
        return I2M

    def get_M2O(self, T):
        # M -> O: hospitized to recovered/dead
        M2O = np.sum([self.pr_MO(t) * self.delta_plus_array[T-t, STATE.M] for t in self.day_offsets])
        return M2O

    def update_delta_plus_array(self, T, S2E, E2I, I2O, I2M, M2O):
        self.delta_plus_array[T, STATE.S] = 0
        self.delta_plus_array[T, STATE.E] = S2E
        self.delta_plus_array[T, STATE.I] = E2I

        # some special attention regarding I -> M or O (due to hospital capacity)
        # some patients need to stay at home
        # when there are more people that needs to go to hospital than the hospital capacity
        remaining_hospital_capacity = self.total_array[T-1, STATE.H] - self.total_array[T-1, STATE.M]
        if (I2M - M2O) >= remaining_hospital_capacity:
            # if hospital is out of capcity
            I2M = remaining_hospital_capacity + M2O  # this many I goes to hospital
            self.I2M_array = I2M / np.sum(self.I2M_array) * self.I2M_array
            if self.verbose > 0:
                print('hospital is full')

        self.delta_plus_array[T, STATE.M] = I2M  # bound I2M by remaining capacity
        self.delta_plus_array[T, STATE.O] = M2O + I2O

    def update_I_array(self, T, E2I):
        # number of I on each day needs to be adjusted (due to I -> M)
        self.num_in_I[T] = E2I
        self.num_in_I[T - np.array(self.day_offsets, dtype=int)] -= self.I2M_array

    def check_and_log(self, S2E, E2I, I2O, I2M, M2O):
        # print and check the transition information
        for trans, v in zip(('S->E', 'E->I', 'I->O', 'I->M', 'M->O'), (S2E, E2I, I2O, I2M, M2O)):
            if np.isclose(v, 0):
                v = 0
            # transition is non-negative
            assert v >= 0, f'{trans}: {v}'
            if self.verbose > 0:
                print(f'{trans}: {v}')

        for v in [S2E, E2I, I2M, I2O, M2O]:
            assert not np.isnan(v)
            assert not np.isinf(v)

    def update_stage_stat(self, T, I2O):
        # print(E2I_by_days, self.E2I_array)
        stage = self.params.get_stage_num(T)
        self.E2I_by_days_by_stage[stage][:len(self.E2I_array)] += self.E2I_array
        self.I2OM_by_days_by_stage[stage][:len(self.I2M_array)] += self.I2M_array
        self.I2OM_by_days_by_stage[stage][-1] += I2O

    def update_major_arrays(self, T, S2E, E2I, I2O, I2M, M2O):
        delta_S = -S2E
        delta_E = S2E - E2I
        delta_I = E2I - I2M - I2O
        delta_M = I2M - M2O
        delta_O = I2O + M2O

        self.total_array[T, STATE.S] = self.total_array[T-1, STATE.S] + delta_S
        self.total_array[T, STATE.E] = self.total_array[T-1, STATE.E] + delta_E
        self.total_array[T, STATE.I] = self.total_array[T-1, STATE.I] + delta_I
        self.total_array[T, STATE.M] = self.total_array[T-1, STATE.M] + delta_M
        self.total_array[T, STATE.O] = self.total_array[T-1, STATE.O] + delta_O

        self.total_array[T, np.isclose(self.total_array[T, :], 0)] = 0   # it might be < 0
        
        self.trans_array[T, TRANS.S2E] = S2E
        self.trans_array[T, TRANS.E2I] = E2I
        self.trans_array[T, TRANS.I2M] = I2M
        self.trans_array[T, TRANS.I2O] = I2O
        self.trans_array[T, TRANS.M2O] = M2O
        self.trans_array[T, TRANS.EbyE] = self.E_by_E
        self.trans_array[T, TRANS.EbyI] = self.E_by_I

        self.delta_array[T, STATE.S] = delta_S
        self.delta_array[T, STATE.E] = delta_E
        self.delta_array[T, STATE.I] = delta_I
        self.delta_array[T, STATE.M] = delta_M
        self.delta_array[T, STATE.O] = delta_O

    def check_total_arrays(self, T):
        # the population size (regardless of states) should not change
        assert np.isclose(self.total_array[T, :-1].sum(), self.total_array[0, :-1].sum()), \
            '{} != {}'.format(self.total_array[T, :-1].sum(), self.total_array[0, :-1].sum())

        # hospital should be not over-capacited
        assert self.total_array[T, STATE.M] <= self.total_array[T, STATE.H]

        assert ((self.total_array[T, :]) >= 0).all(), self.total_array[T, :]  # all values are non-neg

    def print_current_total_info(self, T):
        if self.verbose > 0:
            for s, v in zip(STATES, self.total_array[T, :]):
                print(f'{s}: {v}')
            print(self.total_array[T, :].sum())
    
    def update_total_infected(self, T):
        self.total_infected = self.total_array[T, [STATE.M, STATE.E, STATE.I, STATE.O]].sum()

    def update_O_fraction(self, T):
        self.O_fraction = (self.total_array[T, STATE.O] / self.total_infected)
        
    def step(self, T):
        self.update_inf_probas(T)
        self.update_day_offsets(T)

        S2E = self.get_S2E(T)
        E2I = self.get_E2I(T)
        I2O = self.get_I2O(T)
        I2M = self.get_I2M(T)
        M2O = self.get_M2O(T)

        self.update_delta_plus_array(T, S2E, E2I, I2O, I2M, M2O)
        self.update_I_array(T, E2I)

        self.check_and_log(S2E, E2I, I2O, I2M, M2O)
        
        self.update_stage_stat(T, I2O)
        
        self.update_major_arrays(T, S2E, E2I, I2O, I2M, M2O)
        self.check_total_arrays(T)

        self.print_current_total_info(T)

        self.update_total_infected(T)
        self.update_O_fraction(T)

    def run(self):
        self.end_time = None

        iters = range(1, self.total_days+1)

        if self.show_bar:
            iters = tqdm(iters)

        for T in iters:
            if self.verbose > 0:
                print('-' * 10)
                print(f'at iteration {T}')

            self.step(T)

            # fraction of out-of-system exceeds 0.99
            # the simulation can stop
            # all states fixed
            if self.O_fraction >= 0.99:
                self.end_time = T
                print(f'O fraction  {O_fraction}')
                if (T+1) < self.total_array.shape[0]:
                    for s in range(NUM_STATES):
                        self.total_array[T+1:, s] = self.total_array[T, s]
                break
        stats = self.get_stat()
        return self.total_array, self.delta_array, delta_plus_array, self.trans_array, stats

    """
    below are functions which gather simulation statistics
    """
    def get_stats(self):
        """get statistics of simulation run"""
        stats = dict()

        stats['R0_by_stage'] = self.R0_by_stage()
        stats['end_time'] = self.get_end_time()
        stats['peak_time'] = self.get_peak_time()
        stats['when_dO_gt_dI'] = self.get_when_dO_gt_dI()
        stats['when_dO_gt_dE'] = self.get_when_dO_gt_dE()
        stats['turning_time_real'] = self.get_real_turning_time()
        stats['turning_time_theory'] = self.get_theoretical_turning_time()
        return stats

    def get_R0_by_stage(self):
        """get the R0 value for each stage (e.g., every two weeks)"""
        R0_by_stage = dict()
        for s in range(self.num_stages):
            T1, T2 = get_T1_and_T2(self.I2OM_by_days_by_stage[s], self.E2I_by_days_by_stage[s])
            alpha, beta = self.params.get_alpha_beta_by_stage(s)
            r0 = R0(self.params.total_population, alpha, beta, T1, T2)
            R0_by_stage[s] = (float(T1), float(T2), float(r0))
        return R0_by_stage

    def get_end_time(self):
        """when the epidemic ends, i.e., I and E are zero"""
        if self.end_time is not None:
            return (int(self.end_time), self.plus_time_and_to_string(self.end_time))
        else:
            return None

    def get_peak_time(self):
        """when the total infection count peaks"""
        peak_time = (self.total_array[:, STATE.M] + self.total_array[:, STATE.I]).argmax()
        return (int(peak_time), self.plus_time_and_to_string(peak_time))

    def get_dO_gt_dE(self):
        """when delta_plus O > delta_plus E"""
        try:
            when_dO_gt_dI = (
                self.delta_plus_array[:, STATE.O] > self.delta_plus_array[:, STATE.I]
            ).nonzero()[0].min()
        except ValueError:
            when_dO_gt_dI = None
        ret = ((int(when_dO_gt_dI), self.plus_time_and_to_string(when_dO_gt_dI))
               if when_dO_gt_dI is not None
               else None)
        return ret

    def get_when_dO_gt_dI(self):
        """when delta_plus O > delta_plus I"""
        try:
            when_dO_gt_dI = (
                self.delta_plus_array[:, STATE.O] > self.delta_plus_array[:, STATE.I]
            ).nonzero()[0].min()
        except ValueError:
            when_dO_gt_dI = None
        ret = ((int(when_dO_gt_dI), self.plus_time_and_to_string(when_dO_gt_dI))
               if when_dO_gt_dI is not None
               else None)
        return ret

    def get_real_turning_time(self):
        O = self.total_array[:, STATE.O]
        IM = self.total_array[:, STATE.I] + self.total_array[:, STATE.M]
        
        try:
            turning_time_real = (O > IM).nonzero()[0].min()
        except ValueError:
            turning_time_real = None

        return ((int(turning_time_real), self.plus_time_and_to_string(turning_time_real))
                if turning_time_real is not None
                else None)

    def get_theoretical_turning_time(self):
        O = self.total_array[:, STATE.O]
        IME = self.total_array[:, STATE.I] + self.total_array[:, STATE.M] + total_array[:, STATE.E]
        
        try:
            turning_time_theory = (O > IME).nonzero()[0].min()
        except ValueError:
            turning_time_theory = None
        return ((int(turning_time_theory), self.plus_time_and_to_string(turning_time_theory))
                if turning_time_theory is not None
                else None)
        

def do_simulation(
        total_days, bed_info,
        params,
        p0_time,
        show_bar=False,
        verbose=0
):
    """wrapper function for simulation run, for backward compatability"""
    sim = Simulator(params, p0_time, total_days, bed_info)
    ret = sim.run()
    return ret

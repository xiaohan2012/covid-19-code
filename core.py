
import numpy as np
from functools import partial
from datetime import datetime, timedelta
from tqdm import tqdm

from helpers import pr_EI_long, pr_MO_long, pr_IM_long, get_T1_and_T2, R0, T
from const import STATE, TRANS, STATE_VAC, TRANS_VAC


class Simulator:
    def __init__(
            self, params,
            p0_time=T('1/1/1970'),
            total_days=100,
            bed_info=None,
            show_bar=False,
            verbose=0
    ):
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

        self.init_state_space()
        self.init()

    def plus_time_and_to_string(self, days):
        """return absoluate date, which is p0_time + days"""
        return (self.p0_time + timedelta(days=int(days))).strftime('%d/%m/%y')

    def init_state_space(self):
        self.state_space = STATE
        self.trans_space = TRANS
        
    def init(self):
        self.create_total_array()
        self.create_delta_array()
        self.create_delta_plus_array()
        self.create_trans_array()
        self.populate_bed_info()

        self.create_I_array()
        
    def create_total_array(self):
        # the total number of each state at each day
        self.total_array = np.zeros((self.total_days+1, self.state_space.num_states), dtype=float)
        self.total_array[0, self.state_space.S] = (
            self.params.total_population - self.params.initial_num_E
            - self.params.initial_num_I - self.params.initial_num_M
        )
        self.total_array[0, self.state_space.E] = self.params.initial_num_E
        self.total_array[0, self.state_space.I] = self.params.initial_num_I
        self.total_array[0, self.state_space.M] = self.params.initial_num_M

    def create_delta_array(self):
        self.delta_array = np.zeros((self.total_days+1, self.state_space.num_states), dtype=float)
        self.delta_array[0, self.state_space.S] = (
            self.params.total_population - self.params.initial_num_E
            - self.params.initial_num_I - self.params.initial_num_M
        )
        self.delta_array[0, self.state_space.E] = self.params.initial_num_E
        self.delta_array[0, self.state_space.I] = self.params.initial_num_I
        self.delta_array[0, self.state_space.M] = self.params.initial_num_M

    def create_delta_plus_array(self):
        """
        an array that only counts the population that moves *into* each state
        """
        self.delta_plus_array = np.zeros((self.total_days+1, self.state_space.num_states), dtype=float)
        self.delta_plus_array[0, self.state_space.S] = (
            self.params.total_population - self.params.initial_num_E
            - self.params.initial_num_I - self.params.initial_num_M
        )
        self.delta_plus_array[0, self.state_space.E] = self.params.initial_num_E
        self.delta_plus_array[0, self.state_space.I] = self.params.initial_num_I
        self.delta_plus_array[0, self.state_space.M] = self.params.initial_num_M

    def create_trans_array(self):
        self.trans_array = np.zeros((self.total_days+1, self.trans_space.num_trans), dtype=float)
        self.trans_array[0, self.trans_space.S2E] = 0
        self.trans_array[0, self.trans_space.E2I] = 0
        self.trans_array[0, self.trans_space.I2M] = 0
        self.trans_array[0, self.trans_space.I2O] = 0
        self.trans_array[0, self.trans_space.M2O] = 0
        self.trans_array[0, self.trans_space.EbyE] = 0
        self.trans_array[0, self.trans_space.EbyI] = 0

    def populate_bed_info(self):
        if self.bed_info is not None:
            for T, num in self.bed_info:
                self.delta_array[T, self.state_space.H] = num
                self.delta_plus_array[T, self.state_space.H] = num
            
        self.total_array[:, self.state_space.H] = np.cumsum(self.delta_plus_array[:, self.state_space.H])

    def create_I_array(self):
        # dynamic array recording the number of I at each day
        self.num_in_I = np.zeros((self.total_days+1), dtype=float)
        self.num_in_I[0] = self.params.initial_num_I

    def update_inf_probas(self, T):
        """get infection probability at time T"""
        self.inf_proba_E = min(1, self.total_array[T-1, self.state_space.E] * self.params.alpha_func(T-1))
        self.inf_proba_I = min(1, self.total_array[T-1, self.state_space.I] * self.params.beta_func(T-1))

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
            (self.total_array[T-1, self.state_space.E], self.params.alpha_func(T-1), self.inf_proba_E)
        assert self.inf_proba_I <= 1, \
            (self.total_array[T-1, self.state_space.I], self.params.beta_func(T-1),  self.inf_proba_I)
        assert self.inf_proba <= 1

    def update_day_offsets(self, T):
        # previous days to consider for E-I
        self.day_offsets = [t for t in range(1, self.params.k_days+1) if T - t >= 0]
        
    def update_S2E(self, T):
        self.S2E = (self.total_array[T-1, self.state_space.S] * self.inf_proba)

        # S can be infected by two sources, E or I
        # here we decompose the statistics
        self.E_by_E = self.inf_proba_E * self.total_array[T-1, self.state_space.S]
        self.E_by_I = self.inf_proba_I * self.total_array[T-1, self.state_space.S]
        
    def update_E2I(self, T):
        # each element is the number of infections from E to I at a specific day in the past
        self.E2I_array = [
            self.pr_EI(t) * self.delta_plus_array[T-t, self.state_space.E]
            for t in self.day_offsets
        ]
        
        self.E2I = np.sum(self.E2I_array)

    def update_I2O(self, T):
        # remaining I exceeding k_days go to O
        # (I -> O)
        if T - self.params.k_days - 1 >= 0:
            self.I2O = self.num_in_I[T - self.params.k_days - 1]
            self.num_in_I[T - self.params.k_days - 1] = 0
        else:
            self.I2O = 0

    def update_I2M(self, T):
        # I -> M: infected to hospitized
        self.I2M_array = np.array(
            [
                self.pr_IM(T-1, t, self.total_array) * self.delta_plus_array[T-t, self.state_space.I]
                for t in self.day_offsets
            ]
        )
        # initial value for I2M, before considering hospital capacity
        self.I2M = np.sum(self.I2M_array)

        # some special attention regarding I -> M or O (due to hospital capacity)
        # some patients need to stay at home
        # when there are more people that needs to go to hospital than the hospital capacity
        remaining_hospital_capacity = (
            self.total_array[T-1, self.state_space.H]
            - self.total_array[T-1, self.state_space.M]
        )
        if (self.I2M - self.M2O) >= remaining_hospital_capacity:
            # if hospital is out of capcity
            # NOTE: I2M is change here!
            self.I2M = remaining_hospital_capacity + self.M2O  # this many I goes to hospital
            self.I2M_array = self.I2M / np.sum(self.I2M_array) * self.I2M_array
            if self.verbose > 0:
                print('hospital is full')

        # if hospital is full now
        # I -> M is not allowed (no I goes to hospital)
        # print('M=', self.total_array[T-1,  self.state_space.M], 'H=', self.total_array[T-1, self.state_space.H])
        # print('I2M', self.I2M)
        if self.total_array[T-1,  self.state_space.M] == self.total_array[T-1, self.state_space.H]:
            assert self.I2M == 0

    def update_M2O(self, T):
        # M -> O: hospitized to recovered/dead
        self.M2O = np.sum([
            self.pr_MO(t) * self.delta_plus_array[T-t, self.state_space.M]
            for t in self.day_offsets
        ])

    def update_delta_plus_array(self, T):
        self.delta_plus_array[T, self.state_space.S] = 0
        self.delta_plus_array[T, self.state_space.E] = self.S2E
        self.delta_plus_array[T, self.state_space.I] = self.E2I


        self.delta_plus_array[T, self.state_space.M] = self.I2M  # bound self.I2M by remaining capacity
        self.delta_plus_array[T, self.state_space.O] = self.M2O + self.I2O

    def update_I_array(self, T):
        # number of I on each day needs to be adjusted (due to I -> M)
        self.num_in_I[T] = self.E2I
        self.num_in_I[T - np.array(self.day_offsets, dtype=int)] -= self.I2M_array

    def check_and_log(self):
        # print and check the transition information
        for trans, v in zip(
                ('S->E', 'E->I', 'I->O', 'I->M', 'M->O'),
                (self.S2E, self.E2I, self.I2O, self.I2M, self.M2O)):
            if np.isclose(v, 0):
                v = 0
            # transition is non-negative
            assert v >= 0, f'{trans}: {v}'
            if self.verbose > 0:
                print(f'{trans}: {v}')

        for v in [self.S2E, self.E2I, self.I2M, self.I2O, self.M2O]:
            assert not np.isnan(v)
            assert not np.isinf(v)

    def update_stage_stat(self, T):
        # print(E2I_by_days, self.E2I_array)
        stage = self.params.get_stage_num(T)
        self.E2I_by_days_by_stage[stage][:len(self.E2I_array)] += self.E2I_array
        self.I2OM_by_days_by_stage[stage][:len(self.I2M_array)] += self.I2M_array
        self.I2OM_by_days_by_stage[stage][-1] += self.I2O

    def update_deltas(self, T):
        self.delta_S = -self.S2E
        self.delta_E = self.S2E - self.E2I
        self.delta_I = self.E2I - self.I2M - self.I2O
        self.delta_M = self.I2M - self.M2O
        self.delta_O = self.I2O + self.M2O

    def update_total_array(self, T):
        self.total_array[T, self.state_space.S] = self.total_array[T-1, self.state_space.S] + self.delta_S
        self.total_array[T, self.state_space.E] = self.total_array[T-1, self.state_space.E] + self.delta_E
        self.total_array[T, self.state_space.I] = self.total_array[T-1, self.state_space.I] + self.delta_I
        self.total_array[T, self.state_space.M] = self.total_array[T-1, self.state_space.M] + self.delta_M
        self.total_array[T, self.state_space.O] = self.total_array[T-1, self.state_space.O] + self.delta_O

        self.total_array[T, np.isclose(self.total_array[T, :], 0)] = 0   # it might be < 0

    def update_trans_array(self, T):
        self.trans_array[T, self.trans_space.S2E] = self.S2E
        self.trans_array[T, self.trans_space.E2I] = self.E2I
        self.trans_array[T, self.trans_space.I2M] = self.I2M
        self.trans_array[T, self.trans_space.I2O] = self.I2O
        self.trans_array[T, self.trans_space.M2O] = self.M2O
        self.trans_array[T, self.trans_space.EbyE] = self.E_by_E
        self.trans_array[T, self.trans_space.EbyI] = self.E_by_I

    def update_delta_array(self, T):
        self.delta_array[T, self.state_space.S] = self.delta_S
        self.delta_array[T, self.state_space.E] = self.delta_E
        self.delta_array[T, self.state_space.I] = self.delta_I
        self.delta_array[T, self.state_space.M] = self.delta_M
        self.delta_array[T, self.state_space.O] = self.delta_O
        
    def check_total_arrays(self, T):
        # the population size (regardless of states) should not change
        # print('total_array', self.total_array)
        # print('self.total_array[T, :-1]', self.total_array[T, :-1])        
        assert np.isclose(self.total_array[T, :-1].sum(), self.total_array[0, :-1].sum()), \
            '{} != {}'.format(self.total_array[T, :-1].sum(), self.total_array[0, :-1].sum())

        # hospital should be not over-capacited
        m_val, h_val = self.total_array[T, self.state_space.M], self.total_array[T, self.state_space.H]
        assert m_val <= h_val, '{} > {}'.format(m_val, h_val)

        assert ((self.total_array[T, :]) >= 0).all(), self.total_array[T, :]  # all values are non-neg

    def print_current_total_info(self, T):
        if self.verbose > 0:
            for s, v in zip(self.state_space.all_states, self.total_array[T, :]):
                print(f'{s}: {v}')
            print(self.total_array[T, :].sum())
    
    def update_total_infected(self, T):
        self.total_infected = self.total_array[T, [self.state_space.M, self.state_space.E, self.state_space.I, self.state_space.O]].sum()

    def update_O_fraction(self, T):
        self.O_fraction = (self.total_array[T, self.state_space.O] / self.total_infected)
        
    def step(self, T):
        self.update_inf_probas(T)
        self.update_day_offsets(T)

        # get the transition count
        self.update_S2E(T)
        self.update_E2I(T)
        self.update_I2O(T)
        self.update_M2O(T)
        self.update_I2M(T)

        self.update_delta_plus_array(T)
        self.update_I_array(T)

        self.check_and_log()
        
        self.update_stage_stat(T)

        self.update_deltas(T)
        self.update_total_array(T)
        self.check_total_arrays(T)
        
        self.update_delta_array(T)
        self.update_trans_array(T)
        
        self.print_current_total_info(T)

        self.update_total_infected(T)
        self.update_O_fraction(T)

    def run(self):
        self.end_time = None
        print('total_days', self.total_days)
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
                    for s in range(self.state_space.num_states):
                        self.total_array[T+1:, s] = self.total_array[T, s]
                break
        stats = self.get_stats()
        return self.total_array, self.delta_array, self.delta_plus_array, self.trans_array, stats

    """
    below are functions which gather simulation statistics
    """
    def get_stats(self):
        """get statistics of simulation run"""
        stats = dict()

        stats['R0_by_stage'] = self.get_R0_by_stage()
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
        peak_time = (self.total_array[:, self.state_space.M] + self.total_array[:, self.state_space.I]).argmax()
        return (int(peak_time), self.plus_time_and_to_string(peak_time))

    def get_when_dO_gt_dE(self):
        """when delta_plus O > delta_plus E"""
        try:
            when_dO_gt_dI = (
                self.delta_plus_array[:, self.state_space.O] > self.delta_plus_array[:, self.state_space.I]
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
                self.delta_plus_array[:, self.state_space.O] > self.delta_plus_array[:, self.state_space.I]
            ).nonzero()[0].min()
        except ValueError:
            when_dO_gt_dI = None
        ret = ((int(when_dO_gt_dI), self.plus_time_and_to_string(when_dO_gt_dI))
               if when_dO_gt_dI is not None
               else None)
        return ret

    def get_real_turning_time(self):
        O = self.total_array[:, self.state_space.O]
        IM = self.total_array[:, self.state_space.I] + self.total_array[:, self.state_space.M]
        
        try:
            turning_time_real = (O > IM).nonzero()[0].min()
        except ValueError:
            turning_time_real = None

        return ((int(turning_time_real), self.plus_time_and_to_string(turning_time_real))
                if turning_time_real is not None
                else None)

    def get_theoretical_turning_time(self):
        O = self.total_array[:, self.state_space.O]
        IME = self.total_array[:, self.state_space.I] + self.total_array[:, self.state_space.M] + self.total_array[:, self.state_space.E]
        
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


class SimulatorWithVaccination(Simulator):
    """
    assumptions:

    - S->V takes over S->E
    - V cannot be infected (before vaccination takes effect, the vaccinated population is protected from illness)
    - V1 becomes EV1 after one day at the earliest (V1 cannot immediately become EV1 on the same day)
    - V2 stays at V2 (why not go to R)?
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inf_proba_E = 0.0
        self.inf_proba_I = 0.0
        self.inf_proba_EV1 = 0.0

    def init_state_space(self):
        self.state_space = STATE_VAC
        self.trans_space = TRANS_VAC

    def create_total_array(self):
        # the total number of each state at each day
        super().create_total_array()
        
    def create_delta_array(self):
        super().create_delta_array()

    def create_delta_plus_array(self):
        super().create_delta_plus_array()

    def create_trans_array(self):
        super().create_trans_array()

    def update_inf_probas(self, T):
        """get infection probability at time T"""
        self.inf_proba_E = min(1, self.total_array[T-1, self.state_space.E] * self.params.alpha_func(T-1))
        self.inf_proba_I = min(1, self.total_array[T-1, self.state_space.I] * self.params.beta_func(T-1))
        self.inf_proba_EV1 = min(1, self.total_array[T-1, self.state_space.I] * self.params.gamma_func(T-1))

        if np.isclose(self.inf_proba_E, 0):
            self.inf_proba_E = 0

        if np.isclose(self.inf_proba_I, 0):
            self.inf_proba_I = 0

        if np.isclose(self.inf_proba_EV1, 0):
            self.inf_proba_EV1 = 0
            
        # infection by E or I
        inf_proba_sum = self.inf_proba_E + self.inf_proba_I + self.inf_proba_EV1
        if inf_proba_sum > 1:
            # bound it from above by 1
            self.inf_proba_E /= inf_proba_sum
            self.inf_proba_I /= inf_proba_sum
            self.inf_proba_EV1 /= inf_proba_sum

        self.inf_proba = self.inf_proba_E + self.inf_proba_I + self.inf_proba_EV1
        # self.inf_proba = min(1, self.inf_proba_E + self.inf_proba_I)  # bound it by 1

        assert self.inf_proba_E >= 0, self.inf_proba_E
        assert self.inf_proba_I >= 0, self.inf_proba_I
        assert self.inf_proba_EV1 >= 0, self.inf_proba_I
        assert self.inf_proba_E <= 1, \
            (self.total_array[T-1, self.state_space.E], self.params.alpha_func(T-1), self.inf_proba_E)
        assert self.inf_proba_I <= 1, \
            (self.total_array[T-1, self.state_space.I], self.params.beta_func(T-1),  self.inf_proba_I)
        assert self.inf_proba_EV1 <= 1, \
            (self.total_array[T-1, self.state_space.EV1], self.params.gamma_func(T-1),  self.inf_proba_EV1)
        assert self.inf_proba <= 1

    def update_delta_plus_array(self, T):
        self.delta_plus_array[T, self.state_space.S] = 0
        self.delta_plus_array[T, self.state_space.E] = self.S2E
        self.delta_plus_array[T, self.state_space.I] = self.E2I

        # some special attention regarding I -> M or O (due to hospital capacity)
        # some patients need to stay at home
        # when there are more people that needs to go to hospital than the hospital capacity
        remaining_hospital_capacity = (
            self.total_array[T-1, self.state_space.H] - self.total_array[T-1, self.state_space.M]
        )
        if (self.I2M - self.M2O) >= remaining_hospital_capacity:
            # if hospital is out of capcity
            # NOTE: I2M is change here!
            self.I2M = remaining_hospital_capacity + self.M2O  # this many I goes to hospital
            self.I2M_array = self.I2M / np.sum(self.I2M_array) * self.I2M_array
            if self.verbose > 0:
                print('hospital is full')

        self.delta_plus_array[T, self.state_space.M] = self.I2M  # bound self.I2M by remaining capacity
        self.delta_plus_array[T, self.state_space.O] = self.M2O + self.I2O

    def check_and_log(self):
        # print and check the transition information
        for trans, v in zip(
                ('S->E', 'E->I', 'I->O', 'I->M', 'M->O', 'S->V', 'V->S', 'V->V1', 'V->V2', 'V1->EV1'),
                (self.S2E, self.E2I, self.I2O, self.I2M, self.M2O,
                 self.S2V, self.V2S, self.V_to_V1, self.V_to_V2, self.V1_to_EV1)):
            if np.isclose(v, 0):
                v = 0
            # transition is non-negative
            assert v >= 0, f'{trans}: {v}'
            if self.verbose > 0:
                print(f'{trans}: {v}')

        for v in [self.S2E, self.E2I, self.I2M, self.I2O, self.M2O]:
            assert not np.isnan(v)
            assert not np.isinf(v)

    def update_total_array(self, T):
        super().update_total_array(T)
        self.total_array[T, self.state_space.V] = self.total_array[T-1, self.state_space.V] + self.delta_V
        self.total_array[T, self.state_space.V1] = (
            self.total_array[T-1, self.state_space.V1] + self.delta_V1
        )
        self.total_array[T, self.state_space.V2] = (
            self.total_array[T-1, self.state_space.V2] + self.delta_V2
        )
        self.total_array[T, self.state_space.EV1] = (
            self.total_array[T-1, self.state_space.EV1] + self.delta_EV1
        )

        self.total_array[T, np.isclose(self.total_array[T, :], 0)] = 0   # it might be < 0

    def update_delta_array(self, T):
        super().update_delta_array(T)
        self.delta_array[T, self.state_space.V] = self.delta_V
        self.delta_array[T, self.state_space.V1] = self.delta_V1
        self.delta_array[T, self.state_space.V2] = self.delta_V2
        self.delta_array[T, self.state_space.EV1] = self.delta_EV1

    def update_delta_plus_array(self, T):
        super().update_delta_plus_array(T)
        self.delta_plus_array[T, self.state_space.S] = self.V2S  # some vaccinations are ineffective
        self.delta_plus_array[T, self.state_space.V] = self.S2V
        self.delta_plus_array[T, self.state_space.V1] = self.V_to_V1
        self.delta_plus_array[T, self.state_space.V2] = self.V_to_V2
        self.delta_plus_array[T, self.state_space.EV1] = self.V1_to_EV1
        
    def update_deltas(self, T):
        self.delta_S = - self.S2E - self.S2V + self.V2S
        self.delta_E = self.S2E - self.E2I
        self.delta_I = self.E2I - self.I2M - self.I2O
        self.delta_M = self.I2M - self.M2O
        self.delta_O = self.I2O + self.M2O

        self.delta_V = self.S2V - self.V2S - self.V_to_V1 - self.V_to_V2
        self.delta_V1 = self.V_to_V1 - self.V1_to_EV1
        self.delta_V2 = self.V_to_V2
        self.delta_EV1 = self.V1_to_EV1

    def update_S2V(self, T):
        """
        we assume that S->V rules *over* S->E,
        meaning that if there are not enough population to be both infected and vaccinated,
        we choose vaccinated
        """
        if T >= self.params.vac_time:
            self.S2V = min(
                self.total_array[T-1, self.state_space.S],
                self.params.vac_count_per_day
            )
        else:
            self.S2V = 0

    def update_S2E(self, T):
        """
        we assume that S->V rules *over* S->E,
        meaning that if there are not enough population to be both infected and vaccinated,
        we choose vaccinated
        """
        self.S2E = max(
            0,
            (self.total_array[T-1, self.state_space.S] * self.inf_proba) - self.S2V
        )

        # S can be infected by three sources, E, I, or EV1
        # here we decompose the statistics
        self.E_by_E = self.inf_proba_E * self.total_array[T-1, self.state_space.S]
        self.E_by_I = self.inf_proba_I * self.total_array[T-1, self.state_space.S]
        # TODO: should we add the following
        # self.E_by_EV1 = self.inf_proba_I * self.total_array[T-1, self.state_space.S]

    def update_V2S(self, T):
        t = T - self.params.time_to_take_effect
        if t >= self.params.vac_time:
            self.V2S = self.delta_plus_array[t, self.state_space.V] * self.params.s_proba
        else:
            self.V2S = 0

    def update_V_to_V1(self, T):
        t = T - self.params.time_to_take_effect
        if t >= self.params.vac_time:
            self.V_to_V1 = self.delta_plus_array[t, self.state_space.V] * self.params.v1_proba
        else:
            self.V_to_V1 = 0
        
    def update_V_to_V2(self, T):
        t = T - self.params.time_to_take_effect
        if t >= self.params.vac_time:
            self.V_to_V2 = self.delta_plus_array[t, self.state_space.V] * self.params.v2_proba
        else:
            self.V_to_V2 = 0

    def update_V1_to_EV1(self, T):
        self.V1_to_EV1 = (self.inf_proba * self.total_array[T-1, self.state_space.V1])

    def step(self, T):
        self.update_inf_probas(T)
        self.update_day_offsets(T)

        # get the transition count
        # vaccination related
        self.update_S2V(T)
        self.update_V2S(T)
        self.update_V_to_V1(T)
        self.update_V_to_V2(T)
        self.update_V1_to_EV1(T)

        # what we have before
        self.update_S2E(T)
        self.update_E2I(T)
        self.update_I2O(T)
        self.update_M2O(T)
        self.update_I2M(T)


        self.update_delta_plus_array(T)
        self.update_I_array(T)

        self.check_and_log()
        
        self.update_stage_stat(T)
        
        self.update_deltas(T)
        self.update_total_array(T)
        self.check_total_arrays(T)
        
        self.update_delta_array(T)
        self.update_trans_array(T)

        self.print_current_total_info(T)

        self.update_total_infected(T)
        self.update_O_fraction(T)
        

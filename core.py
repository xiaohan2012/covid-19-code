
import numpy as np
from functools import partial
from datetime import  datetime, timedelta
from tqdm import tqdm

from helpers import  pr_EI_long, pr_MO_long, pr_IM_long, get_T1_and_T2, R0
from const import *


def create_total_array(total_days, params):
    # the total number of each state at each day
    total_data = np.zeros((total_days+1, NUM_STATES), dtype=float)
    total_data[0, STATE.S] = params.total_population
    total_data[0, STATE.E] = params.initial_num_E
    total_data[0, STATE.I] = params.initial_num_I
    total_data[0, STATE.M] = params.initial_num_M

    return total_data
    

def create_delta_array(total_days, params):
    delta_data = np.zeros((total_days+1, NUM_STATES), dtype=float)
    delta_data[0, STATE.S] = params.total_population
    delta_data[0, STATE.E] = params.initial_num_E
    delta_data[0, STATE.I] = params.initial_num_I
    delta_data[0, STATE.M] = params.initial_num_M
    return delta_data


def create_delta_plus_array(total_days, params):
    delta_plus = np.zeros((total_days+1, NUM_STATES), dtype=float)
    delta_plus[0, STATE.S] = params.total_population
    delta_plus[0, STATE.E] = params.initial_num_E
    delta_plus[0, STATE.I] = params.initial_num_I
    delta_plus[0, STATE.M] = params.initial_num_M
    return delta_plus


def create_trans_array(total_days):
    trans_data = np.zeros((total_days+1, NUM_TRANS), dtype=float)
    trans_data[0, TRANS.S2E] = 0
    trans_data[0, TRANS.E2I] = 0
    trans_data[0, TRANS.I2M] = 0
    trans_data[0, TRANS.I2O] = 0
    trans_data[0, TRANS.M2O] = 0
    trans_data[0, TRANS.EbyE] = 0
    trans_data[0, TRANS.EbyI] = 0

    return trans_data


def populate_bed_info(bed_info, total_data, delta_data, delta_plus):
    for T, num in bed_info:
        delta_data[T, STATE.H] = num
        delta_plus[T, STATE.H] = num
        
    total_data[:, STATE.H] = np.cumsum(delta_plus[:, STATE.H])
    

def do_simulation(
        total_days, bed_info,
        params,
        p0_time,
        show_bar=False,
        verbose=0
):
    """
    total_days: total number of days to simulate
    bed_info: list of tuples (time T,  number of new  beds at T)
    params: a Params object
    """
    # T=0 is the day before simulation
    pr_EI = partial(pr_EI_long, mu_ei=params.mu_ei, k=params.k_days)
    pr_MO = partial(pr_MO_long, mu_mo=params.mu_mo, k=params.k_days)
    pr_IM = partial(pr_IM_long, k=params.k_pt,  x0=params.x0_pt)
    
    num_stages = params.num_stages
    # used to calculate the R0 values for each stage
    E2I_by_days_by_stage = {s: np.zeros(params.k_days) for s in range(num_stages)}
    I2OM_by_days_by_stage = {s: np.zeros(params.k_days+1) for s in range(num_stages)}

    # the total number of each state at each day
    total_data = create_total_array(total_days, params)
    
    # the change of each state at each day compared to the previou day
    # the ith row means total_data[i, state] - total_data[i-1, state]
    # so the value can be positive, negative or zero
    delta_data = create_delta_array(total_days, params)

    # the number of additions of each state at each day
    # essentially, detal_plus[time, state] = max(0, detal_data[time, state])
    delta_plus = create_delta_plus_array(total_days, params)
    
    # number of state transitions happening at each day
    trans_data = create_trans_array(total_days)

    # populate bed information
    populate_bed_info(bed_info, total_data, delta_data, delta_plus)

    # dynamic array
    num_in_I = np.zeros((total_days+1), dtype=float)
    num_in_I[0] = params.initial_num_I

    end_time = None

    iters = range(1, total_days+1)

    if show_bar:
        iters = tqdm(iters)

    for T in iters:
        if verbose > 0:
            print('-' * 10)
            print(f'at iteration {T}')

        inf_proba_E = min(1, total_data[T-1, STATE.E] * params.alpha_func(T-1))
        inf_proba_I = min(1, total_data[T-1, STATE.I] * params.beta_func(T-1))

        if np.isclose(inf_proba_E, 0):
            inf_proba_E = 0

        if np.isclose(inf_proba_I, 0):
            inf_proba_I = 0

        # infection by E or I
        inf_proba_sum = inf_proba_E + inf_proba_I
        if inf_proba_sum > 1:
            # bound it from above by 1
            inf_proba_E /= inf_proba_sum
            inf_proba_I /= inf_proba_sum

        inf_proba = inf_proba_E + inf_proba_I
        # inf_proba = min(1, inf_proba_E + inf_proba_I)  # bound it by 1

        assert inf_proba_E >= 0, inf_proba_E
        assert inf_proba_I >= 0, inf_proba_I
        assert inf_proba_E <= 1, (total_data[T-1, STATE.E], params.alpha_func(T-1), inf_proba_E)
        assert inf_proba_I <= 1, (total_data[T-1, STATE.I], params.beta_func(T-1),  inf_proba_I)
        assert inf_proba <= 1

        # what do they mean?
        E_by_E = inf_proba_E * total_data[T-1, STATE.S]
        E_by_I = inf_proba_I * total_data[T-1, STATE.S]

        # previous days to consider for E-I
        day_offsets = [t for t in range(1, params.k_days+1) if T - t >= 0]

        S2E = (total_data[T-1, STATE.S] * inf_proba)

        # each element is the number of infections from E to I at a specific day in the past
        E2I_array = [pr_EI(t) * delta_plus[T-t, STATE.E] for t in day_offsets]
        
        E2I = np.sum(E2I_array)
        
        # remaining I exceeding k_days go to O
        # (I -> O)
        if T-params.k_days-1 >= 0:
            I2O = num_in_I[T-params.k_days-1]
            num_in_I[T-params.k_days-1] = 0
        else:
            I2O = 0

        # I -> M: infected to hospitized
        I2M_array = np.array(
            [
                pr_IM(T-1, t, total_data) * delta_plus[T-t, STATE.I]
                for t in day_offsets
            ]
        )
        I2M = np.sum(I2M_array)

        # M -> O: hospitized to recovered/dead
        M2O = np.sum([pr_MO(t) * delta_plus[T-t, STATE.M] for t in day_offsets])

        # if hospital is full now
        # I -> M is not allowed (no I goes to hospital)
        if total_data[T-1,  STATE.M] == total_data[T-1, STATE.H]:
            assert I2M == 0

        delta_plus[T, STATE.S] = 0
        delta_plus[T, STATE.E] = S2E
        delta_plus[T, STATE.I] = E2I

        # some special attention regarding I -> M or O (due to hospital capacity)
        # some patients need to stay at home
        # when there are more people that needs to go to hospital than the hospital capacity
        remaining_hospital_capacity = total_data[T-1, STATE.H] - total_data[T-1, STATE.M]
        if (I2M - M2O) >= remaining_hospital_capacity:
            # if hospital is out of capcity
            I2M = remaining_hospital_capacity + M2O  # this many I goes to hospital
            I2M_array = I2M / np.sum(I2M_array) * I2M_array
            if verbose > 0:
                print('hospital is full')

        delta_plus[T, STATE.M] = I2M  # bound I2M by remaining capacity
        delta_plus[T, STATE.O] = M2O + I2O

        # number of I on each day needs to be adjusted (due to I -> M)
        num_in_I[T] = E2I
        num_in_I[T-np.array(day_offsets, dtype=int)] -= I2M_array

        # print and check the transition information
        for trans, v in zip(('S->E', 'E->I', 'I->O', 'I->M', 'M->O'), (S2E, E2I, I2O, I2M, M2O)):
            if np.isclose(v, 0):
                v = 0
            # transition is non-negative
            assert v >= 0, f'{trans}: {v}'
            if verbose > 0:
                print(f'{trans}: {v}')

        for v in [S2E, E2I, I2M, I2O, M2O]:
            assert not np.isnan(v)
            assert not np.isinf(v)

        # print(E2I_by_days, E2I_array)
        stage = params.get_stage_num(T)
        E2I_by_days_by_stage[stage][:len(E2I_array)] += E2I_array
        I2OM_by_days_by_stage[stage][:len(I2M_array)] += I2M_array
        I2OM_by_days_by_stage[stage][-1] += I2O
        
        delta_S = -S2E
        delta_E = S2E - E2I
        delta_I = E2I - I2M - I2O
        delta_M = I2M - M2O
        delta_O = I2O + M2O

        total_data[T, STATE.S] = total_data[T-1, STATE.S] + delta_S
        total_data[T, STATE.E] = total_data[T-1, STATE.E] + delta_E
        total_data[T, STATE.I] = total_data[T-1, STATE.I] + delta_I
        total_data[T, STATE.M] = total_data[T-1, STATE.M] + delta_M
        total_data[T, STATE.O] = total_data[T-1, STATE.O] + delta_O

        trans_data[T, TRANS.S2E] = S2E
        trans_data[T, TRANS.E2I] = E2I
        trans_data[T, TRANS.I2M] = I2M
        trans_data[T, TRANS.I2O] = I2O
        trans_data[T, TRANS.M2O] = M2O
        trans_data[T, TRANS.EbyE] = E_by_E
        trans_data[T, TRANS.EbyI] = E_by_I
        
        if verbose > 0:
            for s, v in zip(STATES, total_data[T, :]):
                print(f'{s}: {v}')
            print(total_data[T, :].sum())

        # the population size (regardless of states) should not change
        assert np.isclose(total_data[T, :-1].sum(), total_data[0, :-1].sum()), \
            '{} != {}'.format(total_data[T, :-1].sum(), total_data[0, :-1].sum())

        # hospital should be not over-capacited
        assert total_data[T, STATE.M] <= total_data[T, STATE.H]

        total_data[T, np.isclose(total_data[T, :], 0)] = 0   # it might be < 0
        assert ((total_data[T, :]) >= 0).all(), total_data[T, :]  # all values are non-neg

        delta_data[T, STATE.S] = delta_S
        delta_data[T, STATE.E] = delta_E
        delta_data[T, STATE.I] = delta_I
        delta_data[T, STATE.M] = delta_M
        delta_data[T, STATE.O] = delta_O

        total_infected = total_data[T, [STATE.M, STATE.E, STATE.I, STATE.O]].sum()
        O_fraction = (total_data[T, STATE.O] / total_infected)
        if False and O_fraction >= 0.99:
            end_time = T
            print(f'O fraction  {O_fraction}')
            # fraction of out-of-system exceeds 0.99
            # the simulation can stop
            # all states fixed
            if (T+1) < total_data.shape[0]:
                for s in range(NUM_STATES):
                    total_data[T+1:, s] = total_data[T, s]
            break

    def plus_time_and_to_string(days):
        return (p0_time + timedelta(days=int(days))).strftime('%d/%m/%y')
    
    stats = dict()
    R0_by_stage = dict()

    # get the R0 value for each stage (e.g., two weeks)
    for s in range(num_stages):
        T1, T2 = get_T1_and_T2(I2OM_by_days_by_stage[s], E2I_by_days_by_stage[s])
        alpha, beta = params.get_alpha_beta_by_stage(s)
        r0 = R0(params.total_population, alpha, beta, T1, T2)
        R0_by_stage[s] = (float(T1), float(T2), float(r0))
    stats['R0_by_stage'] = R0_by_stage
    if end_time:
        stats['end_time'] = (int(end_time), plus_time_and_to_string(end_time))
    else:
        stats['end_time'] = None
    peak_time = (total_data[:, STATE.M] + total_data[:, STATE.I]).argmax()
    stats['peak_time'] = (int(peak_time), plus_time_and_to_string(peak_time))
    
    O = total_data[:, STATE.O]
    IM = total_data[:, STATE.I] + total_data[:, STATE.M]
    IME = IM + total_data[:, STATE.E]

    try:
        when_dO_gt_dI = (delta_plus[:, STATE.O] > delta_plus[:, STATE.I]).nonzero()[0].min()
    except ValueError:
        when_dO_gt_dI = None

    try:
        when_dO_gt_dE = (delta_plus[:, STATE.O] > delta_plus[:, STATE.E]).nonzero()[0].min()
    except ValueError:
        when_dO_gt_dE = None

    try:
        turning_time_real = (O > IM).nonzero()[0].min()
    except ValueError:
        turning_time_real = None

    try:
        turning_time_theory = (O > IME).nonzero()[0].min()
    except ValueError:
        turning_time_theory = None

    stats['when_dO_gt_dI'] = ((int(when_dO_gt_dI), plus_time_and_to_string(when_dO_gt_dI))
                              if when_dO_gt_dI is not None
                              else None)
    stats['when_dO_gt_dE'] = ((int(when_dO_gt_dE), plus_time_and_to_string(when_dO_gt_dE))
                              if when_dO_gt_dE is not None
                              else None)

    stats['turning_time_real'] = ((int(turning_time_real), plus_time_and_to_string(turning_time_real))
                                  if turning_time_real is not None
                                  else None)
    stats['turning_time_theory'] = ((int(turning_time_theory), plus_time_and_to_string(turning_time_theory))
                                    if turning_time_theory is not None
                                    else None)
    
    return total_data, delta_data, delta_plus, trans_data, stats

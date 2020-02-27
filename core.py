
import numpy as np
from functools import partial

from helpers import  pr_EI_long, pr_MO_long, pr_IM_long
from const import *


def do_simulation(
        total_days, bed_info,
        params, verbose=0
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

    data = np.zeros((total_days+1, NUM_STATES), dtype=float)
    data[0, STATE.S] = params.total_population
    data[0, STATE.E] = params.initial_num_E
    data[0, STATE.I] = params.initial_num_I

    # the ith row means data[i, state] - data[i-1, state]
    delta_data = np.zeros((total_days+1, NUM_STATES), dtype=float)
    delta_data[0, STATE.S] = params.total_population
    delta_data[0, STATE.E] = params.initial_num_E
    delta_data[0, STATE.I] = params.initial_num_I

    # the number of increaset of each state per day
    increase_data = np.zeros((total_days+1, NUM_STATES), dtype=float)
    increase_data[0, STATE.S] = params.total_population
    increase_data[0, STATE.E] = params.initial_num_E
    increase_data[0, STATE.I] = params.initial_num_I

    for T, num in bed_info:
        delta_data[T, STATE.H] = num
        increase_data[T, STATE.H] = num
        
    data[:, STATE.H] = np.cumsum(increase_data[:, STATE.H])

    # dynamic array
    num_in_I = np.zeros((total_days+1), dtype=float)
    num_in_I[0] = params.initial_num_I

    for T in range(1, total_days+1):
        if verbose > 0:
            print('-' * 10)
            print(f'at iteration {T}')

        inf_proba_E = min(1, data[T-1, STATE.E] * params.alpha)
        inf_proba_I = min(1, data[T-1, STATE.I] * params.beta)

        if np.isclose(inf_proba_E, 0):
            inf_proba_E = 0

        if np.isclose(inf_proba_I, 0):
            inf_proba_I = 0

        inf_proba = min(1, inf_proba_E + inf_proba_I)  # bound it by 1

        assert inf_proba_E >= 0, inf_proba_E
        assert inf_proba_I >= 0, inf_proba_I
        assert inf_proba_E <= 1, (data[T-1, STATE.E], params.alpha, inf_proba_E)
        assert inf_proba_I <= 1, (data[T-1, STATE.I], params.beta,  inf_proba_I)
        assert inf_proba <= 1

        day_offsets = [t for t in range(1, params.k_days+1) if T - t >= 0]

        S2E = (data[T-1, STATE.S] * inf_proba)

        E2I = np.sum([pr_EI(t) * increase_data[T-t, STATE.E] for t in day_offsets])

        # remaining I exceeding k days go to O
        if T-params.k_days-1 >= 0:
            I2O = num_in_I[T-params.k_days-1]
            num_in_I[T-params.k_days-1] = 0
        else:
            I2O = 0

        I2M_array = np.array(
            [
                pr_IM(T-1, t, data) * increase_data[T-t, STATE.I]
                for t in day_offsets
            ]
        )
        I2M = np.sum(I2M_array)

        M2O = np.sum([pr_MO(t) * increase_data[T-t, STATE.M] for t in day_offsets])

        if data[T-1,  STATE.M] == data[T-1, STATE.H]:
            assert I2M == 0

        increase_data[T, STATE.S] = 0
        increase_data[T, STATE.E] = S2E
        increase_data[T, STATE.I] = E2I

        # some patients need to stay at home
        # when there are more people that needs to go to hospital than the hospital capacity
        remaining_hospital_capacity = data[T-1, STATE.H] - data[T-1, STATE.M]
        if (I2M - M2O) >= remaining_hospital_capacity:
            I2M = remaining_hospital_capacity + M2O
            I2M_array = I2M / np.sum(I2M_array) * I2M_array
            if verbose > 0:
                print('hospital is full')

        increase_data[T, STATE.M] = I2M  # bound I2M by remaining capacity
        increase_data[T, STATE.O] = M2O + I2O

        num_in_I[T] = E2I
        num_in_I[T-np.array(day_offsets, dtype=int)] -= I2M_array

        for trans, v in zip(('S->E', 'E->I', 'I->O', 'I->M', 'M->O'), (S2E, E2I, I2O, I2M, M2O)):
            assert v >= 0, f'{trans}: {v}'
            if verbose > 0:
                print(f'{trans}: {v}')

        for v in [S2E, E2I, I2M, I2O, M2O]:
            assert not np.isnan(v)
            assert not np.isinf(v)

        delta_S = -S2E
        delta_E = S2E - E2I
        delta_I = E2I - I2M - I2O
        delta_M = I2M - M2O
        delta_O = I2O + M2O

        data[T, STATE.S] = data[T-1, STATE.S] + delta_S
        data[T, STATE.E] = data[T-1, STATE.E] + delta_E
        data[T, STATE.I] = data[T-1, STATE.I] + delta_I
        data[T, STATE.M] = data[T-1, STATE.M] + delta_M
        data[T, STATE.O] = data[T-1, STATE.O] + delta_O

        if verbose > 0:
            for s, v in zip(STATES, data[T, :]):
                print(f'{s}: {v}')
            print(data[T, :].sum())

        assert np.isclose(data[T, :-1].sum(), data[0, :-1].sum()), \
            '{} != {}'.format(data[T, :-1].sum(), data[0, :-1].sum())

        assert data[T, STATE.M] <= data[T, STATE.H]

        data[T, np.isclose(data[T, :], 0)] = 0   # it might be < 0 sometimes
        assert ((data[T, :]) >= 0).all(), data[T, :]

        delta_data[T, STATE.S] = delta_S
        delta_data[T, STATE.E] = delta_E
        delta_data[T, STATE.I] = delta_I
        delta_data[T, STATE.M] = delta_M
        delta_data[T, STATE.O] = delta_O

    return data, delta_data, increase_data

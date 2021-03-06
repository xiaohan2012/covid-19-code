import pickle as pkl

from helpers import Params, T
from core_old import do_simulation as do_simulation_old
from core import do_simulation

from numpy.testing import assert_almost_equal as assert_areq


def get_input():
    p0_time = T('29/11/2019')
    bed_info = pkl.load(open('data/bed_info.pkl', 'rb'))
    days_before_ld = 55

    offset = 14  # each stage = 2 weeks
    n_offsets = 4  # we consider n stages
    total_days = days_before_ld + offset * n_offsets

    alpha_before, alpha_after = 3.2e-08, 1.6e-08
    days_offsets = list(range(offset, offset*n_offsets+1, offset))
    fine_grained_alpha = [(0, alpha_before), (days_before_ld, alpha_after)]
    fine_grained_alpha += [
        (days_before_ld + i, alpha_after) for i in days_offsets
        
    ]
    beta_before, beta_after = 3.6e-09, 1.8e-09
    fine_grained_beta = [(0, beta_before), (days_before_ld, beta_after)]
    fine_grained_beta += [
            (days_before_ld + i, beta_after) for i in days_offsets
        
    ]
    params = Params(
            total_population=9000000,
            initial_num_E=1,
            initial_num_I=0,
            initial_num_M=0,
            mu_ei=6,
            mu_mo=10,
            k_days=14,
            x0_pt=12000,
            alpha=fine_grained_alpha,
            beta=fine_grained_beta,
            stages=[days_before_ld] + [(days_before_ld + i) for i in days_offsets]
        
    )
    return p0_time, total_days, bed_info, params


def test_equivalence():
    p0_time, total_days, bed_info, params = get_input()

    total_actual, delta_actual, increase_actual, trans_data_actual, stats_actual = do_simulation(
        total_days, bed_info, params, p0_time=p0_time, verbose=0, show_bar=True
    )
    
    total_expected, delta_expected, increase_expected, trans_data_expected, stats_expected = do_simulation_old(
        total_days, bed_info, params, p0_time=p0_time, verbose=0, show_bar=True
    )

    assert_areq(total_expected, total_actual)
    assert_areq(delta_expected, delta_actual)
    assert_areq(increase_expected, increase_actual)
    assert_areq(trans_data_expected, trans_data_actual)
    assert stats_actual == stats_expected


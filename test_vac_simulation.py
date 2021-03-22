import pytest
import numpy as np
from helpers import Params, ParamsVac
from core import SimulatorWithVaccination


VERBOSE = 0

@pytest.fixture
def infectiousless_params():
    return Params(
        total_population=5,
        initial_num_E=0,
        initial_num_I=0,
        initial_num_M=0,
        alpha=0.0,
        beta=0.0
    )


def test_prefect_vaccination_1(infectiousless_params):
    """test case: no infection + perfect vaccination (v2_proba = 1)
    and it takes long for the vaccination to take effect (people stay in V)
    """
    params_vac = ParamsVac(
        vac_time=2,
        vac_count_per_day=2,
        gamma=0.0,
        s_proba=.0,
        v2_proba=1.0,
        v1_proba=0.0,
        time_to_take_effect=100,
        **infectiousless_params.kwargs
    )

    total_days = 5

    sim = SimulatorWithVaccination(params_vac, total_days=total_days, verbose=VERBOSE)
    sim.run()
    # check S array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.S],
        [5, 5, 3, 1, 0, 0]
    )
    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.S],
        [5, 0, -2, -2, -1, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.S],
        [5, 0, 0, 0, 0, 0]
    )

    # check V array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V],
        [0, 0, 2, 4, 5, 5]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V],
        [0, 0, 2, 2, 1, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V],
        [0, 0, 2, 2, 1, 0]
    )

    # check V1 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0]
    )
    
    # check V2 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V2],
        [0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V2],
        [0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V2],
        [0, 0, 0, 0, 0, 0]
    )


def test_prefect_vaccination_2(infectiousless_params):
    """test case: no infection + perfect vaccination (v2_proba = 1)
    time for vaccination to take effec is short
    """
    params_vac = ParamsVac(
        vac_time=2,
        vac_count_per_day=2,
        gamma=0.0,
        s_proba=.0,
        v2_proba=1.0,
        v1_proba=0.0,
        time_to_take_effect=1,
        **infectiousless_params.kwargs
    )

    total_days = 5

    sim = SimulatorWithVaccination(params_vac, total_days=total_days, verbose=VERBOSE)
    sim.run()
    # check S array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.S],
        [5, 5, 3, 1, 0, 0]
    )
    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.S],
        [5, 0, -2, -2, -1, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.S],
        [5, 0, 0, 0, 0, 0]
    )

    # check V array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V],
        [0, 0, 2, 2, 1, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V],
        [0, 0, 2, 0, -1, -1]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V],
        [0, 0, 2, 2, 1, 0]
    )

    # check V1 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0]
    )
    
    # check V2 array
    # basically, V2 is the same as V but shifted by `time_to_take_effect` days
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V2],
        [0, 0, 0, 2, 4, 5]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V2],
        [0, 0, 0, 2, 2, 1]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V2],
        [0, 0, 0, 2, 2, 1]
    )


def test_worst_possible_vaccination_1(infectiousless_params):
    """test case: no infection + worst possible vaccination (s_proba = 1.0)
    time for vaccination to take effect is 1 day
    """
    params_vac = ParamsVac(
        vac_time=2,
        vac_count_per_day=2,
        gamma=0.0,
        s_proba=1.0,
        v2_proba=0.0,
        v1_proba=0.0,
        time_to_take_effect=1,
        **infectiousless_params.kwargs
    )

    total_days = 6

    sim = SimulatorWithVaccination(params_vac, total_days=total_days, verbose=VERBOSE)
    sim.run()
    # check S array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.S],
        [5, 5, 3, 3, 3, 3, 3]
    )
    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.S],
        [5, 0, -2, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.S],
        [5, 0, 0, 2, 2, 2, 2]
    )

    # check V array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V],
        [0, 0, 2, 2, 2, 2, 2]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V],
        [0, 0, 2, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V],
        [0, 0, 2, 2, 2, 2, 2]
    )

    # check V1 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0, 0]
    )
    
    # check V2 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V2],
        [0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V2],
        [0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V2],
        [0, 0, 0, 0, 0, 0, 0]
    )


def test_worst_possible_vaccination_2(infectiousless_params):
    """test case: no infection + worst possible vaccination (s_proba = 1.0)
    time for vaccination to take effect is 2 days
    """
    params_vac = ParamsVac(
        vac_time=2,
        vac_count_per_day=2,
        gamma=0.0,
        s_proba=1.0,
        v2_proba=0.0,
        v1_proba=0.0,
        time_to_take_effect=2,
        **infectiousless_params.kwargs
    )

    total_days = 6

    sim = SimulatorWithVaccination(params_vac, total_days=total_days, verbose=VERBOSE)
    sim.run()
    # check S array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.S],
        [5, 5, 3, 1, 2, 2, 1]
    )
    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.S],
        [5, 0, -2, -2, 1, 0, -1]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.S],
        [5, 0, 0, 0, 2, 2, 1]
    )

    # check V array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V],
        [0, 0, 2, 4, 3, 3, 4]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V],
        [0, 0, 2, 2, -1, 0, 1]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V],
        [0, 0, 2, 2, 1, 2, 2]
    )

    # check V1 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0, 0]
    )
    
    # check V2 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V2],
        [0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V2],
        [0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V2],
        [0, 0, 0, 0, 0, 0, 0]
    )


def test_imprefect_vaccination_1(infectiousless_params):
    """test case: no infection + imperfect vaccination (v2_proba < 1)
    time for vaccination to take effect is short

    note that s_proba = 0
    """
    params_vac = ParamsVac(
        vac_time=2,
        vac_count_per_day=2,
        gamma=0.0,
        s_proba=0.0,
        v2_proba=0.5,
        v1_proba=0.5,
        time_to_take_effect=1,
        **infectiousless_params.kwargs
    )

    total_days = 5

    sim = SimulatorWithVaccination(params_vac, total_days=total_days, verbose=VERBOSE)
    sim.run()
    # check S array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.S],
        [5, 5, 3, 1, 0, 0]
    )
    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.S],
        [5, 0, -2, -2, -1, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.S],
        [5, 0, 0, 0, 0, 0]
    )

    # check V array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V],
        [0, 0, 2, 2, 1, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V],
        [0, 0, 2, 0, -1, -1]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V],
        [0, 0, 2, 2, 1, 0]
    )

    # check V1 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V1],
        [0, 0, 0, 1, 2, 2.5]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V1],
        [0, 0, 0, 1, 1, 0.5]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V1],
        [0, 0, 0, 1, 1, 0.5]
    )
    
    # check V2 array
    # basically, V2 is the same as V but shifted by `time_to_take_effect` days
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V2],
        [0, 0, 0, 1, 2, 2.5]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V2],
        [0, 0, 0, 1, 1, 0.5]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V2],
        [0, 0, 0, 1, 1, 0.5]
    )

    
def test_imprefect_vaccination_2(infectiousless_params):
    """test case: no infection + imperfect vaccination (v2_proba < 1)
    time for vaccination to take effect is short

    note that s_proba > 0 and v1_proba = 0
    """
    params_vac = ParamsVac(
        vac_time=2,
        vac_count_per_day=2,
        gamma=0.0,
        s_proba=0.5,
        v2_proba=0.5,
        v1_proba=0.0,
        time_to_take_effect=1,
        **infectiousless_params.kwargs
    )

    total_days = 8

    sim = SimulatorWithVaccination(params_vac, total_days=total_days, verbose=VERBOSE)
    sim.run()
    # check S array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.S],
        [5, 5, 3, 2, 1, 1, 1/2, 1/2, 1/4]
    )
    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.S],
        [5, 0, -2, -1, -1, 0, -1/2, 0, -1/4]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.S],
        [5, 0, 0, 1, 1, 1, 1/2, 1/2, 1/4]
    )

    # check V array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V],
        [0, 0, 2, 2, 2, 1, 1, 1/2, 1/2]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V],
        [0, 0, 2, 0, 0, -1, 0, -1/2, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V],
        [0, 0, 2, 2, 2, 1, 1, 1/2, 1/2]
    )

    # check V1 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    
    # check V2 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V2],
        [0, 0, 0, 1, 2, 3, 3+1/2, 4, 4+1/4]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V2],
        [0, 0, 0, 1, 1, 1, 1/2, 1/2, 1/4]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V2],
        [0, 0, 0, 1, 1, 1, 1/2, 1/2, 1/4]
    )


def test_infection_from_E_and_EV1():
    params = Params(
        total_population=6,
        initial_num_E=1,
        initial_num_I=0,
        initial_num_M=0,
        mu_ei=14,  # takes a long time for E to goto I
        alpha=1.0,  # E is very infectivous
        beta=0.0
    )

    params_vac = ParamsVac(
        vac_time=1,
        vac_count_per_day=5,
        gamma=1.0,
        s_proba=0.0,
        v2_proba=0.0,
        v1_proba=1.0,  # all go to V1
        time_to_take_effect=1,
        **params.kwargs
    )

    total_days = 3
    sim = SimulatorWithVaccination(params_vac, total_days=total_days, verbose=VERBOSE)
    assert sim.inf_proba_EV1 == 0.0
    sim.run()
    assert sim.inf_proba_EV1 > 0.0
    assert sim.inf_proba == 1.0

    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.S],
        [5, 0, 0, 0]
    )
    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.S],
        [5, -5, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.S],
        [5, 0, 0, 0]
    )

    # check V array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V],
        [0, 5, 0, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V],
        [0, 5, -5, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V],
        [0, 5, 0, 0]
    )

    # check V1 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.V1],
        [0, 0, 5, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V1],
        [0, 0, 5, -5]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V1],
        [0, 0, 5, 0]
    )

    # check EV1 array
    np.testing.assert_almost_equal(
        sim.total_array[:, sim.state_space.EV1],
        [0, 0, 0, 5]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.EV1],
        [0, 0, 0, 5]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.EV1],
        [0, 0, 0, 5]
    )

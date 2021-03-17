import pytest
import numpy as np
from helpers import Params, ParamsVac
from core import SimulatorWithVaccination


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


def test_prefect_vaccination(infectiousless_params):
    """test case: no infection + perfect vaccination (v2_proba)"""
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

    sim = SimulatorWithVaccination(params_vac, total_days=total_days, verbose=1)
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

    # TODO: make the following pass
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
        [0, 0, 2, 4, 5, 5]
    )

    np.testing.assert_almost_equal(
        sim.delta_array[:, sim.state_space.V2],
        [0, 0, 2, 2, 1, 0]
    )

    np.testing.assert_almost_equal(
        sim.delta_plus_array[:, sim.state_space.V2],
        [0, 0, 2, 2, 1, 0]
    )

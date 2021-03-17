import pytest
import pickle as pkl
from helpers import Params, ParamsVac, T
from core import SimulatorWithVaccination

@pytest.fixture
def infectiousless_params():
    return Params(
        total_population=5,
        initial_num_E=1,
        initial_num_I=0,
        initial_num_M=0,
        alpha=0.0,
        beta=0.0
    )


def test_V_array(infectiousless_params):
    params_vac = ParamsVac(
        vac_time=1,
        vac_count_per_day=2,
        gamma=0.0,
        s_proba=.0,
        v2_proba=.8,
        v1_proba=.2,
        **infectiousless_params.kwargs
    )

    total_days = 5

    sim = SimulatorWithVaccination(params_vac, total_days)
    sim.run()
    assert (sim.total_array[1:3, sim.state_space.V] == 2).all()
        
    assert sim.total_array[3, sim.state_space.V] == 1
    assert (sim.total_array[4:, sim.state_space.V] == 0).all()

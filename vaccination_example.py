#!/usr/bin/env python
# coding: utf-8

import pickle as pkl

from helpers import Params, ParamsVac, T
from core import do_simulation, SimulatorWithVaccination, Simulator


params = ParamsVac(
    # original parameters
    total_population=1000,
    initial_num_E=1,
    initial_num_I=0,
    initial_num_M=0,
    mu_ei=6,
    mu_mo=10,
    k_days=14,
    x0_pt=12000,
    alpha=0.0, # 3.2e-10,
    beta=0.0, # 1.6e-10,
    stages=None,
    # vaccination-related parameters
    vac_time=10,
    vac_count_per_day=100,
    time_to_take_effect=14,
    s_proba=0.05,
    v2_proba=0.7,
    v1_proba=0.25,
    ev1_to_r_time=14,
    gamma=0.001
)


p0_time = T('29/11/2019')
bed_info = pkl.load(open('data/bed_info.pkl', 'rb'))
total_days = 90

sim = SimulatorWithVaccination(params, p0_time, total_days, bed_info)
ret = sim.run()


sim.set_state_ids_to_plot(exclude_ids=[sim.state_space.S])
sim.plot_total()


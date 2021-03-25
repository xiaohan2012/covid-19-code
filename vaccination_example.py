#!/usr/bin/env python
# coding: utf-8

import pickle as pkl

from helpers import Params, ParamsVac, T
from core import do_simulation, SimulatorWithVaccination, Simulator

# **Issue 3**: changing alpha and beta
# you need to set the `stages`, `alpha` and `beta` arguments in ParamsVac class to reflect this
# for instance:
# alpha is a list of (starting day of the new alpha, alpha value)
alpha_list = [
    (0, 1e-9),  # initial (day, alpha value)
    (10, 1e-10),  # alpha value after 10 days
    (50, 1e-8),  # alpha value after 50 days
]
# so the alpha changes like:
# 1e-9 from day 0 to 9
# 1e-10 from day 10 to day 49
# 1e-8 from day 50 and on

# beta has the same format as alpha
beta_list = [
    (0, 1e-9),  # initial (day, beta value)
    (10, 1e-10),  # beta value after 10 days
    (50, 1e-8),  # beta value after 50 days
]

# each change of alpha/beta corresponds to entering into a new "stage"
# in this example, we have 3 stages:
# 0-9, 10-49, 50-end
stages = [10, 50]  # day 0 is excluded by convention

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
    alpha=alpha_list,  # set the changing alpha and beta here
    beta=beta_list,
    stages=stages,
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

# Issue 3 ends here

p0_time = T('29/11/2019')  # set the start time of epidemic

# **Issue 1:**
# the `bed_info` below was for the Wuhan case
# bed_info = pkl.load(open('data/bed_info.pkl', 'rb'))
# to set a fixed value for bed info, use the following
bed_info = [(0, 3000)]  # say there are 3000 beds at the begining of the epidemic

total_days = 90

sim = SimulatorWithVaccination(params, p0_time, total_days, bed_info)

# **Issue 2:**
# the returned variables below store time-dependent statistics
# - total_array: total number of each state on each day
# - delta_array: the delta (change of value of day t w.r.t day t-1) value of each state on each day
# - delta_array: the delta plus (newly incoming population on day t w.r.t day t-1) value of each state on each day
# - trans_array: number of transitions of each transition type (S->E, E->I, etc) on each day
# - stats: some statistics
total_array, delta_array, delta_plus_array, trans_array, stats = sim.run()

# to access the value of some state (say V1) on some variable (say total_array), do the following
total_V1_over_time = total_array[:, sim.state_space.V1]
total_V2_over_time = total_array[:, sim.state_space.V2]

sim.set_state_ids_to_plot(exclude_ids=[sim.state_space.S])
sim.plot_total()


import json
import os
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib as mpl

from datetime import datetime, timedelta
from scipy.stats import poisson
from matplotlib import pyplot as plt

from const import STATE, STATES, NUM_STATES, COLORS, TRANS

# mpl.style.use('paper')

DATE_FORMAT = '%d/%m/%Y'


def p_t(H_val, M_val, k, x0):
    """probability of going to hospital
    depends on the number of vacant beds H_val - M_val
    and logistic function parameters, k (controls steepiness) and x0
    """
    x = H_val - M_val
    if x > 0:
        return 1 / (1 + np.exp(-k * (x - x0)))
    else:
        return 0


def pr_IM_long(T, t, data,  k, x0):
    """
    probability of transitting from I at time T-t to M at time T
    """
    p_T = p_t(data[T, STATE.H], data[T, STATE.M], k, x0)
    if p_T == 0:
        return 0
    p_T_minus_t_list = np.array([p_t(data[T-i, STATE.H], data[T-i, STATE.M], k, x0) for i in range(1, t)])
    p = np.prod(1-p_T_minus_t_list) * p_T
    return p


def truncated_poisson(x, mu, min_x, max_x):
    assert x == int(x)
    x = int(x)
    assert x <= max_x
    all_probas = np.array([poisson.pmf(xi, mu) for xi in range(min_x, max_x+1)])
    all_probas /= all_probas.sum()
    assert np.isclose(all_probas.sum(), 1), all_probas
    return all_probas[x-min_x]


def pr_EI_long(t, mu_ei, k):
    assert t >= 1
    return truncated_poisson(t, mu_ei, 1, k)


def pr_MO_long(t, mu_mo, k):
    assert t >= 1
    return truncated_poisson(t, mu_mo, 1, k)


class Params:
    def __init__(
            self,
            # infection-related parameters:
            alpha=0.02, beta=0.01,
            mu_ei=6.0,
            mu_mo=14.0,
            mean_IM=7,
            x0_pt=10000, # k_pt=0.0001,
            # after k_days, an I becomes O (recovered/dead)
            k_days=28,
            # city-related
            total_population=9000000,
            initial_num_E=100,
            initial_num_I=20,
            initial_num_M=0,
            stages=None
        ):
        self.total_population = total_population
        self.initial_num_E = initial_num_E
        self.initial_num_I = initial_num_I
        self.initial_num_M = initial_num_M
        # probability  parameters
        # S -> E

        self.alpha = alpha
        self.beta = beta

        self.alpha_array = None
        self.beta_array = None
        
        # E -> I: Poisson
        self.mu_ei = mu_ei
        
        # I -> M: geoemtric
        self.x0_pt = x0_pt
        self.mean_IM = mean_IM
        self.k_pt = np.log(mean_IM-1) / x0_pt
        
        # M -> O: Poisson
        self.mu_mo = mu_mo

        # time window size
        self.k_days = k_days

        self.stages = stages
        self.num_stages = 1 if stages is None else (len(stages) + 1)

    def get_stage_num(self, t):
        if self.stages is None:
            return 0
        else:
            for i, time in enumerate(self.stages):
                assert time > 0
                if t < time:
                    return i
            return self.num_stages - 1
    
    def populate_alpha_array(self):
        """
        create an array which stores alpha values for each day

        assuming alpha is given as a list of tuples (time, value)
        """
        times = np.array([t for t, _ in self.alpha])
        values = np.array([v for _, v in self.alpha])  # alpha values
        
        max_t = times.max()
        self.alpha_array = np.zeros(max_t+1)
        for value, t1, t2 in zip(values[:-1], times[:-1], times[1:]):
            for i in range(t1, t2):
                self.alpha_array[i] = value
        self.alpha_array[max_t:] = values[-1]
        # print(self.alpha_array)
        
    def populate_beta_array(self):
        times = np.array([t for t, _ in self.beta])
        values = np.array([v for _, v in self.beta])
        
        max_t = times.max()
        self.beta_array = np.zeros(max_t+1)
        for value, t1, t2 in zip(values[:-1], times[:-1], times[1:]):
            for i in range(t1, t2):
                self.beta_array[i] = value
        self.beta_array[max_t:] = values[-1]
        # print(self.beta_array)
        
    def alpha_func(self, t):
        if isinstance(self.alpha, float):
            return self.alpha
        elif isinstance(self.alpha, list):
            if self.alpha_array is None:
                self.populate_alpha_array()
                
            if t >= len(self.alpha_array):
                return self.alpha_array[-1]
            else:
                return self.alpha_array[t]
        else:
            raise ValueError(f'cannot understand: {self.alpha}')

    def beta_func(self, t):
        if isinstance(self.beta, float):
            return self.beta
        elif isinstance(self.beta, list):
            if self.beta_array is None:
                self.populate_beta_array()
                
            if t >= len(self.beta_array):
                return self.beta_array[-1]
            else:
                return self.beta_array[t]
        else:
            raise ValueError(f'cannot understand: {self.beta}')

    def get_alpha_beta_by_stage(self, s):
        assert s < self.num_stages
        if self.num_stages == 1:
            return self.alpha, self.beta

        if s == self.num_stages - 1:
            t = self.stages[-1]
        else:
            t = self.stages[s] - 1

        return self.alpha_func(t), self.beta_func(t)

    @property
    def kwargs(self):
        """return the input parameters as a dict"""
        return dict(
            alpha=self.alpha,
            beta=self.beta,
            mu_ei=self.mu_ei,
            mu_mo=self.mu_mo,
            mean_IM=self.mean_IM,
            x0_pt=self.x0_pt,
            k_days=self.k_days,
            total_population=self.total_population,
            initial_num_E=self.initial_num_E,
            initial_num_I=self.initial_num_I,
            initial_num_M=self.initial_num_M,
            stages=self.stages
        )

    def __repr__(self):
        return f"""total_population: {self.total_population}
initial_num_E: {self.initial_num_E}
initial_num_I: {self.initial_num_I}
initial_num_M: {self.initial_num_M}

alpha: {self.alpha}
beta:  {self.beta}

mu_ei: {self.mu_ei}
mu_mo: {self.mu_mo}

x0_pt: {self.x0_pt}
k_pt:  {self.k_pt}
mean_IM: {self.mean_IM}

k_days: {self.k_days}
        """


class ParamsVac(Params):
    def __init__(
            self,
            vac_time=1,
            vac_count_per_day=50000,
            time_to_take_effect=14,
            s_proba=0.05,
            v2_proba=0.7,
            v1_proba=0.25,
            ev1_to_r_time=14,
            gamma=0.001,
            **kwargs):
        """
        vac_time: time to vacinate the population
        vac_count_per_day: how many people are vacinated per days
        time_to_take_effect: how many days does the vacination take effect
        s_proba: probability of transiting to S after vacination
        v1_proba: probability of transiting to V1 (protected but can tranmit virus after becoming EV1)  after vacination
        v2_proba: probability of transiting to V2 (protected and cannot tranmit virus) after vacination
        ev1_to_r_time: how many days does EV1 recover
        gamma: infection coefficient related to EV1
        (type): float or list of (time/int, value/float)
        """
        super().__init__(**kwargs)

        assert vac_time > 0
        self.vac_time = vac_time
        self.time_to_take_effect = time_to_take_effect

        self.vac_count_per_day = vac_count_per_day
        self.s_proba = s_proba
        self.v2_proba = v2_proba
        self.v1_proba = v1_proba
        assert np.isclose(self.s_proba + self.v2_proba + self.v1_proba, 1.0), 'vaccination probas not summing up to 1'

        self.ev1_to_r_time = ev1_to_r_time
        self.gamma = gamma

        self.gamma_array = None

    def populate_gamma_array(self):
        """gamma is the infection probability related to EV1"""
        times = np.array([t for t, _ in self.gamma])
        values = np.array([v for _, v in self.gamma])
        
        max_t = times.max()
        self.gamma_array = np.zeros(max_t+1)
        for value, t1, t2 in zip(values[:-1], times[:-1], times[1:]):
            for i in range(t1, t2):
                self.gamma_array[i] = value
        self.gamma_array[max_t:] = values[-1]

    def gamma_func(self, t):
        if isinstance(self.gamma, float):
            return self.gamma
        elif isinstance(self.gamma, list):
            if self.gamma_array is None:
                self.populate_gamma_array()
                
            if t >= len(self.gamma_array):
                return self.gamma_array[-1]
            else:
                return self.gamma_array[t]
        else:
            raise ValueError(f'cannot understand: {self.gamma}')

    def __repr__(self):
        s = super().__repr__()
        s_extra = """
-----------------
Vaccination params:
-----------------

vac_time:          {}
vac_count_per_day: {}
s_proba:           {}
v2_proba:          {}
v1_proba:          {}
ev1_to_r_time:     {}
gamma:             {}
        """.format(
            self.vac_time,
            self.vac_count_per_day,
            self.s_proba,
            self.v2_proba,
            self.v1_proba,
            self.ev1_to_r_time,
            self.gamma
        )
        return s + s_extra


def T(s):
    return datetime.strptime(s, DATE_FORMAT)


def get_T1_and_T2(I2OM_by_days, E2I_by_days):
    """what do these two days mean?"""
    I_num_and_day_array = np.array(
        [[num, d] for num, d in zip(I2OM_by_days, range(1, len(I2OM_by_days) + 1))]
    )
    total_num_I = I_num_and_day_array[:, 0].sum()

    if total_num_I > 0:
        mean_I_days = (I_num_and_day_array[:, 0] * I_num_and_day_array[:, 1]).sum() / total_num_I
    else:
        mean_I_days = float("nan")
    
    E_num_and_day_array = np.array(
        [[num, d] for num, d in zip(E2I_by_days, range(1, len(E2I_by_days) + 1))]
    )
    total_num_E = E_num_and_day_array[:, 0].sum()

    if total_num_E > 0:
        mean_E_days = (E_num_and_day_array[:, 0] * E_num_and_day_array[:, 1]).sum() / total_num_E
    else:
        mean_E_days = float("nan")

    return mean_E_days, mean_I_days


def R0(total_population, alpha, beta, T1, T2):
    return (1 + total_population * alpha * T1) * (1 + total_population * beta * T2)


def plot_total(total):
    fig, ax = plt.subplots(1, 1)
    for color, s in zip(COLORS[1:], range(1, NUM_STATES)):
        ax.plot(total[:, s], c=color)
    fig.legend(STATES[1:])
    fig.tight_layout()
    return fig, ax


def trans2df(trans, p0_time, total_days):
    df = pd.DataFrame.from_dict({
        'date': pd.date_range(p0_time, p0_time+timedelta(days=total_days)),
        'S2E':  trans[:, TRANS.S2E],
        'E2I':  trans[:, TRANS.E2I],
        'I2M':  trans[:, TRANS.I2M],
        'M2O':  trans[:, TRANS.M2O],
        'EbyE': trans[:, TRANS.EbyE],
        'EbyI': trans[:, TRANS.EbyI]
    })
    return df


def data2df(total, p0_time, total_days):
    df = pd.DataFrame.from_dict({
        'date': pd.date_range(p0_time, p0_time+timedelta(days=total_days)),
        'S': total[:, STATE.S],
        'E': total[:, STATE.E],
        'I': total[:, STATE.I],
        'M': total[:, STATE.M],
        'O': total[:, STATE.O],
        'H': total[:, STATE.H]
    })
    return df


def enhance_total(df):
    df['EIMO'] = df['E'] + df['I'] + df['M'] + df['O']
    df['IMO'] = df['I'] + df['M'] + df['O']
    df['IM'] = df['I'] + df['M']
    return df


def total_to_csv(p0_time, total_days, total, path):
    df = data2df(total, p0_time, total_days)
    df.to_csv(path, index=None)


def plot_total(total, p0_time, total_days):
    sbn.set_style("whitegrid")
    
    def np_to_dt(d):
        return pd.to_datetime(str(d))
    
    df = data2df(total, p0_time, total_days)
    df['date_str'] = df['date'].apply(lambda d: np_to_dt(d).strftime('%d/%m/%y'))

    def process_state(state):
        subdf = df[['date', state]]
        subdf['index'] = df.index
        subdf['value'] = subdf[state].copy()
        del subdf[state]
        subdf['state'] = state
        return subdf

    # S = process_state('S')
    E = process_state('E')
    I = process_state('I')
    M = process_state('M')
    O = process_state('O')
    H = process_state('H')
    ndf = pd.concat([E, I, M, O, H], ignore_index=True)

    nticks = 5
    step = int(np.floor(df.shape[0] / nticks))

    xticks = df['date_str'].index[::step].values
    xtick_labels = df['date_str'][::step].values
    print(xtick_labels)

    fig, ax = plt.subplots(1, 1)
    stuff = sbn.lineplot(
        x="index", y="value", hue='state', data=ndf, ax=ax,
        palette=['orange', 'red', 'pink', 'gray', 'blue'],
        legend=None
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=15)
    ax.set_xlabel('date')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,  0))
    ax.legend(stuff.lines, ['E', 'I', 'M', 'O', 'H'], loc='best')
    fig.tight_layout()

    return fig, ax


def save_to_json(obj, path):
    s = json.dumps(obj, indent=4, sort_keys=True)
    with open(path, 'w') as f:
        f.write(s)


def save_bundle(bundle, p0_time, total_days, dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    names = ['total', 'delta', 'increase', 'transition']
    for d, name in zip(bundle, names):
        if name == 'transition':
            df = trans2df(d, p0_time, total_days)
        else:
            df = data2df(d, p0_time, total_days)
            if name == 'total':
                df = enhance_total(df)
        df.to_csv(f'{dir_name}/{name}.csv', index=None)


def makedir_if_not_there(d):
    if not os.path.exists(d):
        os.makedirs(d)

        

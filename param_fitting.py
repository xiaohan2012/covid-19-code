import numpy as np
import pandas as pd
import pickle as pkl

from datetime import timedelta
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from joblib import delayed, Parallel


from core import  do_simulation
from helpers import Params, T, get_T1_and_T2, R0, makedir_if_not_there
from const import STATE, COLORS, NUM_STATES,  STATES


df = pd.read_csv('data/wuhan.csv', sep=',')
df['date'] = df['date'].apply(T)


### ATTENTION
# start and end date of date being  fitted
start_date = T('27/01/2020')  # T=0
end_date = T('09/02/2020')  #09/02



subdf = df[(df['date'] > start_date) & (df['date'] < end_date)] 

I_true = subdf['infected'].values
O_true = subdf['death'].values + subdf['cured'].values



### ATTENTION
#  below is the number of new beds on  some days
#  it  is increment, not  total number
total_days = subdf.shape[0]
bed_info_raw = [
    (T('27/01/2020'), 4000),
    (T('31/01/2020'), 6000), # 10000),
    (T('04/02/2020'), 1000), # 11000),
    (T('07/02/2020'), 2000), # 13000)
    (T('11/02/2020'), 6000),  # 19000
    (T('17/02/2020'), 1000),  # 20000
]
# number of new beds at  some days
bed_info = [((d-start_date).days, n) for d, n in bed_info_raw if d < end_date]


#### ATTENTION
#  below are the parameter search ranges
alpha_list = np.arange(0.1,  1.91,  step=0.1) * 1e-08
beta_list = np.arange(0.1,  1.91,  step=0.1) * 1e-09


# alpha_list = 1 / np.power(10, np.arange(7, 11, 1))
# beta_list = 1 / np.power(10, np.arange(7, 11, 1))

# assumption: alpha > beta
alpha_beta_choices = [
    (alpha,  beta)
    for alpha, beta in product(alpha_list, beta_list)
    if alpha > beta
] 
num_I_list = np.arange(5000, 10001, step=1000)
I2E_factors = [1., 1.5, 2]
I2M_factors = [0.25, 0.5, 1.0]

k_days_list = np.arange(8, 31, 3)
x0_pt_list = np.arange(3000, 21001, 3000)
mu_ei_list = [6] 
mu_mo_list = [14]

def one_run(initial_num_I, I2E_factor, I2M_factor, alpha, beta, mu_ei, mu_mo, k_days, x0_pt):
    initial_num_E = initial_num_I * I2E_factor
    initial_num_M = initial_num_I * I2M_factor
    initial_num_M = min(bed_info[0][1], initial_num_M)
    
    params = Params(
        initial_num_I=initial_num_I,
        initial_num_E=initial_num_E,
        initial_num_M=initial_num_M,
        alpha=alpha,
        beta=beta,
        mu_ei=mu_ei,
        mu_mo=mu_mo,
        k_days=k_days,
        x0_pt=x0_pt
    )
    total, delta, increase, trans_data, ax = do_simulation(
        total_days+3, bed_info, params, p0_time=start_date
    )
    
    I_mae = mean_absolute_error(I_true, increase[1:(total_days+1), STATE.I])
    O_mae = mean_absolute_error(O_true, increase[1:(total_days+1), STATE.O])

    is_decreasing_after_Feb09 = True
    for i in range((T('09/02/2020') - start_date).days, total.shape[0]-1):
        if total[i, STATE.I] < total[i+1, STATE.I]:
            is_decreasing_after_Feb09 = False
    row = (
        initial_num_I, initial_num_E, initial_num_M,
        alpha, beta, k_days, mu_ei, mu_mo, x0_pt,
        I_mae,  O_mae, is_decreasing_after_Feb09
    )
    return row
    
total_num_configs = (
    len(num_I_list)
    * len(alpha_beta_choices)
    * len(I2E_factors)
    * len(mu_ei_list)
    * len(mu_mo_list) 
    * len(I2M_factors)
    * len(k_days_list)
    * len(x0_pt_list)
)

rows = Parallel(n_jobs=-1)(
    delayed(one_run)(num_I, I2E_factor, I2M_factor, alpha, beta, mu_ei, mu_mo, k_days, x0_pt)
    for num_I, I2E_factor, I2M_factor, (alpha,  beta), mu_ei, mu_mo, k_days, x0_pt in tqdm(
        product(
            num_I_list, I2E_factors, I2M_factors, alpha_beta_choices,
            mu_ei_list, mu_mo_list, k_days_list, x0_pt_list
        ), total=total_num_configs
    )
)

res_df = pd.DataFrame(
    rows,
    columns=[
        'initial_num_I', 'initial_num_E', 'initial_num_M', 'alpha', 'beta',
        'k_days',
        'mu_ei', 'mu_mo', 'x0_pt',
        'I_mae', 'O_mae', 'is_feasible'
    ]
)


br = res_df[res_df['is_feasible']].sort_values(by='I_mae').iloc[0]
print('best parameter:')
print('=' * 10)
print(br)

# number of new beds at  some days
bed_info = [((d-start_date).days, n) for d, n in bed_info_raw if d <= end_date]

params = Params(
    initial_num_I=br.initial_num_I, 
    initial_num_E=br.initial_num_E,
    initial_num_M=br.initial_num_M,
    alpha=br.alpha, beta=br.beta,
    mu_ei=br.mu_ei, mu_mo=br.mu_mo,
    k_days=int(br.k_days),
    x0_pt=br.x0_pt
)

pkl.dump(
    params,
    open('output/params_after_lockdown.pkl', 'wb')
)

total, delta, increase, trans_data, aux = do_simulation(
    total_days+60,
    bed_info, params, p0_time=start_date
)

I_true_all = df[(df['date'] > start_date)] ['infected'].values
I_pred_all = increase[1:len(I_true_all)+1, STATE.I]


I_pred = increase[1:(len(I_true)+1), STATE.I]
dates = pd.date_range(start_date+timedelta(days=1), end_date+timedelta(days=2))
data = {
    'date': dates,
    'true_I': I_true_all,
    'pred_I': I_pred_all,
    'abs_error': np.abs(I_true_all - I_pred_all),
    'squared_error': np.power(I_true_all - I_pred_all, 2),
    'used_in_fitting': [(d >= start_date) & (d <= end_date) for d in dates]
}

fit_df = pd.DataFrame.from_dict(data)

makedir_if_not_there('output/tbl/parameter_fitting')
fit_df.to_csv('output/tbl/parameter_fitting/daily-data.csv', index=None)

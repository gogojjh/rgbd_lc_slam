import pandas as pd

summary = 'results/summary_tum_icl_loop_pgo.csv'
df = pd.read_csv(summary)
icl = df[df.dataset=='ICL'].copy()
icl['improve'] = icl['ATE_raw_rmse'] - icl['ATE_loop_pgo_rmse']
icl['rel_improve_%'] = 100*icl['improve']/icl['ATE_raw_rmse']
print(icl[['seq','ATE_raw_rmse','ATE_loop_pgo_rmse','rel_improve_%','num_loops','loop_mean_ms']].to_string(index=False))

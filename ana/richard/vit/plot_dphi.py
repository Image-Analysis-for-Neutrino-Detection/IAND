# %%
import pandas as pd

import matplotlib.pyplot as plt
import plotly
import plotly.express as px

#df = pd.read_csv('/fs/ess/PAS2159/neutrino/ana/test_results_dphi.csv')
df = pd.read_csv('/fs/ess/PAS2159/neutrino/ana/pd_results_dphi.csv')


# %%
fig = px.scatter(df,x='label', y='dphi',color='data_type')
fig.show()

# %%
fig = px.scatter(df,x='label', y='output',color='data_type')
fig.show()

# %%

fig = px.histogram(df, x="dphi",color='data_type', barmode="overlay")
fig.show()


# %%
df = pd.read_csv('/fs/ess/PAS2159/neutrino/ana/pd_results_dtheta.csv')
print(len(df),df.columns)
# %%
fig = px.scatter(df,x='label', y='dphi',color='data_type')
fig.show()

fig = px.scatter(df,x='label', y='output',color='data_type')
fig.show()

fig = px.histogram(df, x="dphi",color='data_type', barmode="overlay")
fig.show()


# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from sklearn.linear_model import LinearRegression

# %% [markdown]
# # Linear problem with noise
#
# Toy problem for using pyro. This notebook diplays fitting to a linear problem with noise.
#
# 1) LinearRegressor from sklearn
# 2) Probabilistic model using Pyro
#

# %%
x = np.random.normal(0, 2, 1000)
noise_scale = 0.2
a = 1.2
b = 0.5
# Original a + bx
y = a + b * x + np.random.normal(0, noise_scale, 1000)
plt.scatter(x, y, c="magenta", s=10)

# %%
# Fit model
reg = LinearRegression().fit(x.reshape(-1, 1), y)
lin_reg_score = reg.score(x.reshape(-1, 1), y)
print("Linear regression score: ", lin_reg_score)

# Plot prediction
y_pred_lin_reg = reg.predict(x.reshape(-1, 1))
plt.scatter(x, y, c="magenta", s=10)
plt.plot(x, y_pred_lin_reg, color="blue")
print("Linear regression coefficients: ", reg.coef_, reg.intercept_)

# %%

# %%
# Convert to tensor
x = torch.tensor(x)
y = torch.tensor(y)


# Define model
def lin_pyro_model(x: torch.Tensor, y: torch.Tensor):
    # Model
    # y = a + bx + sigma
    a = pyro.param("a", lambda: torch.randn(()))
    b = pyro.param("b", lambda: torch.randn(()))
    sigma = pyro.param("sigma", lambda: torch.randn(()), constraint=constraints.positive)

    # priors
    mean = a + b * x

    with pyro.plate("data", len(x)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=y)


pyro.render_model(lin_pyro_model, model_args=(x, y), render_distributions=True, render_params=True)

# %%
# %%time
pyro.clear_param_store()

auto_guide = pyro.infer.autoguide.AutoNormal(lin_pyro_model)
adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(lin_pyro_model, auto_guide, adam, elbo)

losses = []
for step in range(10000):  # Consider running for more steps.
    loss = svi.step(x, y)
    losses.append(loss)
    if step % 100 == 0:
        print("Elbo loss: {}".format(loss))

plt.figure(figsize=(5, 2))
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss")


# %%

a = pyro.param("a").item()
b = pyro.param("b").item()
sigma = pyro.param("sigma").item()
y_pred_pyro = a + b * x

sweep_x = torch.arange(x.min(), x.max(), 0.1)
sweep_y = a + b * sweep_x
print("a: ", a)
print("b: ", b)
print("sigma: ", sigma)
print("MAE (pyro): ", torch.abs(y - y_pred_pyro).mean())
print("MAE (sklearn): ", torch.abs(y - y_pred_lin_reg).mean())
# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(x, y, c="magenta", s=10)
plt.plot(x, y_pred_pyro, color="blue", label="Probabilistic Linear Regression")
plt.fill_between(sweep_x, sweep_y - sigma, sweep_y + sigma, color="blue", alpha=0.3, label=r"$\sigma$ noise")
plt.plot(x, y_pred_lin_reg, color="red", label="Linear Regression")
plt.legend()

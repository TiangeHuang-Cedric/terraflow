# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (terraflow-dev)
#     language: python
#     name: terraflow-dev
# ---

# %%
import importlib.metadata as m
m.entry_points(group="terraflow.registrations")

# %%
import terraflow.core as tf

# Use Catalog to load from entry points
catalog = tf.manager.load_info(source="entry_points")

# View both mesh and op tables
print("=== Term Table ===")
print(catalog.view("equterm"))
print("\n=== Equ Aux Table ===")
print(catalog.view("equaux"))
print("\n=== Diag Table ===")
print(catalog.view("diag"))

catalog.load_source()

# %%
m = tf.manager.load_info()

m.load_source()

m.view("equterm")["unique_id"].to_list()

# %%
import polars as pl

df = m.view("equterm")

df = df.filter(pl.col("group_id") == "earth::atm::swe::vif::normal_momentum_equ::explicit::fvm::triskC::odm::1level::nonlinear_coriolis")

df["factory"].to_list()


# %%
# m.to_disk()
m.load_source()

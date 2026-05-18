# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     notebook_metadata_filter: jupytext,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (terraflow-dev)
#     language: python
#     name: terraflow-dev
# ---

# %%
import polars as pl
import terraflow.core as tf
import importlib.metadata as m
m.entry_points(group="terraflow.registrations")

# %%

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

df = m.view("equterm")

df = df.filter(
    pl.col("group_id") == "earth::atm::swe::vif::normal_momentum_equ::explicit::fvm::triskC::scvt::1level::nonlinear_coriolis"
)

df["factory"].to_list()


# %%
# m.to_disk()
m.load_source()

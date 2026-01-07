# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (terraflow-dev)
#     language: python
#     name: terraflow-dev
# ---

# %%
import importlib.metadata as m
m.entry_points(group="terraflow.registrations")

# %%
import terraflow

# Use Catalog to load from entry points
catalog = terraflow.load_info(source="entry_points")

# View both mesh and op tables
print("=== Mesh Table ===")
print(catalog.view("mesh"))
print("\n=== Op Table ===")
print(catalog.view("op"))

catalog.load_source()

# %%
import terraflow

m = terraflow.load_info()

m.load_source()

m.view("equset")["unique_id"].to_list()

# %%
import polars as pl

df = m.view("op")

df = df.filter(pl.col("group_id") == "fvm::triskC::odm::1level::div")

df["factory"].to_list()


# %%
# m.to_disk()
m.load_source()

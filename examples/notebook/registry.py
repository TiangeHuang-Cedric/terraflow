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
# 清除模块缓存，强制重新导入
import sys
for mod in list(sys.modules.keys()):
    if mod.startswith('terraflow'):
        del sys.modules[mod]

# 重新导入并测试
import terraflow

m = terraflow.load_info()
print(f"✅ 成功加载 {m.view('mesh').height} 个 mesh")
print(f"✅ 成功加载 {m.view('op').height} 个 operator")
print("\n" + "="*80)
print("Operator 详情:")
print("="*80)
m.view("op")

# %%
# 测试 mesh_applicability 显示格式修复
import sys
for mod in list(sys.modules.keys()):
    if mod.startswith('terraflow'):
        del sys.modules[mod]

import terraflow

m = terraflow.load_info()
ops = m.view("op")

print("✅ 成功加载", ops.height, "个 operator\n")
print("="*80)
print("检查 mesh_applicability 字段:")
print("="*80)

# 显示前3个operator的关键字段
print(ops.select(["name", "method", "layout", "horizontal_mesh", "group"]).head(3))

print("\n" + "="*80)
print("horizontal_mesh 的唯一值:")
print("="*80)
print(ops["horizontal_mesh"].unique().to_list())

# %%
# m.to_disk()
m.load_source()

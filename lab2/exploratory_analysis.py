from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "autos/imports-85.data"
)
DATA = Path("data")
FIGS = Path("figures")
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
RAW = DATA / "imports-85.data"
if not RAW.exists():
    urlretrieve(URL, RAW)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({
    "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

print("pandas", pd.__version__, "seaborn", sns.__version__, "scipy", scipy.__version__)

headers = [
    "symboling", "normalized-losses", "make", "fuel-type",
    "aspiration", "num-of-doors", "body-style", "drive-wheels",
    "engine-location", "wheel-base", "length", "width", "height",
    "curb-weight", "engine-type", "num-of-cylinders", "engine-size",
    "fuel-system", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price",
]
df = pd.read_csv(RAW, names=headers)
df = df.replace("?", np.nan)
print("raw:", df.shape, "missing price:", int(df["price"].isnull().sum()))

df = df.dropna(subset=["price"]).reset_index(drop=True)
numeric = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]
df[numeric] = df[numeric].astype(float)
for col in numeric[:-1]:
    df[col] = df[col].fillna(df[col].mean())
df["num-of-doors"] = df["num-of-doors"].fillna(df["num-of-doors"].mode()[0])
print("clean:", df.shape, "NaN left:", int(df.isnull().sum().sum()))

print(df.dtypes.value_counts())
print(df[["peak-rpm", "price", "horsepower"]].dtypes)

print(df.describe().T.round(3))

print(df.describe(include=["object", "str"]).T)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
pairs = [("engine-size", PALETTE[0]), ("highway-mpg", PALETTE[1]), ("peak-rpm", PALETTE[2])]
for ax, (col, color) in zip(axes, pairs):
    sns.regplot(x=col, y="price", data=df, ax=ax, color=color,
                scatter_kws={"s": 14, "alpha": 0.7}, line_kws={"color": "#0b0b0b"})
    r = df[[col, "price"]].corr().iloc[0, 1]
    ax.set_title(f"{col}: r = {r:.4f}")
    ax.set_ylim(0,)
fig.tight_layout()
fig.savefig(FIGS / "regplots_price.png", dpi=200)
plt.close("all")
print(df[["engine-size", "highway-mpg", "peak-rpm", "price"]].corr().round(4))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [5, 2, 3]})
order = df.groupby("body-style")["price"].median().sort_values().index
sns.boxplot(x="body-style", y="price", data=df, order=order, ax=axes[0], color=PALETTE[0])
sns.boxplot(x="engine-location", y="price", data=df, ax=axes[1], color=PALETTE[1])
sns.boxplot(x="drive-wheels", y="price", data=df, ax=axes[2], color=PALETTE[2])
for ax in axes:
    ax.tick_params(axis="x", labelsize=9)
fig.tight_layout()
fig.savefig(FIGS / "boxplots_price.png", dpi=200)
plt.close("all")

drive_wheels_counts = df["drive-wheels"].value_counts().to_frame()
drive_wheels_counts = drive_wheels_counts.rename(columns={"count": "value_counts"})
drive_wheels_counts.index.name = "drive-wheels"
engine_loc_counts = df["engine-location"].value_counts().to_frame()
engine_loc_counts = engine_loc_counts.rename(columns={"count": "value_counts"})
engine_loc_counts.index.name = "engine-location"
print(drive_wheels_counts, end="\n\n")
print(engine_loc_counts)

print(df["drive-wheels"].unique())
df_group_one = df[["drive-wheels", "body-style", "price"]]
df_group_one = df_group_one.groupby(["drive-wheels"], as_index=False)["price"].mean()
print(df_group_one)

df_gptest = df[["drive-wheels", "body-style", "price"]]
grouped_test1 = df_gptest.groupby(["drive-wheels", "body-style"], as_index=False)
grouped_test1 = grouped_test1["price"].mean()
grouped_pivot = grouped_test1.pivot(index="drive-wheels", columns="body-style",
                                    values="price")
grouped_pivot = grouped_pivot.fillna(0)
print(grouped_pivot.round(1))

fig, ax = plt.subplots(figsize=(8, 3.6))
im = ax.pcolor(grouped_pivot, cmap="RdBu")
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, grouped_pivot.columns)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, grouped_pivot.index)
ax.set_xlabel("body-style")
ax.set_ylabel("drive-wheels")
ax.grid(False)
for i in range(grouped_pivot.shape[0]):
    for j in range(grouped_pivot.shape[1]):
        ax.text(j + 0.5, i + 0.5, f"{grouped_pivot.iloc[i, j]:.0f}",
                ha="center", va="center", fontsize=9)
fig.colorbar(im)
fig.tight_layout()
fig.savefig(FIGS / "heatmap_drive_body.png", dpi=200)
plt.close("all")

task1 = df[["bore", "stroke", "compression-ratio", "horsepower"]].corr()
print(task1.round(4))

fig, ax = plt.subplots(figsize=(4.8, 3.8))
sns.heatmap(task1, annot=True, fmt=".3f", cmap="RdBu", vmin=-1, vmax=1, square=True, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.tick_params(axis="x", rotation=30, labelsize=9)
ax.tick_params(axis="y", labelsize=9)
fig.tight_layout()
fig.savefig(FIGS / "task1_corr.png", dpi=200)
plt.close("all")

task2 = df[["stroke", "price"]].corr()
r_stroke = task2.loc["stroke", "price"]
print("r(stroke, price) =", round(r_stroke, 4), " r^2 =", round(r_stroke ** 2, 4))
print(task2.round(4))

r_s, p_s = stats.pearsonr(df["stroke"], df["price"])
rho_s, p_rho = stats.spearmanr(df["stroke"], df["price"])
print("Pearson  r =", round(r_s, 4), " p =", round(p_s, 4))
print("Spearman rho =", round(rho_s, 4), " p =", round(p_rho, 4))
fig, ax = plt.subplots(figsize=(6, 3.8))
sns.regplot(x="stroke", y="price", data=df, ax=ax, color=PALETTE[3],
            scatter_kws={"s": 14, "alpha": 0.7}, line_kws={"color": "#0b0b0b"})
ax.set_title(f"stroke: r = {r_s:.4f}")
ax.set_ylim(0,)
fig.tight_layout()
fig.savefig(FIGS / "task2_stroke_price.png", dpi=200)
plt.close("all")

task4 = df[["body-style", "price"]].groupby(["body-style"], as_index=False)["price"].mean()
task4 = task4.sort_values("price", ascending=False).reset_index(drop=True)
task4["count"] = task4["body-style"].map(df["body-style"].value_counts())
print(task4.round(2))

pearson_coef, p_value = stats.pearsonr(df["wheel-base"], df["price"])
print("wheel-base: Pearson =", pearson_coef, " P-value =", p_value)

pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])
print("horsepower: Pearson =", pearson_coef, " P-value =", p_value)

rows = []
for col in df.select_dtypes("number").columns.drop("price"):
    coef, p = stats.pearsonr(df[col], df["price"])
    rows.append((col, coef, p))
pearson_table = pd.DataFrame(rows, columns=["variable", "pearson", "p_value"])
pearson_table = pearson_table.sort_values("pearson", key=abs, ascending=False)
pearson_table = pearson_table.reset_index(drop=True)
pearson_table["p_value"] = pearson_table["p_value"].map(lambda v: f"{v:.3e}")
pearson_table["pearson"] = pearson_table["pearson"].round(4)
print(pearson_table)

fig, ax = plt.subplots(figsize=(8, 4.2))
vals = pearson_table.set_index("variable")["pearson"].sort_values()
colors = [PALETTE[0] if v > 0 else PALETTE[1] for v in vals]
ax.barh(vals.index, vals.values, color=colors)
ax.axvline(0, color="#52514e", linewidth=0.8)
ax.set_xlabel("Pearson r with price")
ax.set_xlim(-1, 1)
ax.tick_params(axis="y", labelsize=9)
fig.tight_layout()
fig.savefig(FIGS / "pearson_price.png", dpi=200)
plt.close("all")

grouped_test2 = df_gptest[["drive-wheels", "price"]].groupby("drive-wheels")
for name in ["fwd", "rwd", "4wd"]:
    g = grouped_test2.get_group(name)["price"]
    print(f"{name}: n = {len(g)}, mean = {g.mean():.2f}, std = {g.std():.2f}")

f_val, p_val = stats.f_oneway(grouped_test2.get_group("fwd")["price"],
                              grouped_test2.get_group("rwd")["price"],
                              grouped_test2.get_group("4wd")["price"])
print("ANOVA fwd, rwd, 4wd: F =", f_val, " P =", p_val)
for a, b in [("fwd", "rwd"), ("4wd", "rwd"), ("4wd", "fwd")]:
    f_val, p_val = stats.f_oneway(grouped_test2.get_group(a)["price"],
                                  grouped_test2.get_group(b)["price"])
    print(f"ANOVA {a} vs {b}: F = {f_val:.4f}  P = {p_val:.4e}")

def anova(factor, target="price"):
    groups = [g[target].values for _, g in df.groupby(factor)]
    f_val, p_val = stats.f_oneway(*groups)
    lev_stat, lev_p = stats.levene(*groups)
    kw_stat, kw_p = stats.kruskal(*groups)
    return {"factor": factor, "groups": len(groups), "min_n": min(len(g) for g in groups),
            "F": round(f_val, 4), "p_value": f"{p_val:.3e}",
            "levene_p": f"{lev_p:.3e}", "kruskal_p": f"{kw_p:.3e}"}

factors = ["drive-wheels", "body-style", "engine-location", "fuel-type",
           "aspiration", "num-of-doors", "num-of-cylinders"]
anova_table = pd.DataFrame([anova(f) for f in factors])
print(anova_table)

pairs = [("fuel-type", "highway-mpg"), ("aspiration", "horsepower")]
extra = []
for factor, target in pairs:
    groups = [g[target].values for _, g in df.groupby(factor)]
    f_val, p_val = stats.f_oneway(*groups)
    extra.append({"factor": factor, "target": target,
                  "F": round(f_val, 4), "p_value": f"{p_val:.3e}"})
    print(df.groupby(factor)[target].agg(["count", "mean", "std"]).round(2), end="\n\n")
print(pd.DataFrame(extra))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
sns.boxplot(x="fuel-type", y="highway-mpg", data=df, ax=axes[0], color=PALETTE[2])
sns.boxplot(x="aspiration", y="horsepower", data=df, ax=axes[1], color=PALETTE[4])
fig.tight_layout()
fig.savefig(FIGS / "anova_extra_boxplots.png", dpi=200)
plt.close("all")

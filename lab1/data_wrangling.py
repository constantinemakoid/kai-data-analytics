from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
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

print("pandas", pd.__version__, "numpy", np.__version__)

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
           "num-of-doors", "body-style", "drive-wheels", "engine-location",
           "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
           "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
           "highway-mpg", "price"]
df = pd.read_csv(RAW, names=headers)
print(df.shape)
print(df.head())

df = df.replace("?", np.nan)
missing_data = df.isnull()
missing_before = missing_data.sum()
missing_before = missing_before[missing_before > 0]
print("missing cells:", int(missing_data.values.sum()))
print(missing_before.to_frame("NaN"))

for column in missing_before.index:
    print(column, missing_data[column].value_counts().to_dict())

numeric_cols = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]
df[numeric_cols] = df[numeric_cols].astype("float")
print(df.dtypes.value_counts())

avg_norm_loss = df["normalized-losses"].mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"] = df["normalized-losses"].fillna(avg_norm_loss)

avg_bore = df["bore"].mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"] = df["bore"].fillna(avg_bore)

stroke_nan_rows = df.index[df["stroke"].isnull()].tolist()
avg_stroke = df["stroke"].mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"] = df["stroke"].fillna(avg_stroke)
print("NaN in stroke:", df["stroke"].isnull().sum())
print(df.loc[stroke_nan_rows, ["make", "engine-type", "bore", "stroke"]])

print("rows without price:", df["price"].isnull().sum(),
      "shape after dropna(price):", df.dropna(subset=["price"]).shape)
fill_report = {}
for column in df.columns[df.isnull().any()]:
    if pd.api.types.is_numeric_dtype(df[column]):
        value = df[column].mean()
        rule = "mean"
    else:
        value = df[column].mode()[0]
        rule = "mode"
    fill_report[column] = (int(df[column].isnull().sum()), rule, value)
    df[column] = df[column].fillna(value)
print("NaN left:", int(df.isnull().sum().sum()))
print(pd.DataFrame(fill_report, index=["NaN", "rule", "value"]).T)

df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["horsepower"]] = df[["horsepower"]].astype("int")
df[["price", "peak-rpm", "bore", "stroke"]] = df[["price", "peak-rpm", "bore", "stroke"]].astype("float")
print(df[["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]].dtypes)

df["city-L/100km"] = 235 / df["city-mpg"]
df["highway-L/100km"] = 235 / df["highway-mpg"]
print(df[["city-mpg", "city-L/100km", "highway-mpg", "highway-L/100km"]].head())

height_raw = df["height"].copy()
h_min, h_max = height_raw.min(), height_raw.max()
h_mean, h_std = height_raw.mean(), height_raw.std()
h_range = np.ptp(height_raw)
print("min", h_min, "max", h_max, "range", round(h_range, 4), "mean", round(h_mean, 4), "std", round(h_std, 4))
df["height"] = height_raw.sub(h_min) / h_range
comparison = pd.DataFrame({
    "height_raw": height_raw,
    "min_max": df["height"],
    "simple_scaling": height_raw / h_max,
    "z_score": height_raw.sub(h_mean) / h_std,
})
print(comparison.describe().loc[["count", "mean", "std", "min", "max"]].round(4))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].hist(height_raw, bins=12, color=PALETTE[0], edgecolor="white")
axes[0].set_title("height, inches")
axes[0].set_xlabel("height")
axes[0].set_ylabel("count")
axes[1].hist(df["height"], bins=12, color=PALETTE[1], edgecolor="white")
axes[1].set_title("height after min max scaling")
axes[1].set_xlabel("normalized height")
axes[1].set_ylabel("count")
fig.tight_layout()
fig.savefig(FIGS / "height_normalization.png", dpi=200)

bins = np.linspace(df["horsepower"].min(), df["horsepower"].max(), 4)
group_names = ["Low", "Medium", "High"]
df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)
print("bins:", bins.round(2).tolist())
print(df["horsepower-binned"].value_counts().reindex(group_names))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].hist(df["horsepower"], bins=bins, color=PALETTE[0], edgecolor="white")
axes[0].set_xticks(bins)
axes[0].set_title("horsepower histogram, 3 bins")
axes[0].set_xlabel("horsepower")
axes[0].set_ylabel("count")
counts = df["horsepower-binned"].value_counts().reindex(group_names)
axes[1].bar(group_names, counts.values, color=PALETTE[0])
for x, y in zip(group_names, counts.values):
    axes[1].text(x, y + 2, str(y), ha="center", color="#0b0b0b")
axes[1].set_title("horsepower bins")
axes[1].set_xlabel("horsepower-binned")
axes[1].set_ylabel("count")
fig.tight_layout()
fig.savefig(FIGS / "horsepower_binning.png", dpi=200)

dummy_variable_2 = pd.get_dummies(df["aspiration"], dtype=int)
dummy_variable_2 = dummy_variable_2.rename(columns={"std": "aspiration-std", "turbo": "aspiration-turbo"})
print(dummy_variable_2.sum().to_dict())
print(dummy_variable_2.head())

df = pd.concat([df, dummy_variable_2], axis=1)
df = df.drop("aspiration", axis=1)
print("aspiration" in df.columns, df.shape)
print(df[["make", "fuel-type", "aspiration-std", "aspiration-turbo", "height", "horsepower-binned"]].head())

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(missing_before.index[::-1], missing_before.values[::-1], color=PALETTE[0])
for y, v in enumerate(missing_before.values[::-1]):
    ax.text(v + 0.5, y, str(v), va="center", color="#0b0b0b")
ax.set_xlabel("NaN count before cleaning")
ax.set_title("Missing values by column")
fig.tight_layout()
fig.savefig(FIGS / "missing_values.png", dpi=200)
print("NaN after cleaning:", int(df.isnull().sum().sum()), "shape:", df.shape)

df.to_csv(DATA / "auto_clean.csv", index=False)
print(df.dtypes.astype(str).value_counts())

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.base import BaseEstimator, RegressorMixin

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

print("pandas", pd.__version__, "seaborn", sns.__version__,
      "scikit-learn", sklearn.__version__)

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

Y = df["price"]
x_train_idx, x_test_idx = train_test_split(df.index, test_size=0.2, random_state=1)
CV = KFold(n_splits=5, shuffle=True, random_state=1)
print("train:", len(x_train_idx), " test:", len(x_test_idx))


def evaluate(name, model, X):
    model.fit(X, Y)
    yhat = model.predict(X)
    r2_in = r2_score(Y, yhat)
    mse_in = mean_squared_error(Y, yhat)
    model.fit(X.loc[x_train_idx], Y.loc[x_train_idx])
    yhat_test = model.predict(X.loc[x_test_idx])
    r2_test = r2_score(Y.loc[x_test_idx], yhat_test)
    mse_test = mean_squared_error(Y.loc[x_test_idx], yhat_test)
    cv = cross_val_score(model, X, Y, cv=CV, scoring="r2").mean()
    model.fit(X, Y)
    return {"model": name, "R2_in": round(r2_in, 4), "MSE_in": f"{mse_in:.4e}",
            "R2_test": round(r2_test, 4), "MSE_test": f"{mse_test:.4e}",
            "R2_cv5": round(cv, 4)}

lm = LinearRegression()
X = df[["engine-size"]]
lm.fit(X, Y)
Yhat = lm.predict(X)
print("intercept a =", lm.intercept_)
print("slope b     =", lm.coef_[0])
print(f"price = {lm.intercept_:.2f} + {lm.coef_[0]:.4f} * engine-size")
print("Yhat[0:5]   =", Yhat[0:5].round(2))
print("R^2         =", lm.score(X, Y))
print("MSE         =", mean_squared_error(Y, Yhat))
print("RMSE        =", mean_squared_error(Y, Yhat) ** 0.5)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.regplot(x="engine-size", y="price", data=df, ax=axes[0], color=PALETTE[0],
            scatter_kws={"s": 14, "alpha": 0.7}, line_kws={"color": "#0b0b0b"})
axes[0].set_ylim(0,)
axes[0].set_title("Regression plot: engine-size vs price")
sns.residplot(x=df["engine-size"], y=df["price"], ax=axes[1], color=PALETTE[1],
              scatter_kws={"s": 14, "alpha": 0.7})
axes[1].set_title("Residual plot: engine-size")
axes[1].set_ylabel("residual")
fig.tight_layout()
fig.savefig(FIGS / "task1_engine_size.png", dpi=200)
plt.close("all")

task1 = pd.DataFrame([
    evaluate("SLR engine-size", LinearRegression(), df[["engine-size"]]),
    evaluate("SLR highway-mpg", LinearRegression(), df[["highway-mpg"]]),
])
resid = Y - Yhat
print("residual mean =", round(resid.mean(), 6), " residual std =", round(resid.std(), 2))
print("share of |residual| > 5000:", round((resid.abs() > 5000).mean(), 4))
print(task1)

lm2 = LinearRegression()
Z = df[["normalized-losses", "highway-mpg"]]
lm2.fit(Z, Y)
Yhat2 = lm2.predict(Z)
print("intercept   =", lm2.intercept_)
print("coefficients =", {k: round(float(v), 4) for k, v in zip(Z.columns, lm2.coef_)})
print(f"price = {lm2.intercept_:.2f} + ({lm2.coef_[0]:.4f}) * normalized-losses"
      f" + ({lm2.coef_[1]:.4f}) * highway-mpg")
print("R^2 =", lm2.score(Z, Y))
print("MSE =", mean_squared_error(Y, Yhat2))
print(df[["normalized-losses", "highway-mpg", "price"]].corr().round(4))

task2 = pd.DataFrame([
    evaluate("SLR highway-mpg", LinearRegression(), df[["highway-mpg"]]),
    evaluate("SLR normalized-losses", LinearRegression(), df[["normalized-losses"]]),
    evaluate("MLR normalized-losses + highway-mpg", LinearRegression(), Z),
])
print(task2)

fig, ax = plt.subplots(figsize=(7, 4))
sns.kdeplot(Y, color=PALETTE[0], label="Actual Value", ax=ax)
sns.kdeplot(Yhat2, color=PALETTE[1], label="Fitted Values", ax=ax)
ax.set_title("MLR normalized-losses + highway-mpg: actual vs fitted")
ax.set_xlabel("price")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "task2_distribution.png", dpi=200)
plt.close("all")

x = df["highway-mpg"]
y = df["price"]
f11 = np.polyfit(x, y, 11)
p11 = np.poly1d(f11)
print("coefficients (x^11 ... x^0):")
print(np.array2string(f11, precision=4, max_line_width=110))
print("R^2 deg 11 =", r2_score(y, p11(x)))
print("MSE deg 11 =", mean_squared_error(y, p11(x)))
f3 = np.polyfit(x, y, 3)
p3 = np.poly1d(f3)
print(p3)
print("R^2 deg 3  =", r2_score(y, p3(x)))
print("MSE deg 3  =", mean_squared_error(y, p3(x)))

pipe11 = Pipeline([("scale", StandardScaler()),
                   ("polynomial", PolynomialFeatures(degree=11, include_bias=False)),
                   ("model", LinearRegression())])
pipe11.fit(df[["highway-mpg"]], y)
yhat_pipe = pipe11.predict(df[["highway-mpg"]])
print("Pipeline deg 11 R^2 =", r2_score(y, yhat_pipe))
print("Pipeline deg 11 MSE =", mean_squared_error(y, yhat_pipe))
print("max |polyfit - pipeline| =", np.abs(p11(x) - yhat_pipe).max().round(2))

def plot_polly(ax, model, xs, ys, name, color):
    x_new = np.linspace(15, 55, 300)
    ax.plot(xs, ys, ".", color="#52514e", alpha=0.7)
    ax.plot(x_new, model(x_new), "-", color=color)
    ax.set_title(name)
    ax.set_xlabel("highway-mpg")
    ax.set_ylabel("price")
    ax.set_ylim(-10000, 60000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_polly(axes[0], p3, x, y, "Polynomial degree 3", PALETTE[2])
plot_polly(axes[1], p11, x, y, "Polynomial degree 11", PALETTE[1])
fig.tight_layout()
fig.savefig(FIGS / "task3_poly.png", dpi=200)
plt.close("all")

rows = []
for d in range(1, 12):
    pipe = Pipeline([("scale", StandardScaler()),
                     ("polynomial", PolynomialFeatures(degree=d, include_bias=False)),
                     ("model", LinearRegression())])
    res = evaluate(f"degree {d}", pipe, df[["highway-mpg"]])
    rows.append(res)
degree_table = pd.DataFrame(rows)
print(degree_table)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 12), degree_table["R2_in"], "o-", color=PALETTE[0], label="R^2 in-sample")
ax.plot(range(1, 12), degree_table["R2_test"], "s-", color=PALETTE[1], label="R^2 test")
ax.plot(range(1, 12), degree_table["R2_cv5"], "^-", color=PALETTE[2], label="R^2 5-fold CV")
ax.set_xlabel("polynomial degree (highway-mpg)")
ax.set_ylabel("R^2")
ax.set_xticks(range(1, 12))
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "task3_degree_r2.png", dpi=200)
plt.close("all")

features = ["horsepower", "curb-weight", "engine-size", "highway-mpg"]
print(df[features + ["price"]].corr()["price"].round(4))


class Poly1D(RegressorMixin, BaseEstimator):
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y):
        self.poly_ = np.poly1d(np.polyfit(np.asarray(X)[:, 0], y, self.degree))
        return self

    def predict(self, X):
        return self.poly_(np.asarray(X)[:, 0])


models = [
    ("SLR horsepower", LinearRegression(), df[["horsepower"]]),
    ("MLR 4 features", LinearRegression(), df[features]),
    ("Poly deg 2 horsepower", Poly1D(2), df[["horsepower"]]),
    ("Poly deg 3 horsepower", Poly1D(3), df[["horsepower"]]),
    ("Pipeline deg 2, 4 features",
     Pipeline([("scale", StandardScaler()),
               ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
               ("model", LinearRegression())]), df[features]),
]
task4 = pd.DataFrame([evaluate(n, m, X) for n, m, X in models])
print(task4)

mlr = LinearRegression().fit(df[features], Y)
print("intercept =", round(mlr.intercept_, 2))
print("coefficients =", {k: round(float(v), 4) for k, v in zip(features, mlr.coef_)})
Y_hat_mlr = mlr.predict(df[features])
fig, ax = plt.subplots(figsize=(7, 4))
sns.kdeplot(Y, color=PALETTE[0], label="Actual Value", ax=ax)
sns.kdeplot(Y_hat_mlr, color=PALETTE[1], label="Fitted Values", ax=ax)
ax.set_title("MLR 4 features: actual vs fitted")
ax.set_xlabel("price")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "task4_distribution.png", dpi=200)
plt.close("all")

new_cars = pd.DataFrame({
    "horsepower": [70, 110, 180],
    "curb-weight": [2000, 2600, 3300],
    "engine-size": [90, 130, 200],
    "highway-mpg": [38, 30, 22],
})
new_cars["price_pred_mlr"] = mlr.predict(new_cars[features]).round(0)
new_cars["price_pred_slr_engine"] = lm.predict(new_cars[["engine-size"]]).round(0)
print(new_cars)

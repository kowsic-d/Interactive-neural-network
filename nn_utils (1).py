"""nn_utils.py — helper functions for NeuroLab.

Contains: synthetic datasets, feature engineering, train/test preparation,
evaluation metrics, and every Plotly figure used by the Streamlit app.
"""
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
DATASETS = ["Moons", "Checkerboard", "Pinwheel", "Sine Wave"]

# Optional engineered inputs the network can receive (x1, x2 are standardised).
FEATURES = {
    "x1": lambda X: X[:, 0],
    "x2": lambda X: X[:, 1],
    "x1²": lambda X: X[:, 0] ** 2,
    "x2²": lambda X: X[:, 1] ** 2,
    "x1·x2": lambda X: X[:, 0] * X[:, 1],
    "sin(x1)": lambda X: np.sin(X[:, 0]),
    "sin(x2)": lambda X: np.sin(X[:, 1]),
}

# One-click starting configurations shown in the sidebar.
PRESETS = {
    "Balanced starter": dict(widths="8, 8", act="tanh", lr=0.03, epochs=120, dropout=0.0, l2=0.0),
    "Too small (underfit)": dict(widths="2", act="tanh", lr=0.03, epochs=120, dropout=0.0, l2=0.0),
    "Wide & shallow": dict(widths="32", act="relu", lr=0.01, epochs=120, dropout=0.0, l2=0.0),
    "Deep & narrow": dict(widths="6, 6, 6, 6", act="tanh", lr=0.03, epochs=160, dropout=0.0, l2=0.0),
    "Big net (overfit risk)": dict(widths="64, 64, 64", act="relu", lr=0.01, epochs=240, dropout=0.0, l2=0.0),
    "Big net + regularised": dict(widths="64, 64, 64", act="relu", lr=0.01, epochs=240, dropout=0.3, l2=0.001),
}

CLS0, CLS1 = "#0ea5a4", "#f97316"           # teal / orange
PROB_SCALE = [[0.0, CLS0], [0.5, "#f8fafc"], [1.0, CLS1]]
INK = "#111827"


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def make_dataset(kind, n, noise, seed):
    """Return (X, y) for a 2-D binary classification problem."""
    rng = np.random.default_rng(seed)

    if kind == "Moons":
        n1 = n // 2
        n2 = n - n1
        t1 = rng.uniform(0, np.pi, n1)
        t2 = rng.uniform(0, np.pi, n2)
        upper = np.c_[np.cos(t1), np.sin(t1)]
        lower = np.c_[1 - np.cos(t2), 0.45 - np.sin(t2)]
        X = np.vstack([upper, lower]) + rng.normal(0, 0.05 + noise * 0.45, (n, 2))
        y = np.r_[np.zeros(n1), np.ones(n2)]

    elif kind == "Checkerboard":
        X = rng.uniform(-2, 2, (n, 2))
        y = (np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2
        X = X + rng.normal(0, noise * 0.25, X.shape)

    elif kind == "Pinwheel":
        arm = rng.integers(0, 4, n)
        r = rng.uniform(0.15, 2.0, n)
        ang = arm * (np.pi / 2) + 0.9 * r + rng.normal(0, noise * 0.6, n)
        X = np.c_[r * np.cos(ang), r * np.sin(ang)]
        y = arm % 2

    elif kind == "Sine Wave":
        X = rng.uniform(-2.5, 2.5, (n, 2))
        y = (X[:, 1] > 1.1 * np.sin(2.0 * X[:, 0])).astype(int)
        X = X + rng.normal(0, noise * 0.3, X.shape)

    else:
        raise ValueError(f"Unknown dataset: {kind}")

    return X, np.asarray(y).astype(int)


def split_and_scale(X, y, test_frac, seed):
    """Stratified split; scaler is fitted on the training part only."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte


def expand(X, names):
    """Turn 2-D coordinates into the network's input features."""
    return np.column_stack([FEATURES[n](X) for n in names]).astype("float32")


def make_grid(X, res=110, pad=0.5):
    """Regular grid over the data; returns (gx, gy, coords[N,2]) in row-major order."""
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, res)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, res)
    xx, yy = np.meshgrid(gx, gy)
    return gx, gy, np.c_[xx.ravel(), yy.ravel()]


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, prob):
    """Return (metrics dict, confusion matrix, (fpr, tpr))."""
    prob = np.clip(np.asarray(prob).ravel(), 1e-7, 1 - 1e-7)
    pred = (prob > 0.5).astype(int)
    two_classes = len(np.unique(y_true)) > 1
    scores = dict(
        accuracy=accuracy_score(y_true, pred),
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        f1=f1_score(y_true, pred, zero_division=0),
        auc=roc_auc_score(y_true, prob) if two_classes else float("nan"),
        log_loss=log_loss(y_true, prob, labels=[0, 1]),
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, prob) if two_classes else (np.array([0, 1]), np.array([0, 1]), None)
    return scores, cm, (fpr, tpr)


def diagnose(train_acc, test_acc):
    """Plain-language verdict: (streamlit level, message)."""
    if train_acc - test_acc > 0.07:
        return "warning", ("**Overfitting** — the model memorises the training set. "
                           "Try dropout, L2, fewer neurons, more samples or early stopping.")
    if train_acc < 0.85:
        return "warning", ("**Underfitting** — the model can't capture the pattern yet. "
                           "Try wider/deeper layers, extra features, a higher learning rate or more epochs.")
    return "success", "**Healthy fit** — train and test scores are both high and close together."


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def _style(fig, title, height=430):
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15)),
        template="plotly_white", height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
    )
    return fig


def _point_traces(Xtr, ytr, Xte, yte):
    traces = []
    for X, y, part in ((Xtr, ytr, "Train"), (Xte, yte, "Test")):
        for cls, colour in ((0, CLS0), (1, CLS1)):
            m = y == cls
            is_test = part == "Test"
            traces.append(go.Scatter(
                x=X[m, 0], y=X[m, 1], mode="markers", name=f"{part} · class {cls}",
                marker=dict(
                    color=colour, size=9 if is_test else 6,
                    symbol="diamond" if is_test else "circle",
                    line=dict(color=INK if is_test else "white", width=1.4 if is_test else 0.5),
                ),
            ))
    return traces


def fig_data(Xtr, ytr, Xte, yte, title):
    fig = go.Figure(_point_traces(Xtr, ytr, Xte, yte))
    fig.update_xaxes(title="x1 (standardised)")
    fig.update_yaxes(title="x2 (standardised)")
    return _style(fig, title)


def fig_boundary(gx, gy, Z, Xtr, ytr, Xte, yte, title, point=None):
    """Probability surface + 0.5 contour + data points."""
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=gx, y=gy, z=Z, zmin=0, zmax=1, colorscale=PROB_SCALE, opacity=0.8,
        contours=dict(coloring="heatmap"), line=dict(width=0),
        colorbar=dict(title="P(class 1)", thickness=10, len=0.8), hoverinfo="skip",
    ))
    fig.add_trace(go.Contour(
        x=gx, y=gy, z=Z, showscale=False, hoverinfo="skip", name="0.5 boundary",
        contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
        line=dict(width=2.5, color=INK),
    ))
    for t in _point_traces(Xtr, ytr, Xte, yte):
        fig.add_trace(t)
    if point is not None:
        fig.add_trace(go.Scatter(
            x=[point[0]], y=[point[1]], mode="markers", name="Your point",
            marker=dict(symbol="star", size=18, color="#facc15", line=dict(color=INK, width=1.5)),
        ))
    fig.update_xaxes(title="x1 (standardised)", range=[gx.min(), gx.max()])
    fig.update_yaxes(title="x2 (standardised)", range=[gy.min(), gy.max()])
    return _style(fig, title)


def fig_history(hist):
    """Loss and accuracy curves side by side."""
    ep = list(range(1, len(hist["loss"]) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss (binary cross-entropy)", "Accuracy"))
    for key, name, colour, dash in (
        ("loss", "Train", CLS0, "solid"), ("val_loss", "Test", CLS1, "solid"),
    ):
        fig.add_trace(go.Scatter(x=ep, y=hist[key], name=name, line=dict(color=colour, dash=dash)), row=1, col=1)
    for key, name, colour in (("accuracy", "Train", CLS0), ("val_accuracy", "Test", CLS1)):
        fig.add_trace(go.Scatter(x=ep, y=hist[key], name=name, line=dict(color=colour), showlegend=False), row=1, col=2)
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    return _style(fig, "Learning curves")


def fig_roc(fpr, tpr, auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {auc:.3f})",
                             line=dict(color=CLS1, width=3), fill="tozeroy",
                             fillcolor="rgba(249,115,22,0.12)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color="#94a3b8", dash="dash")))
    fig.update_xaxes(title="False-positive rate")
    fig.update_yaxes(title="True-positive rate")
    return _style(fig, "ROC curve (test set)", height=360)


def fig_confusion(cm):
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], text=cm,
        texttemplate="%{text}", textfont=dict(size=22), colorscale="Teal", showscale=False,
    ))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Confusion matrix (test set)", height=360)


def fig_neuron_maps(acts, gx, gy, max_units=16, cols=4):
    """One small heat-map per neuron: what does this neuron 'see' across the plane?"""
    k = min(acts.shape[1], max_units)
    rows = math.ceil(k / cols)
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[f"neuron {i + 1}" for i in range(k)],
        horizontal_spacing=0.03, vertical_spacing=0.09,
    )
    for i in range(k):
        r, c = divmod(i, cols)
        fig.add_trace(go.Heatmap(
            z=acts[:, i].reshape(len(gy), len(gx)), x=gx, y=gy,
            colorscale="Viridis", showscale=False,
        ), row=r + 1, col=c + 1)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_annotations(font_size=11)
    fig.update_layout(template="plotly_white", height=200 * rows + 40,
                      margin=dict(l=5, r=5, t=35, b=5))
    return fig

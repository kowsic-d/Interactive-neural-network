"""NeuroLab — a hands-on neural-network sandbox.

Built with TensorFlow/Keras + Streamlit + Plotly.
Run locally:  streamlit run app.py
"""
import json
import math
import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

import nn_utils as U

st.set_page_config(page_title="NeuroLab · Neural Network Sandbox", page_icon="🧪", layout="wide")

# --------------------------------------------------------------------------- #
# Defaults & session state                                                    #
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    dataset="Moons", n=500, noise=0.2, test_frac=0.25, feats=["x1", "x2"],
    widths="8, 8", act="tanh", dropout=0.0, l2=0.0,
    opt="Adam", lr=0.03, sched="Constant", epochs=120, batch=32,
    early=False, patience=15, seed=7,
)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("log", [])
st.session_state.setdefault("run", None)
st.session_state.setdefault("run_counter", 0)

OPTIMIZERS = {
    "Adam": lambda lr: tf.keras.optimizers.Adam(lr),
    "Nadam": lambda lr: tf.keras.optimizers.Nadam(lr),
    "RMSprop": lambda lr: tf.keras.optimizers.RMSprop(lr),
    "SGD + momentum": lambda lr: tf.keras.optimizers.SGD(lr, momentum=0.9),
    "Adagrad": lambda lr: tf.keras.optimizers.Adagrad(lr),
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def apply_preset():
    p = U.PRESETS[st.session_state["preset"]]
    for key, value in p.items():
        st.session_state[key] = value


def parse_widths(text):
    """'8, 8, 4' -> [8, 8, 4]; returns None when invalid."""
    try:
        w = [int(t) for t in text.replace(";", ",").split(",") if t.strip()]
    except ValueError:
        return None
    if not w or len(w) > 6 or any(x < 1 or x > 64 for x in w):
        return None
    return w


def make_lr(cfg, steps_per_epoch):
    base = cfg["lr"]
    if cfg["sched"] == "Exponential decay":
        return tf.keras.optimizers.schedules.ExponentialDecay(
            base, decay_steps=max(1, steps_per_epoch * cfg["epochs"] // 4), decay_rate=0.5)
    if cfg["sched"] == "Cosine decay":
        return tf.keras.optimizers.schedules.CosineDecay(
            base, decay_steps=max(1, steps_per_epoch * cfg["epochs"]), alpha=0.05)
    return base


def build_model(n_inputs, cfg, steps_per_epoch):
    tf.keras.utils.set_random_seed(cfg["seed"])
    reg = tf.keras.regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    model = tf.keras.Sequential(name="neurolab_net")
    model.add(tf.keras.Input(shape=(n_inputs,)))
    for i, width in enumerate(cfg["widths"], start=1):
        model.add(tf.keras.layers.Dense(width, activation=cfg["act"],
                                        kernel_regularizer=reg, name=f"hidden_{i}"))
        if cfg["dropout"] > 0:
            model.add(tf.keras.layers.Dropout(cfg["dropout"], name=f"dropout_{i}"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output"))
    optimizer = OPTIMIZERS[cfg["opt"]](make_lr(cfg, steps_per_epoch))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def layer_activations(model, layer_name, feats, coords):
    """Run the grid through the network and stop at the requested layer."""
    h = tf.constant(U.expand(coords, feats))
    for layer in model.layers:
        h = layer(h, training=False)
        if layer.name == layer_name:
            break
    return h.numpy()


class Painter(tf.keras.callbacks.Callback):
    """Streams progress, decision surface and curves into the page while training."""

    def __init__(self, epochs, every, bar, slot_b, slot_c, grid_fn, gx, gy, pts):
        super().__init__()
        self.epochs, self.every, self.bar = epochs, every, bar
        self.slot_b, self.slot_c = slot_b, slot_c
        self.grid_fn, self.gx, self.gy, self.pts = grid_fn, gx, gy, pts
        self.hist = {k: [] for k in ("loss", "accuracy", "val_loss", "val_accuracy")}
        self.last_painted = 0

    def paint(self, n):
        Z = self.grid_fn(self.model)
        self.slot_b.plotly_chart(
            U.fig_boundary(self.gx, self.gy, Z, *self.pts, f"Decision surface · epoch {n}"))
        self.slot_c.plotly_chart(U.fig_history(self.hist))
        self.last_painted = n

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.hist:
            self.hist[k].append(float(logs.get(k, np.nan)))
        n = epoch + 1
        self.bar.progress(
            min(n / self.epochs, 1.0),
            text=f"Epoch {n}/{self.epochs} · train {logs.get('accuracy', 0):.1%} · "
                 f"test {logs.get('val_accuracy', 0):.1%}")
        if n == 1 or n % self.every == 0:
            self.paint(n)

    def on_train_end(self, logs=None):
        n = len(self.hist["loss"])
        if n and n != self.last_painted:
            self.paint(n)


# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.title("🧪 NeuroLab")

sb.subheader("Quick start")
sb.selectbox("Preset architecture", list(U.PRESETS), key="preset")
sb.button("Apply preset", on_click=apply_preset)

sb.subheader("① Data")
sb.selectbox("Dataset", U.DATASETS, key="dataset")
sb.slider("Samples", 100, 1500, step=50, key="n")
sb.slider("Noise level", 0.0, 0.6, step=0.05, key="noise")
sb.slider("Test fraction", 0.1, 0.5, step=0.05, key="test_frac")
sb.multiselect("Input features", list(U.FEATURES), key="feats",
               help="x1 and x2 are the raw coordinates; the rest are engineered inputs.")

sb.subheader("② Architecture")
sb.text_input("Hidden layer widths", key="widths",
              help="Comma-separated neurons per layer, e.g. `16, 8, 4` (max 6 layers, 1–64 each).")
sb.selectbox("Activation", ["tanh", "relu", "elu", "selu", "swish", "sigmoid"], key="act")
sb.slider("Dropout rate", 0.0, 0.5, step=0.05, key="dropout")
sb.select_slider("L2 penalty", options=[0.0, 0.0001, 0.001, 0.01, 0.05], key="l2")

sb.subheader("③ Optimisation")
sb.selectbox("Optimizer", list(OPTIMIZERS), key="opt")
sb.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], key="lr")
sb.selectbox("Learning-rate schedule", ["Constant", "Exponential decay", "Cosine decay"], key="sched")
sb.slider("Max epochs", 20, 400, step=20, key="epochs")
sb.select_slider("Batch size", options=[8, 16, 32, 64, 128], key="batch")
sb.checkbox("Early stopping (monitor test loss)", key="early")
if st.session_state["early"]:
    sb.slider("Patience (epochs)", 5, 50, step=5, key="patience")
sb.number_input("Random seed", 0, 9999, step=1, key="seed")

widths = parse_widths(st.session_state["widths"])
feats = list(st.session_state["feats"])
inputs_ok = widths is not None and len(feats) > 0
if widths is None:
    sb.error("Widths must be 1–6 whole numbers between 1 and 64, e.g. `8, 8`.")
if not feats:
    sb.error("Select at least one input feature.")

cfg = {k: st.session_state[k] for k in DEFAULTS}
cfg["widths"], cfg["feats"] = widths, feats

# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
X, y = U.make_dataset(cfg["dataset"], cfg["n"], cfg["noise"], cfg["seed"])
Xtr, Xte, ytr, yte = U.split_and_scale(X, y, cfg["test_frac"], cfg["seed"])
gx, gy, coords = U.make_grid(np.vstack([Xtr, Xte]))
pts = (Xtr, ytr, Xte, yte)

st.title("🧪 NeuroLab — Neural Network Sandbox")
st.caption("Design a network, train it with TensorFlow/Keras, and watch it carve up the plane in real time.")

tab_lab, tab_log, tab_guide = st.tabs(["🔬 Lab", "🗂️ Run history", "📖 Guide"])

# --------------------------------------------------------------------------- #
# LAB TAB                                                                     #
# --------------------------------------------------------------------------- #
with tab_lab:
    top_l, top_r = st.columns([1, 3])
    go_train = top_l.button("🚀 Train model", type="primary", disabled=not inputs_ok)
    status = top_r.empty()
    left, right = st.columns(2)
    slot_b, slot_c = left.empty(), right.empty()
    results = st.container()

    if go_train:
        F_tr, F_te = U.expand(Xtr, feats), U.expand(Xte, feats)
        steps = math.ceil(len(Xtr) / cfg["batch"])
        model = build_model(len(feats), cfg, steps)

        def grid_fn(m):
            return m(U.expand(coords, feats), training=False).numpy().reshape(len(gy), len(gx))

        bar = status.progress(0.0, text="Warming up…")
        painter = Painter(cfg["epochs"], max(1, cfg["epochs"] // 12), bar,
                          slot_b, slot_c, grid_fn, gx, gy, pts)
        callbacks = [painter]
        if cfg["early"]:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg["patience"], restore_best_weights=True))

        t0 = time.time()
        model.fit(F_tr, ytr, validation_data=(F_te, yte), epochs=cfg["epochs"],
                  batch_size=cfg["batch"], verbose=0, callbacks=callbacks)
        secs = time.time() - t0

        m_tr, _, _ = U.compute_metrics(ytr, model.predict(F_tr, verbose=0))
        m_te, cm, roc = U.compute_metrics(yte, model.predict(F_te, verbose=0))
        ran = len(painter.hist["loss"])

        st.session_state["run_counter"] += 1
        rid = st.session_state["run_counter"]
        st.session_state["run"] = dict(
            id=rid, cfg=dict(cfg), model=model, hist=painter.hist, secs=secs, epochs_run=ran,
            m_tr=m_tr, m_te=m_te, cm=cm, roc=roc, pts=pts, gx=gx, gy=gy, coords=coords,
            feats=list(feats), params=model.count_params(),
        )
        st.session_state["log"].append(dict(
            run=rid, dataset=cfg["dataset"], features=", ".join(feats), widths=cfg["widths"] and "-".join(map(str, cfg["widths"])),
            activation=cfg["act"], optimizer=cfg["opt"], lr=cfg["lr"], schedule=cfg["sched"],
            dropout=cfg["dropout"], l2=cfg["l2"], epochs=ran, params=model.count_params(),
            train_acc=round(m_tr["accuracy"], 4), test_acc=round(m_te["accuracy"], 4),
            test_f1=round(m_te["f1"], 4), test_auc=round(m_te["auc"], 4),
        ))
        status.success(f"Run #{rid} finished — {ran} epochs in {secs:.1f} s")

    run = st.session_state["run"]

    if run is None:
        slot_b.plotly_chart(U.fig_data(*pts, f"{cfg['dataset']} · circles = train, diamonds = test"))
        slot_c.info("👈 Pick a dataset and architecture in the sidebar (or apply a preset), "
                    "then press **Train model**. The decision surface and learning curves update live.")
    else:
        r_pts, rf = run["pts"], run["feats"]
        model = run["model"]
        if not go_train:
            Z = model(U.expand(run["coords"], rf), training=False).numpy().reshape(len(run["gy"]), len(run["gx"]))
            slot_b.plotly_chart(U.fig_boundary(run["gx"], run["gy"], Z, *r_pts,
                                               f"Decision surface · run #{run['id']}"))
            slot_c.plotly_chart(U.fig_history(run["hist"]))
            if run["cfg"] != cfg:
                status.warning("Sidebar settings changed since the last run — press **Train model** "
                               "to apply them. Showing the previous result.")

        with results:
            m_tr, m_te = run["m_tr"], run["m_te"]
            c = st.columns(5)
            c[0].metric("Train accuracy", f"{m_tr['accuracy']:.1%}")
            c[1].metric("Test accuracy", f"{m_te['accuracy']:.1%}",
                        f"{(m_te['accuracy'] - m_tr['accuracy']) * 100:+.1f} pts vs train")
            c[2].metric("Test F1", f"{m_te['f1']:.3f}")
            c[3].metric("Test AUC", f"{m_te['auc']:.3f}")
            c[4].metric("Parameters", f"{run['params']:,}")

            level, msg = U.diagnose(m_tr["accuracy"], m_te["accuracy"])
            (st.success if level == "success" else st.warning)(msg)

            t_maps, t_scores, t_probe = st.tabs(["🧩 Neuron maps", "📊 Scores", "🎯 Probe a point"])

            with t_maps:
                hidden_names = [l.name for l in model.layers if l.name.startswith("hidden_")]
                layer_name = st.selectbox("Layer to inspect", hidden_names, key=f"map_{run['id']}")
                acts = layer_activations(model, layer_name, rf, run["coords"])
                st.plotly_chart(U.fig_neuron_maps(acts, run["gx"], run["gy"]))
                st.caption("Each tile shows one neuron's output across the plane. "
                           "Later layers combine these simple shapes into the final boundary "
                           f"(showing up to 16 of {acts.shape[1]} neurons).")
                with st.expander("Keras model summary"):
                    lines = []
                    model.summary(print_fn=lambda s, **k: lines.append(s))
                    st.code("\n".join(lines))

            with t_scores:
                a, b = st.columns(2)
                a.plotly_chart(U.fig_confusion(run["cm"]))
                b.plotly_chart(U.fig_roc(*run["roc"], m_te["auc"]))
                table = pd.DataFrame({"Train": m_tr, "Test": m_te}).round(3)
                st.dataframe(table)

            with t_probe:
                p1, p2 = st.columns(2)
                px = p1.slider("x1 (standardised)", -3.0, 3.0, 0.0, 0.1, key="probe_x")
                py = p2.slider("x2 (standardised)", -3.0, 3.0, 0.0, 0.1, key="probe_y")
                prob = float(model(U.expand(np.array([[px, py]]), rf), training=False).numpy()[0, 0])
                st.metric("Predicted class", f"{int(prob > 0.5)}", f"P(class 1) = {prob:.1%}")
                Z = model(U.expand(run["coords"], rf), training=False).numpy().reshape(len(run["gy"]), len(run["gx"]))
                st.plotly_chart(U.fig_boundary(run["gx"], run["gy"], Z, *r_pts,
                                               "Where does your point land?", point=(px, py)))

# --------------------------------------------------------------------------- #
# HISTORY TAB                                                                 #
# --------------------------------------------------------------------------- #
with tab_log:
    log = st.session_state["log"]
    if not log:
        st.info("Every training run is logged here so you can compare experiments.")
    else:
        df = pd.DataFrame(log)
        st.dataframe(df)
        st.bar_chart(df.set_index("run")[["train_acc", "test_acc"]])
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Download log (CSV)", df.to_csv(index=False), "neurolab_runs.csv", "text/csv")
        last = st.session_state["run"]
        if last:
            d2.download_button("⬇️ Download last config (JSON)", json.dumps(last["cfg"], indent=2),
                               f"neurolab_run_{last['id']}.json", "application/json")
        d3.button("🗑️ Clear history", on_click=lambda: st.session_state.update(log=[]))

# --------------------------------------------------------------------------- #
# GUIDE TAB                                                                   #
# --------------------------------------------------------------------------- #
with tab_guide:
    st.markdown("""
### Workflow
1. **Choose data** — four 2-D binary problems with adjustable noise.
2. **Design the network** — type layer widths like `16, 8`, pick an activation, add dropout or L2.
3. **Train** — Keras minimises binary cross-entropy; the page repaints as epochs complete.
4. **Diagnose** — compare train vs test scores, then inspect neurons, ROC and the confusion matrix.
5. **Iterate** — the Run history tab keeps every experiment side by side.

### Knobs and what they teach
| Control | Effect | Experiment |
|---|---|---|
| Layer widths | Model capacity | `2` vs `16, 16` on Sine Wave |
| Input features | Easier problem for the same network | Add `sin(x1)` or `x1·x2` on Checkerboard |
| Activation | Shape of each neuron's response | tanh vs relu on Moons |
| Dropout / L2 | Fights overfitting | Compare the two *Big net* presets |
| Optimizer & LR schedule | Speed and stability | Adam vs SGD at learning rate 0.3 |
| Early stopping | Stops when test loss stalls | Set 400 epochs with patience 15 |

### Reading the diagnosis
- **Train ≫ test** → overfitting. **Both low** → underfitting. **Both high and close** → healthy fit.
- **AUC** measures ranking quality across all thresholds; **F1** balances precision and recall at 0.5.
""")
    st.caption("Stack: TensorFlow/Keras · Streamlit · Plotly · scikit-learn · NumPy · pandas")

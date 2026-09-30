# 🧪 NeuroLab — Neural Network Sandbox

An interactive, browser-based neural-network lab built with **TensorFlow/Keras**, **Streamlit** and **Plotly**.
Design a network, train it, and watch the decision surface and learning curves update live.

## What you can do
- Choose from four datasets: **Moons, Checkerboard, Pinwheel, Sine Wave** (adjustable samples, noise, test split)
- Feed the network engineered features (`x1²`, `x1·x2`, `sin(x1)` …) as well as raw coordinates
- Type any architecture, e.g. `16, 8, 4` (up to 6 hidden layers)
- Pick activation (tanh, relu, elu, selu, swish, sigmoid), dropout and L2 regularisation
- Choose optimizer (Adam, Nadam, RMSprop, SGD + momentum, Adagrad), learning rate and LR schedule (constant / exponential / cosine)
- Enable early stopping
- Apply one-click **preset architectures** (underfit, overfit, regularised …)
- Live probability surface with 0.5 boundary + loss/accuracy curves
- **Neuron maps** — see what each hidden neuron responds to
- Accuracy, precision, recall, F1, AUC, confusion matrix and ROC curve
- Probe any point and see the predicted probability
- **Run history** table with CSV / JSON export for comparing experiments

## Project structure
```
app.py            # Streamlit interface + Keras training loop
nn_utils.py       # datasets, features, metrics and Plotly figures
requirements.txt  # dependencies
README.md
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push the four files to a public GitHub repository.
2. Go to <https://share.streamlit.io>, click **New app**, select the repo and set the main file to `app.py`.
3. Deploy and copy the public URL.

> Tip: if the cloud build fails on TensorFlow, add a `runtime.txt` containing `python-3.11`.

## Suggested demo flow (5 min)
1. Explain the goal: a network learns a boundary between two classes.
2. Show **Moons** and train the *Balanced starter* preset — point out the live surface.
3. Apply *Too small (underfit)* and compare the diagnosis.
4. Switch to **Sine Wave** or **Checkerboard**, try deeper layers or extra features.
5. Train *Big net (overfit risk)* then *Big net + regularised* and compare in **Run history**.
6. Open **Neuron maps** and **Scores**, then probe a custom point.

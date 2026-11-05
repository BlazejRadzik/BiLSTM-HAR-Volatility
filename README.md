# DeepVol-HAR

### A Comparative Study of Bi-LSTM Neural Networks and Econometric HAR Models in Realized Volatility Forecasting — The Role of Market Liquidity and Maturity

> **Author:** Błażej Radzik
> **ISBN:** 978-83-67620-66-6 p.43
> **Link:** https://books.google.pl/books?id=x9QAEgAAQBAJ&newbks=0&printsec=frontcover&pg=PA99&dq=Analiza+ilo%C5%9Bciowa+wybranych+problem%C3%B3w+z+zakresu+ekonomii+i+finans%C3%B3w&hl=pl&source=newbks_fb&redir_esc=y#v=onepage&q=Analiza%20ilo%C5%9Bciowa%20wybranych%20problem%C3%B3w%20z%20zakresu%20ekonomii%20i%20finans%C3%B3w&f=false

---

## 1. Abstract

Forecasting realized volatility is a cornerstone of quantitative risk management. In an era dominated by machine learning, it has become almost a default reflex to expect deep neural networks to outperform classical econometric tools such as the **HAR (Heterogeneous Autoregressive)** model.

This study challenges that reflex. It evaluates a hybrid **DL-HAR** architecture — a **Bidirectional LSTM with an Attention mechanism**, fed both raw price-derived features *and* the canonical HAR components — against a strict OLS HAR benchmark. The horse race is run, **out-of-sample**, on four ETFs spanning the full liquidity / efficiency spectrum:

| Ticker | Market | Profile |
|--------|--------|---------|
| **SPY** | United States (S&P 500) | Extremely liquid, highly informationally efficient |
| **EWG** | Germany | Liquid, developed, mid-efficiency |
| **EPOL** | Poland | Emerging, lower liquidity, low signal-to-noise |
| **EEM** | Emerging Markets basket | Noisy, structurally inefficient |

The central hypothesis was straightforward: **the deeper the market, the bigger the edge of deep learning.** The results turn out to be considerably more interesting than that.

---

## 2. Results

All forecasts are produced under a **daily walk-forward** protocol with model-specific re-training every trading day, **QLIKE** as the loss, and a **Diebold–Mariano** test for statistical significance. Lower QLIKE is better.

| Market | QLIKE HAR | QLIKE DL | DM stat | p-value | Verdict |
|--------|-----------|----------|---------|---------|---------|
| **SPY**  | 0.4183 | 0.4206 | −0.10 | 0.9191 | No significant difference |
| **EWG**  | 0.4163 | **0.3728** | **+3.99** | **0.0001** | **DL significantly beats HAR** |
| **EPOL** | 0.3322 | 0.3316 | +0.07 | 0.9431 | No significant difference |
| **EEM**  | 0.3355 | 0.3411 | −0.62 | 0.5355 | No significant difference |

**One sentence:** the Bi-LSTM only delivers a statistically significant gain on Germany — i.e. on the *intermediate* market, not on the most liquid one and not on the noisiest ones.

---

## 3. The "Triad of Market Efficiency"

The empirical pattern crystallizes into three regimes — what I call the **Triad of Market Efficiency**:

### 3.1. Noisy / Emerging markets — EPOL, EEM
Low signal-to-noise ratios punish flexible function approximators. Earlier (less regularized) iterations of the Bi-LSTM lost up to **−22%** to HAR through pure overfitting on microstructure noise. After aggressive regularization (high dropout, early stopping with `patience=5`, gradient clipping, AMP-stabilized training), the network finally **drew level with HAR** but never beat it. Classical econometrics is the *safer* tool here.

### 3.2. Highly efficient markets — SPY
On the S&P 500 the price process absorbs information almost instantaneously. The HAR model, simple as it is, sits very close to the **linear ceiling** of what is forecastable from past realized variance. The Bi-LSTM matches it but cannot extract any additional non-linear *alpha* — there is essentially none left to extract. This is what the paper calls the **"linearity barrier"** of mature markets.

### 3.3. The sweet spot — EWG (Germany)
Germany offers the **best of both worlds**: liquid enough to provide a clean mathematical signal, yet still inefficient enough non-linearly that classical HAR leaves money on the table. Here — and only here — the Attention-augmented Bi-LSTM **decisively** wins (DM = +3.99, p = 0.0001).

The takeaway: deep learning is not a universal upgrade. It is a *regime-dependent* tool whose edge appears on markets of **intermediate efficiency**.

---

## 4. Methodology

### 4.1. Target variable — Parkinson estimator
Instead of the common Close-to-Close variance — which is fragile under overnight gaps and intraday jumps — the target is the **Parkinson estimator** built from daily *High* and *Low*:

$$RV_t^{(P)} = \frac{(\ln H_t - \ln L_t)^2}{4 \ln 2} \cdot 10^4$$

This recovers most of the intraday-information advantage of high-frequency variance estimators at a fraction of the data and compute cost.

### 4.2. Features
The model is intentionally **hybrid**. The neural net does **not** start from raw prices — it is handed the same building blocks the econometrician would use:

- `RV` — Parkinson realized volatility
- `Ret` — log returns
- `Neg_Ret` — squared negative returns (to capture the **leverage effect** / asymmetric volatility)
- `RV_lag1`, `RV_week (5d)`, `RV_month (22d)` — the canonical HAR components

This makes the architecture a **DL-HAR**: the network either confirms the linear HAR structure or improves on it by finding *non-linear* interactions on top of it.

### 4.3. Architecture — Bi-LSTM + Attention

```
Input (15-day window × 6 features)
        │
   Bi-LSTM  (2 layers, hidden=64, bidirectional, dropout=0.3)
        │
   Attention layer  (softmax over time steps)
        │
   FC (128 → 32, ReLU) → FC (32 → 1) → Softplus
        │
       RV̂_{t+1}  (strictly positive)
```

Why each piece is there:

- **Bi-LSTM** — vanilla RNNs forget distant shocks; LSTM gates fix the vanishing-gradient pathology, and the *bidirectional* version lets the model contextualize a given day from both sides of the 15-day window.
- **Attention** — not every day in the window is equally informative. A crash 7 sessions ago should not be weighed the same as a quiet Wednesday. Attention learns those weights *dynamically*.
- **Softplus output** — variance is non-negative by definition. A plain linear head can produce −0.05; Softplus enforces strict positivity smoothly (and plays well with QLIKE which is undefined at zero).

### 4.4. Loss — QLIKE (asymmetric)

$$\mathcal{L}_{\text{QLIKE}}(y, \hat y) = \frac{y}{\hat y} - \ln\!\frac{y}{\hat y} - 1$$

QLIKE is the *natural* loss for variance forecasting. Unlike MSE, it punishes **under-prediction** of volatility much harder than over-prediction — exactly the asymmetry a real risk manager cares about. A model that says "calm market" the day before a crash is far more dangerous than one that is mildly too cautious.

### 4.5. Evaluation — strict daily walk-forward
This is where most "deep learning beats X" papers break down. To avoid look-ahead bias and any form of data leakage, the evaluation pipeline:

1. **Re-fits** both HAR (OLS) and the Bi-LSTM **every single trading day** of the out-of-sample window.
2. **Re-fits the StandardScaler** only on the rolling training set; the test point is `transform`-ed, never `fit_transform`-ed.
3. Splits the rolling training window into 80% train / 20% validation, with **early stopping (patience = 5)** on validation QLIKE.
4. Uses **gradient clipping** (max-norm 1.0) and AMP autocast on GPU for numerically stable training.
5. Fixes seeds (`torch.manual_seed(42)`, `np.random.seed(42)`) inside *every* daily iteration so the result is fully reproducible.

### 4.6. Statistical test — Diebold–Mariano
Differences in average QLIKE are tested with a **Diebold–Mariano** test (HAC-adjusted, lag = 1). A model is declared the winner only if the *p*-value falls below 5% **and** the QLIKE delta points in its favor.

---

## 5. Why this matters

- **It pushes back on the "deep learning solves everything" narrative.** No Free Lunch is alive and well in volatility forecasting.
- **It localizes where DL actually pays.** Not at the extremes (too noisy → overfit; too efficient → linear ceiling) but on **intermediate-efficiency** markets like Germany.
- **It is built like a real quant pipeline**, not a toy notebook: walk-forward, scaler discipline, asymmetric loss, statistical significance testing, leverage-effect features, positivity-preserving output. The negative results on SPY/EPOL/EEM are *negative results*, not bugs — and that is the point.

---

## 6. Repository contents

| File | What it is |
|------|------------|
| [`DeepVol_HAR.ipynb`](./DeepVol_HAR.ipynb) | Full Jupyter notebook: baseline (illusion of edge), robustness / overfitting stress test, final daily-walk-forward engine, and the comparative evaluation cell. |
| [`requirements.txt`](./requirements.txt) | Reproducibility — pinned-ish package list. |

---

## 7. How to reproduce

> Recommended environment: Python 3.10+, a CUDA GPU (the notebook was finalized on an A100 in Colab; it runs on CPU but the daily walk-forward becomes slow).

```bash
git clone https://github.com/BlazejRadzik/BiLSTM-HAR-Volatility.git
cd BiLSTM-HAR-Volatility
pip install -r requirements.txt
jupyter notebook DeepVol_HAR.ipynb
```

Then run the cells top-to-bottom. Data is pulled live from Yahoo Finance via `yfinance`, so no manual download is needed. The seed is fixed; you should reproduce the headline numbers up to floating-point noise.

---

## 8. Limitations & future work

- **Sample window.** The daily walk-forward currently runs on 2010–2024. Extending to a longer span (and including the 2008 crisis as out-of-sample) is on the roadmap.
- **Daily frequency.** Parkinson is a strong daily proxy, but the natural next step is to swap in true high-frequency realized variance (5-minute) where data is available.
- **Single architecture.** The same protocol could be re-run with Temporal Fusion Transformers, N-BEATS, or hybrid GARCH–DL models — to see whether the Triad pattern holds across architectures.
- **Multi-asset attention.** Letting the network attend across markets (e.g. SPY conditioning EWG) is the most promising direction for breaking the linearity barrier on SPY itself.

---

## 9. Citation

If you use this work, please cite as:

```
Radzik, B. (2026). Analiza porównawcza modelu Bi-LSTM i HAR w prognozowaniu zmienności rynkowej. W: A. Prędki (red.), Analiza ilościowa wybranych problemów z zakresu ekonomii i finansów (s. 43–52). Kraków: Uniwersytet Ekonomiczny w Krakowie.
```

---

## 10. Contact
- **Email:** blazej.radzik@edu.uekat.pl
- **LinkedIn:** [linkedin.com/in/błażej-radzik](https://www.linkedin.com/in/b%C5%82a%C5%BCej-radzik/)

Feedback, replications, and pull requests are very welcome.

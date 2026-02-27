"""
TRACE-SRS vs FSRS-6 vs SM-2 on Real Dataset
============================================
Runs comparison on the fsrs-vs-sm17 dataset from GitHub.
"""

import csv
import glob
import math
import os
import random
import warnings

import numpy as np
from datetime import datetime
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Be specific

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SM-2
# ─────────────────────────────────────────────────────────────────────────────

class SM2:
    def __init__(self):
        self.EF = 2.5
        self.interval = 1.0
        self.reps = 0

    def predict_R(self, t):
        return min(1.0, max(1e-6, math.exp(-t / (9.0 * max(0.1, self.interval)))))

    def update(self, ok, t):
        q = 4 if ok else 1
        if q >= 3:
            ni = {0: 1.0, 1: 6.0}.get(self.reps, self.interval * self.EF)
            self.reps += 1
        else:
            ni = 1.0
            self.reps = 0
        self.EF = max(1.3, self.EF + 0.1 - (5-q)*(0.08+(5-q)*0.02))
        self.interval = max(1.0, ni)

# ─────────────────────────────────────────────────────────────────────────────
# FSRS-6
# ─────────────────────────────────────────────────────────────────────────────

W6 = [0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001,
      1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014,
      1.8729, 0.5425, 0.0912, 0.0658, 0.1542]

class FSRS6:
    def __init__(self):
        self.S = None
        self.D = None
        self.n = 0

    def predict_R(self, t):
        if self.S is None:
            return 0.5
        k = 1.0 / (1.0 + W6[20])
        return min(1.0, max(1e-6, math.pow(1.0 + t / (9.0 * self.S), -k)))

    def _D0(self, g):
        return min(10.0, max(1.0, W6[4] - math.exp(W6[5]*(g-1)) + 1))

    def _next_D(self, D, g):
        delta = -W6[6] * (g - 3)
        Dp = D + delta * (10.0 - D) / 9.0
        return min(10.0, max(1.0, W6[7] * self._D0(4) + (1.0-W6[7]) * Dp))

    def _S_recall(self, S, D, R, g):
        hp = W6[15] if g==2 else 1.0
        eb = W6[16] if g==4 else 1.0
        sinc = (math.exp(W6[8]) * (11-D) * pow(S, -W6[9]) *
                (math.exp(W6[10]*(1-R))-1) * hp * eb + 1)
        return max(S, S * sinc)

    def _S_forget(self, S, D, R):
        sf = W6[11]*pow(D,-W6[12])*(pow(S+1,W6[13])-1)*math.exp(W6[14]*(1-R))
        return min(sf, S)

    def update(self, ok, t):
        g = 3 if ok else 1
        self.n += 1
        if self.S is None:
            self.S = W6[g-1]
            self.D = self._D0(g)
        else:
            R = self.predict_R(t)
            self.S = max(0.1, self._S_recall(self.S, self.D, R, g) if ok
                         else self._S_forget(self.S, self.D, R))
            self.D = self._next_D(self.D, g)

# ─────────────────────────────────────────────────────────────────────────────
# TRACE-SRS
# ─────────────────────────────────────────────────────────────────────────────

TRACE_W = [
    0.212, 1.2931, 2.3065, 8.2956,
    6.4133, 0.8334, 3.0194, 0.001,
    1.8722, 0.1666, 0.796, 1.4835,
    0.0614, 0.2629, 1.6483, 0.6014,
    3.2, 0.088, 0.45, 6.0, 0.40, 0.32, 1.35,
]

class TRACE:
    PRIOR_R = 0.68

    def __init__(self):
        self.w = TRACE_W
        self.S_f = None
        self.S_s = None
        self.w_f = 0.72
        self.D = None
        self.n = 0

    def _R_point(self, t):
        if self.S_f is None:
            return self.PRIOR_R
        w_s = 1.0 - self.w_f
        r_f = math.pow(0.9, t / max(0.1, self.S_f))
        r_s = math.pow(0.9, t / max(0.1, self.S_s))
        return min(1.0, max(1e-6, self.w_f * r_f + w_s * r_s))

    def predict_R(self, t):
        R_raw = self._R_point(t)
        conf = 1.0 - math.exp(-self.n / self.w[19])
        return R_raw * conf + self.PRIOR_R * (1.0 - conf)

    def _D0(self, g):
        w = self.w
        return min(10.0, max(1.0, w[4] - math.exp(w[5]*(g-1)) + 1))

    def _next_D(self, D, g, R):
        w = self.w
        surprise_f = 1.0 + w[20] * abs(R - 0.5) * 2.0
        delta = -w[6] * (g - 3) * surprise_f
        Dp = D + delta * (10.0 - D) / 9.0
        return min(10.0, max(1.0, w[7]*self._D0(4) + (1.0-w[7])*Dp))

    def _S_fast_recall(self, S_f, D, R, g):
        w = self.w
        hp = w[15] if g==2 else 1.0
        eb = w[16] if g==4 else 1.0
        base_sinc = (math.exp(w[8]) * (11-D) * pow(max(0.01, S_f), -w[9]) *
                     (math.exp(w[10]*(1-R))-1) * hp * eb + 1)
        surprise_amp = math.exp(w[21] * max(0.0, 1.0-R))
        sinc = base_sinc * (1.0 + (surprise_amp-1.0) * (1.0-R))
        return max(S_f, S_f * sinc)

    def _S_slow_recall(self, S_f, S_s, D, R, g):
        w = self.w
        hp = w[15] if g==2 else 1.0
        eb = w[16] if g==4 else 1.0
        base_sinc = (math.exp(w[8]) * (11-D) * pow(max(0.01, S_s), -w[9]*0.7) *
                     (math.exp(w[10]*(1-R)*0.8)-1) * hp * eb + 1)
        slow_sinc = base_sinc * w[22]
        new_S_s = max(S_s, S_s * slow_sinc)
        return min(new_S_s, S_f * w[16] * 5.0)

    def _S_fast_lapse(self, D):
        w = self.w
        return max(0.1, w[11] * (11-D) / 10.0)

    def _S_slow_lapse(self, S_s, R):
        return max(0.1, S_s * self.w[18])

    def update(self, ok, t):
        g = 3 if ok else 1
        R_raw = self._R_point(t)
        self.n += 1

        if self.S_f is None:
            self.S_f = self.w[g-1]
            self.S_s = self.S_f * self.w[16]
            self.D = self._D0(g)
        else:
            if ok:
                new_Sf = max(0.1, self._S_fast_recall(self.S_f, self.D, R_raw, g))
                new_Ss = max(0.1, self._S_slow_recall(new_Sf, self.S_s, self.D, R_raw, g))
                self.S_f = new_Sf
                self.S_s = new_Ss
                self.w_f = max(0.05, self.w_f * (1.0 - self.w[17]))
            else:
                self.S_f = self._S_fast_lapse(self.D)
                self.S_s = self._S_slow_lapse(self.S_s, R_raw)
                self.w_f = min(0.85, self.w_f + 0.20)
            self.D = self._next_D(self.D, g, R_raw)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(date_str):
    """Parse date string from CSV."""
    # Handle various date formats
    date_str = date_str.strip()
    # Replace non-breaking spaces and other weird chars
    date_str = date_str.replace('\ufffd', '').replace('\xa0', ' ')
    
    formats = [
        "%b %d %Y %H:%M:%S",
        "%b %d %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    # Fallback: try to extract just the date parts
    try:
        # Handle "May 03 2023 22:38:55" format
        parts = date_str.split()
        if len(parts) >= 3:
            month_day_year = ' '.join(parts[:3])
            return datetime.strptime(month_day_year, "%b %d %Y")
    except ValueError:
        return None
    return None

def load_csv_files(dataset_dir):
    """Load all CSV files and convert to our dataset format."""
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    all_cards = []
    total_rows = 0
    
    for csv_file in csv_files:
        user_name = os.path.basename(csv_file).replace("SM16-v-SM17_", "").replace(".csv", "")
        
        with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            
            # Normalize headers (strip whitespace)
            rows = []
            for row in reader:
                normalized = {k.strip(): v for k, v in row.items()}
                rows.append(normalized)
            
            # Group by Element No
            elements = {}
            for row in rows:
                elem_no = int(row['Element No'])
                if elem_no not in elements:
                    elements[elem_no] = []
                elements[elem_no].append(row)
        
        # Convert each element to card format
        for elem_no, reviews in elements.items():
            # Sort by date
            reviews_with_dates = []
            for row in reviews:
                dt = parse_date(row['Date'])
                if dt:
                    reviews_with_dates.append((dt, row))
            reviews_with_dates.sort(key=lambda x: x[0])
            
            if len(reviews_with_dates) < 2:
                continue
            
            card_reviews = []
            prev_date = None
            
            for dt, row in reviews_with_dates:
                # Calculate elapsed days
                if prev_date:
                    t = (dt - prev_date).total_seconds() / 86400.0
                else:
                    t = 0.0
                prev_date = dt
                
                # Get grade and outcome
                grade = int(row['Grade'])
                success = int(row['Success'])
                outcome = 1 if success == 1 else 0
                
                # Use SM17 R as reference
                r_sm17 = float(row['R (SM17)'])
                
                card_reviews.append({
                    "t": max(0.01, t),
                    "outcome": outcome,
                    "grade": grade,
                    "r_ref": r_sm17,
                    "user": user_name,
                    "element": elem_no,
                })
            
            if len(card_reviews) >= 2:
                all_cards.append(card_reviews)
                total_rows += len(card_reviews)
    
    print(f"Loaded {len(all_cards)} cards with {total_rows} total reviews")
    return all_cards

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def simulate(dataset, AlgoCls):
    pairs = []
    for card in dataset:
        algo = AlgoCls()
        for rev in card:
            t = rev["t"]
            ok = rev["outcome"]
            pred = algo.predict_R(t)
            pairs.append((pred, ok, rev))
            algo.update(ok, t)
    return pairs

def get_bin(x, bins=10):
    """Simple equal-width binning with 1/bins intervals from 0 to 1."""
    return min(1.0, max(0.0, np.round(x * bins) / bins))

def universal_metric(pA, outcomes, p_ref, n_bins=10):
    """
    Universal Metric (as defined by fsrs-vs-sm17):
    
    1. Bin data by reference algorithm's predictions (p_ref)
    2. For each bin, compute mean outcome and mean player prediction
    3. Calculate weighted RMSE between these means
    
    UM = sqrt(sum_b [n_b * (mean_outcome_b - mean_pred_b)^2] / total)
    
    Lower is better. 0% = perfect predictions.
    """
    # Group by bins of reference predictions
    bin_outcome_sum = np.zeros(n_bins + 1)
    bin_pred_sum = np.zeros(n_bins + 1)
    bin_count = np.zeros(n_bins + 1)
    
    for p, y, r in zip(pA, outcomes, p_ref):
        b = int(round(get_bin(r, n_bins) * n_bins))
        b = max(0, min(n_bins, b))
        bin_outcome_sum[b] += y
        bin_pred_sum[b] += p
        bin_count[b] += 1
    
    total = int(bin_count.sum())
    if total == 0:
        return float("nan")
    
    # Compute weighted RMSE between bin means
    weighted_sq_err = 0.0
    for b in range(n_bins + 1):
        if bin_count[b] > 0:
            mean_outcome = bin_outcome_sum[b] / bin_count[b]
            mean_pred = bin_pred_sum[b] / bin_count[b]
            weighted_sq_err += bin_count[b] * (mean_outcome - mean_pred) ** 2
    
    return math.sqrt(weighted_sq_err / total)

def log_loss(preds, outcomes):
    eps = 1e-7
    return -np.mean([y*math.log(max(eps,p))+(1-y)*math.log(max(eps,1-p))
                     for p,y in zip(preds,outcomes)])

def rmse_bins(preds, outcomes, n=20):
    bins = [[] for _ in range(n)]
    for p,y in zip(preds,outcomes):
        bins[min(n-1,int(p*n))].append((p,y))
    tot, wsq = 0, 0.0
    for b in bins:
        if b:
            wsq += len(b)*(np.mean([x[0] for x in b])-np.mean([x[1] for x in b]))**2
            tot += len(b)
    return math.sqrt(wsq/tot) if tot else float("nan")

def auc_roc(preds, outcomes):
    pos = [p for p,y in zip(preds,outcomes) if y]
    neg = [p for p,y in zip(preds,outcomes) if not y]
    if not pos or not neg:
        return 0.5
    p_arr = np.array(pos)
    n_arr = np.array(neg)
    wins = sum(np.sum(p > n_arr) + 0.5*np.sum(p == n_arr) for p in p_arr)
    return wins / (len(pos)*len(neg))

def brier_score(preds, outcomes):
    return np.mean([(p-y)**2 for p,y in zip(preds,outcomes)])

def ece(preds, outcomes, n=15):
    bins = [[] for _ in range(n)]
    for p,y in zip(preds,outcomes):
        bins[min(n-1,int(p*n))].append((p,y))
    tot = len(outcomes)
    err = 0.0
    for b in bins:
        if b:
            err += len(b)/tot * abs(np.mean([x[0] for x in b]) - np.mean([x[1] for x in b]))
    return err

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  TRACE-SRS vs FSRS-6 vs SM-2")
    print("  Real Dataset: fsrs-vs-sm17 GitHub repository")
    print("=" * 72)
    
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    print(f"\nLoading dataset from {dataset_dir}...")
    dataset = load_csv_files(dataset_dir)
    
    total = sum(len(c) for c in dataset)
    recall_rate = np.mean([r["outcome"] for card in dataset for r in card])
    print(f"  {len(dataset)} cards, {total} reviews")
    print(f"  Recall rate: {recall_rate:.3f}")
    
    # Calculate average interval
    all_t = [r["t"] for card in dataset for r in card if r["t"] > 0]
    print(f"  Avg interval: {np.mean(all_t):.2f} days")
    
    print("\nRunning simulations...")
    algos_cls = {"SM-2": SM2, "FSRS-6": FSRS6, "TRACE-SRS": TRACE}
    raw = {}
    for name, cls in algos_cls.items():
        print(f"  Simulating {name}...")
        raw[name] = simulate(dataset, cls)
    
    # Build metric inputs
    preds = {n: [p for p,_,_ in raw[n]] for n in raw}
    outs = [y for _,y,_ in raw["SM-2"]]
    refs = [r["r_ref"] for _,_,r in raw["SM-2"]]  # SM-17 R values from dataset
    
    # ── Universal Metric ────────────────────────────────────────────────────────
    names = list(algos_cls.keys())
    um_cross = {}
    for na in names:
        for nr in names:
            if na != nr:
                um_cross[(na,nr)] = universal_metric(preds[na], outs, preds[nr])
    um_self = {n: universal_metric(preds[n], outs, preds[n]) for n in names}
    um_avg = {n: np.mean([um_cross[(n,r)] for r in names if r!=n]) for n in names}
    
    # Also compute UM with SM-17 as reference (as in original paper)
    um_sm17_ref = {n: universal_metric(preds[n], outs, refs) for n in names}
    
    # ── Traditional metrics ─────────────────────────────────────────────────────
    metrics = {}
    for n in names:
        metrics[n] = {
            "LL": log_loss(preds[n], outs),
            "RMSE": rmse_bins(preds[n], outs),
            "AUC": auc_roc(preds[n], outs),
            "Brier": brier_score(preds[n], outs),
            "ECE": ece(preds[n], outs),
        }
    
    # ── Print ────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  UNIVERSAL METRIC (%, lower = better, 0% = perfect)")
    print("─" * 72)
    print(f"\n  {'Algorithm':<14} {'Self-UM':>9} {'Cross-UM':>10} {'vs FSRS-6':>12} {'UM(SM-17 ref)':>14}")
    print(f"  {'─'*14} {'─'*9} {'─'*10} {'─'*12} {'─'*14}")
    for n in names:
        vs = f"{(um_avg[n]-um_avg['FSRS-6'])*100:+.2f}pp" if n!='FSRS-6' else "baseline"
        print(f"  {n:<14} {um_self[n]*100:>8.2f}% {um_avg[n]*100:>9.2f}%  {vs:>12}  {um_sm17_ref[n]*100:>11.2f}%")
    
    print(f"\n{'─'*72}")
    print("  TRADITIONAL METRICS")
    print("─" * 72)
    print(f"\n  {'Metric':<14} {'SM-2':>10} {'FSRS-6':>10} {'TRACE-SRS':>10}  {'TRACE vs F6':>14}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*10}  {'─'*14}")
    for m, label, better in [("LL","Log Loss","low"),("RMSE","RMSE(bins)","low"),
                              ("AUC","AUC","high"),("Brier","Brier","low"),("ECE","ECE","low")]:
        vals = {n: metrics[n][m] for n in names}
        delta = vals["TRACE-SRS"] - vals["FSRS-6"]
        pct = abs(delta/vals["FSRS-6"])*100 if vals["FSRS-6"] != 0 else 0
        arrow = "↓" if better=="low" else "↑"
        sign = "better" if (delta<0)==(better=="low") else "worse"
        print(f"  {label+' '+arrow:<14} {vals['SM-2']:>10.4f} {vals['FSRS-6']:>10.4f} {vals['TRACE-SRS']:>10.4f}  "
              f"{delta:>+8.4f} ({sign})")
    
    print(f"\n{'─'*72}")
    print("  SUMMARY")
    print("─" * 72)
    
    # Count wins
    trace_wins = 0
    for m in ["LL", "RMSE", "Brier", "ECE"]:
        if metrics["TRACE-SRS"][m] < metrics["FSRS-6"][m]:
            trace_wins += 1
    if metrics["TRACE-SRS"]["AUC"] > metrics["FSRS-6"]["AUC"]:
        trace_wins += 1
    
    print(f"\n  TRACE-SRS wins {trace_wins}/5 metrics vs FSRS-6")
    print(f"  UM improvement: {(um_avg['FSRS-6']-um_avg['TRACE-SRS'])*100:+.2f}pp")
    print(f"  Log Loss improvement: {(metrics['FSRS-6']['LL']-metrics['TRACE-SRS']['LL'])/metrics['FSRS-6']['LL']*100:+.1f}%")

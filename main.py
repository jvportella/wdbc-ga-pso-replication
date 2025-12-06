#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Dependências:
  pip install numpy pandas scikit-learn joblib matplotlib """

from __future__ import annotations

import argparse
import os
import time
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Métricas (accuracy/sens/spec)
# ----------------------------

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def metrics_acc_sens_spec(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    return {"accuracy": float(acc), "sensitivity": float(sens), "specificity": float(spec)}

def mean_std(rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return out


# ----------------------------
# Dataset WDBC (UCI)
# ----------------------------

def load_wdbc(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    WDBC (Wisconsin Diagnostic Breast Cancer):
    formato comum: [ID, diagnosis(M/B), 30 features]
    Classe positiva: M (maligno) -> 1, B (benigno) -> 0
    """
    df = pd.read_csv(path, header=None)
    # coluna 0 = ID (remove)
    df = df.drop(columns=[0])

    y = (df.iloc[:, 0].astype(str).str.strip().to_numpy() == "M").astype(int)
    X = df.iloc[:, 1:].to_numpy(dtype=float)

    # índices no artigo normalmente são 1..30 (após remover ID e diagnosis)
    feature_names = [f"x{i}" for i in range(1, X.shape[1] + 1)]
    return X, y, feature_names


# ----------------------------
# PSO-classifier (fitness)
# ----------------------------

@dataclass
class PSOParams:
    swarm_size: int = 150      # artigo
    inertia: float = 0.7       # artigo
    c1: float = 2.0            # AJUSTE (não especificado no artigo)
    c2: float = 2.0            # AJUSTE (não especificado no artigo)
    max_iters: int = 80        # AJUSTE (não especificado no artigo)
    bounds: Tuple[float, float] = (-4.0, 4.0)
    seed: int = 0

class PSOLinearClassifier:
    """
    PSO otimiza (w,b) de um classificador linear:
      pred = 1 se X@w + b >= 0, senão 0
    Fitness interno (para o PSO): Miss = número de erros (minimizar).
    """
    def __init__(self, params: PSOParams):
        self.params = params
        self.w_: Optional[np.ndarray] = None
        self.b_: Optional[float] = None

    @staticmethod
    def _misses(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        logits = X @ W.T + b  # (n, s)
        preds = (logits >= 0.0).astype(np.int32)
        misses = np.sum(preds != y[:, None], axis=0).astype(np.int32)
        return misses

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PSOLinearClassifier":
        p = self.params
        rng = np.random.default_rng(p.seed)
        n, d = X.shape
        low, high = p.bounds

        W = rng.uniform(low, high, size=(p.swarm_size, d))
        b = rng.uniform(low, high, size=(p.swarm_size,))
        Vw = np.zeros_like(W)
        Vb = np.zeros_like(b)

        m = self._misses(X, y, W, b)
        pbest_W = W.copy()
        pbest_b = b.copy()
        pbest_m = m.copy()

        g = int(np.argmin(pbest_m))
        gbest_W = pbest_W[g].copy()
        gbest_b = float(pbest_b[g])
        gbest_m = int(pbest_m[g])

        for _ in range(p.max_iters):
            if gbest_m == 0:
                break

            r1 = rng.random(size=W.shape)
            r2 = rng.random(size=W.shape)
            r1b = rng.random(size=b.shape)
            r2b = rng.random(size=b.shape)

            Vw = (p.inertia * Vw
                  + p.c1 * r1 * (pbest_W - W)
                  + p.c2 * r2 * (gbest_W[None, :] - W))
            Vb = (p.inertia * Vb
                  + p.c1 * r1b * (pbest_b - b)
                  + p.c2 * r2b * (gbest_b - b))

            W = np.clip(W + Vw, low, high)
            b = np.clip(b + Vb, low, high)

            m = self._misses(X, y, W, b)

            improved = m < pbest_m
            if np.any(improved):
                pbest_W[improved] = W[improved]
                pbest_b[improved] = b[improved]
                pbest_m[improved] = m[improved]

                g = int(np.argmin(pbest_m))
                if int(pbest_m[g]) < gbest_m:
                    gbest_m = int(pbest_m[g])
                    gbest_W = pbest_W[g].copy()
                    gbest_b = float(pbest_b[g])

        self.w_ = gbest_W
        self.b_ = gbest_b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None or self.b_ is None:
            raise RuntimeError("Chame fit() antes de predict().")
        return ((X @ self.w_) + self.b_ >= 0.0).astype(np.int32)


# ----------------------------
# GA para seleção de atributos (params do artigo)
# ----------------------------

@dataclass
class GAParams:
    pop_size: int = 150
    generations: int = 300
    crossover_rate: float = 0.5
    mutation_rate: float = 0.4
    elite_rate: float = 0.1

    # ajustes (não especificados no artigo)
    tournament_k: int = 3
    seed: int = 0
    n_jobs: int = 1
    cache_fitness: bool = True
    bit_flip_prob: Optional[float] = None  # se None: 1/n_features

def ensure_nonempty(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if mask.sum() == 0:
        mask[rng.integers(0, mask.size)] = True
    return mask

def mask_key(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8)).tobytes()

class GAFeatureSelector:
    
    """ GA  para selecionar features.
    Fitness do GA: accuracy obtida pelo PSO-classifier no TREINO (como wrapper). """

    def __init__(self, ga: GAParams, pso: PSOParams):
        self.ga = ga
        self.pso = pso

    def _fitness(self, mask: np.ndarray, Xtr: np.ndarray, ytr: np.ndarray, cache: Dict[bytes, float]) -> float:
        if mask.sum() == 0:
            return 0.0
        k = mask_key(mask)
        if self.ga.cache_fitness and k in cache:
            return cache[k]

        Xs = Xtr[:, mask]

        # seed determinístico derivado da máscara pra reduzir ruído no GA
        seed = (self.ga.seed * 1000003 + int(np.frombuffer(k, dtype=np.uint8).sum())) % (2**32 - 1)
        clf = PSOLinearClassifier(PSOParams(**{**self.pso.__dict__, "seed": int(seed)}))
        clf.fit(Xs, ytr)
        pred_tr = clf.predict(Xs)

        fit = metrics_acc_sens_spec(ytr, pred_tr)["accuracy"]  # maximizar
        if self.ga.cache_fitness:
            cache[k] = float(fit)
        return float(fit)

    def select(self, Xtr: np.ndarray, ytr: np.ndarray) -> Tuple[np.ndarray, float]:
        ga = self.ga
        rng = np.random.default_rng(ga.seed)

        n_features = Xtr.shape[1]
        bit_p = ga.bit_flip_prob if ga.bit_flip_prob is not None else (1.0 / n_features)

        # init população
        pop = (rng.random(size=(ga.pop_size, n_features)) < 0.5)
        for i in range(ga.pop_size):
            pop[i] = ensure_nonempty(pop[i], rng)

        cache: Dict[bytes, float] = {}

        def fit_one(ind: np.ndarray) -> float:
            return self._fitness(ind, Xtr, ytr, cache)

        def eval_pop(pop_arr: np.ndarray) -> np.ndarray:
            if ga.n_jobs == 1:
                return np.array([fit_one(ind) for ind in pop_arr], dtype=float)
            scores = Parallel(n_jobs=ga.n_jobs, prefer="processes")(
                delayed(fit_one)(ind) for ind in pop_arr
            )
            return np.array(scores, dtype=float)

        def tournament(scores: np.ndarray) -> np.ndarray:
            idxs = rng.integers(0, ga.pop_size, size=ga.tournament_k)
            best = idxs[np.argmax(scores[idxs])]
            return pop[best].copy()

        def one_point_cx(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if n_features < 2:
                return a.copy(), b.copy()
            cut = int(rng.integers(1, n_features))
            c1 = np.concatenate([a[:cut], b[cut:]])
            c2 = np.concatenate([b[:cut], a[cut:]])
            return c1, c2

        def mutate(ind: np.ndarray) -> np.ndarray:
            # interpretação: mutation_rate = prob de aplicar mutação no indivíduo
            if rng.random() < ga.mutation_rate:
                flips = rng.random(size=n_features) < bit_p
                ind = np.logical_xor(ind, flips)
                ind = ensure_nonempty(ind, rng)
            return ind

        elite_n = max(1, int(math.floor(ga.elite_rate * ga.pop_size)))

        best_mask = pop[0].copy()
        best_score = -1.0

        for _ in range(ga.generations):
            scores = eval_pop(pop)

            g = int(np.argmax(scores))
            if float(scores[g]) > best_score:
                best_score = float(scores[g])
                best_mask = pop[g].copy()

            elite_idx = np.argsort(scores)[-elite_n:][::-1]
            new_pop = [pop[i].copy() for i in elite_idx]

            while len(new_pop) < ga.pop_size:
                p1 = tournament(scores)
                p2 = tournament(scores)

                if rng.random() < ga.crossover_rate:
                    c1, c2 = one_point_cx(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = mutate(c1)
                if len(new_pop) < ga.pop_size:
                    new_pop.append(c1)

                c2 = mutate(c2)
                if len(new_pop) < ga.pop_size:
                    new_pop.append(c2)

            pop = np.stack(new_pop, axis=0)

        return best_mask, float(best_score)


# ----------------------------
# Experimento: 30 repetições 80/20
# ----------------------------

def run_one_repeat(
    X: np.ndarray, y: np.ndarray,
    rep_seed: int,
    ga_params: GAParams,
    pso_params: PSOParams
) -> Dict[str, object]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=rep_seed, stratify=y
    )

    # padronizar (fit apenas no treino)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # baseline (sem FS)
    base_clf = PSOLinearClassifier(PSOParams(**{**pso_params.__dict__, "seed": rep_seed}))
    base_clf.fit(X_tr_s, y_tr)
    base_pred = base_clf.predict(X_te_s)
    base_m = metrics_acc_sens_spec(y_te, base_pred)

    # GA (FS)
    selector = GAFeatureSelector(GAParams(**{**ga_params.__dict__, "seed": rep_seed}), pso_params)
    t0 = time.time()
    mask, best_fit = selector.select(X_tr_s, y_tr)
    fs_time = time.time() - t0

    # com FS (treina PSO no subconjunto e testa)
    X_tr_fs = X_tr_s[:, mask]
    X_te_fs = X_te_s[:, mask]

    fs_clf = PSOLinearClassifier(PSOParams(**{**pso_params.__dict__, "seed": rep_seed}))
    fs_clf.fit(X_tr_fs, y_tr)
    fs_pred = fs_clf.predict(X_te_fs)
    fs_m = metrics_acc_sens_spec(y_te, fs_pred)

    idx_1based = (np.where(mask)[0] + 1).tolist()

    return {
        "seed": rep_seed,
        "ga_best_fitness_train_acc": float(best_fit),
        "fs_seconds": float(fs_time),
        "n_features_selected": int(mask.sum()),
        "selected_features_1based": " ".join(map(str, idx_1based)),

        "base_accuracy": base_m["accuracy"],
        "base_sensitivity": base_m["sensitivity"],
        "base_specificity": base_m["specificity"],

        "fs_accuracy": fs_m["accuracy"],
        "fs_sensitivity": fs_m["sensitivity"],
        "fs_specificity": fs_m["specificity"],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Caminho para wdbc.data (UCI)")
    ap.add_argument("--out", default="./results", help="Pasta de saída")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    # GA: parâmetros do artigo (fixos por padrão)
    ap.add_argument("--pop", type=int, default=150)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--cx", type=float, default=0.5)
    ap.add_argument("--mut", type=float, default=0.4)
    ap.add_argument("--elite", type=float, default=0.1)

    # Ajustes GA (não especificados no artigo)
    ap.add_argument("--jobs", type=int, default=1, help="Paralelismo na avaliação do GA (CPU)")
    ap.add_argument("--tournament-k", type=int, default=3)
    ap.add_argument("--bit-flip-prob", type=float, default=None)

    # PSO: parte do artigo + ajustes necessários
    ap.add_argument("--swarm", type=int, default=150)
    ap.add_argument("--inertia", type=float, default=0.7)
    ap.add_argument("--c1", type=float, default=2.0)
    ap.add_argument("--c2", type=float, default=2.0)
    ap.add_argument("--pso-iters", type=int, default=80)

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X, y, feat_names = load_wdbc(args.data)
    rng = np.random.default_rng(args.seed)

    ga_params = GAParams(
        pop_size=args.pop,
        generations=args.gens,
        crossover_rate=args.cx,
        mutation_rate=args.mut,
        elite_rate=args.elite,
        tournament_k=args.tournament_k,
        n_jobs=args.jobs,
        bit_flip_prob=args.bit_flip_prob,
        seed=args.seed
    )

    pso_params = PSOParams(
        swarm_size=args.swarm,
        inertia=args.inertia,
        c1=args.c1,
        c2=args.c2,
        max_iters=args.pso_iters,
        seed=args.seed
    )

    print(f"WDBC loaded: X={X.shape}, positive_rate={y.mean():.3f}")
    print(f"GA (artigo): pop={ga_params.pop_size}, gens={ga_params.generations}, cx={ga_params.crossover_rate}, "
          f"mut={ga_params.mutation_rate}, elite={ga_params.elite_rate}")
    print(f"GA (ajustes): tournament_k={ga_params.tournament_k}, bit_flip_prob={ga_params.bit_flip_prob}, jobs={ga_params.n_jobs}")
    print(f"PSO (artigo+ajustes): swarm={pso_params.swarm_size}, inertia={pso_params.inertia}, c1={pso_params.c1}, c2={pso_params.c2}, iters={pso_params.max_iters}")

    rows: List[Dict[str, object]] = []
    for i in range(args.repeats):
        rep_seed = int(rng.integers(0, 2**31 - 1))
        print(f"\n[repeat {i+1:02d}/{args.repeats}] seed={rep_seed}")
        r = run_one_repeat(X, y, rep_seed, ga_params, pso_params)
        print(f"  base acc={r['base_accuracy']:.4f} | fs acc={r['fs_accuracy']:.4f} | "
              f"selected={r['n_features_selected']} | ga_train_fit={r['ga_best_fitness_train_acc']:.4f} | fs_time={r['fs_seconds']:.1f}s")
        rows.append(r)

    df = pd.DataFrame(rows)
    df_path = os.path.join(args.out, "wdbc_results_per_repeat.csv")
    df.to_csv(df_path, index=False)

    # summary
    base_rows = [{"accuracy": float(r["base_accuracy"]), "sensitivity": float(r["base_sensitivity"]), "specificity": float(r["base_specificity"])} for r in rows]
    fs_rows = [{"accuracy": float(r["fs_accuracy"]), "sensitivity": float(r["fs_sensitivity"]), "specificity": float(r["fs_specificity"])} for r in rows]

    base_ms = mean_std(base_rows)
    fs_ms = mean_std(fs_rows)

    # frequência das features
    freq = np.zeros(X.shape[1], dtype=int)
    for r in rows:
        idxs = [int(s) for s in str(r["selected_features_1based"]).split()] if str(r["selected_features_1based"]).strip() else []
        for j in idxs:
            freq[j - 1] += 1

    freq_df = pd.DataFrame({
        "feature_1based": np.arange(1, X.shape[1] + 1),
        "name": feat_names,
        "selected_count": freq,
        "selected_rate": freq / max(1, args.repeats),
    }).sort_values(["selected_count", "feature_1based"], ascending=[False, True])

    freq_path = os.path.join(args.out, "wdbc_feature_frequency.csv")
    freq_df.to_csv(freq_path, index=False)

    summary = {
        "base_accuracy_mean": base_ms["accuracy"][0], "base_accuracy_std": base_ms["accuracy"][1],
        "base_sensitivity_mean": base_ms["sensitivity"][0], "base_sensitivity_std": base_ms["sensitivity"][1],
        "base_specificity_mean": base_ms["specificity"][0], "base_specificity_std": base_ms["specificity"][1],

        "fs_accuracy_mean": fs_ms["accuracy"][0], "fs_accuracy_std": fs_ms["accuracy"][1],
        "fs_sensitivity_mean": fs_ms["sensitivity"][0], "fs_sensitivity_std": fs_ms["sensitivity"][1],
        "fs_specificity_mean": fs_ms["specificity"][0], "fs_specificity_std": fs_ms["specificity"][1],

        "mean_n_features_selected": float(df["n_features_selected"].mean()),
        "mean_fs_seconds": float(df["fs_seconds"].mean()),
    }

    summary_path = os.path.join(args.out, "wdbc_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n================ SUMMARY (média ± desvio) ================")
    print(f"BASE  acc={base_ms['accuracy'][0]:.4f} ± {base_ms['accuracy'][1]:.4f} | "
          f"sens={base_ms['sensitivity'][0]:.4f} ± {base_ms['sensitivity'][1]:.4f} | "
          f"spec={base_ms['specificity'][0]:.4f} ± {base_ms['specificity'][1]:.4f}")
    print(f"FS    acc={fs_ms['accuracy'][0]:.4f} ± {fs_ms['accuracy'][1]:.4f} | "
          f"sens={fs_ms['sensitivity'][0]:.4f} ± {fs_ms['sensitivity'][1]:.4f} | "
          f"spec={fs_ms['specificity'][0]:.4f} ± {fs_ms['specificity'][1]:.4f}")

    print("\nArquivos gerados:")
    print(" -", df_path)
    print(" -", freq_path)
    print(" -", summary_path)
    print("\nTop features (mais selecionadas):")
    print(freq_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

import json
import pandas as pd
import matplotlib.pyplot as plt

# ======= ajuste aqui =======
MY_OUTDIR = r".\final_jobs4"
SWAP = True  # True se quiser trocar sens<->spec para alinhar convenções
# ===========================

# números do ARTIGO (WDBC) — coloque como fração (0.964) ou % (96.4)
ARTICLE_BASE = {"accuracy": 0.964, "specificity": 0.931, "sensitivity": 0.986}
ARTICLE_FS   = {"accuracy": 0.972, "specificity": 0.956, "sensitivity": 0.980}

def norm(v):
    return v/100.0 if v > 1 else v

ARTICLE_BASE = {k: norm(v) for k, v in ARTICLE_BASE.items()}
ARTICLE_FS   = {k: norm(v) for k, v in ARTICLE_FS.items()}

if SWAP:
    ARTICLE_BASE["sensitivity"], ARTICLE_BASE["specificity"] = ARTICLE_BASE["specificity"], ARTICLE_BASE["sensitivity"]
    ARTICLE_FS["sensitivity"], ARTICLE_FS["specificity"] = ARTICLE_FS["specificity"], ARTICLE_FS["sensitivity"]

# ---- carrega seus dados
with open(f"{MY_OUTDIR}\\wdbc_summary.json", "r", encoding="utf-8") as f:
    s = json.load(f)

df = pd.read_csv(f"{MY_OUTDIR}\\wdbc_results_per_repeat.csv")

# ---- extrai médias/desvios
MY_BASE = {
    "accuracy": (s["base_accuracy_mean"], s["base_accuracy_std"]),
    "sensitivity": (s["base_sensitivity_mean"], s["base_sensitivity_std"]),
    "specificity": (s["base_specificity_mean"], s["base_specificity_std"]),
}
MY_FS = {
    "accuracy": (s["fs_accuracy_mean"], s["fs_accuracy_std"]),
    "sensitivity": (s["fs_sensitivity_mean"], s["fs_sensitivity_std"]),
    "specificity": (s["fs_specificity_mean"], s["fs_specificity_std"]),
}

# ========== Gráfico 1: barras com erro + artigo ==========
metrics = ["accuracy", "sensitivity", "specificity"]
x = range(len(metrics))

base_means = [MY_BASE[m][0] for m in metrics]
base_stds  = [MY_BASE[m][1] for m in metrics]
fs_means   = [MY_FS[m][0] for m in metrics]
fs_stds    = [MY_FS[m][1] for m in metrics]

width = 0.35

plt.figure(figsize=(9, 5))
plt.bar([i - width/2 for i in x], base_means, width, yerr=base_stds, capsize=4, label="Seu BASE (média±dp)")
plt.bar([i + width/2 for i in x], fs_means,   width, yerr=fs_stds,   capsize=4, label="Seu FS (média±dp)")

# pontos do artigo
plt.plot(x, [ARTICLE_BASE[m] for m in metrics], marker="o", linestyle="None", label="Artigo BASE (média)")
plt.plot(x, [ARTICLE_FS[m] for m in metrics],   marker="x", linestyle="None", label="Artigo FS (média)")

plt.xticks(list(x), metrics)
plt.ylim(0.85, 1.0)
plt.title(f"Comparação WDBC (SWAP={SWAP})")
plt.ylabel("Valor (0–1)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{MY_OUTDIR}\\compare_bars.png", dpi=200)

# ========== Gráfico 2: boxplot 30 repetições + linhas do artigo ==========
plt.figure(figsize=(9, 5))

data = [
    df["base_accuracy"], df["fs_accuracy"],
    df["base_sensitivity"], df["fs_sensitivity"],
    df["base_specificity"], df["fs_specificity"],
]
labels = [
    "BASE acc", "FS acc",
    "BASE sens", "FS sens",
    "BASE spec", "FS spec",
]
plt.boxplot(data, labels=labels)

# linhas do artigo
plt.axhline(ARTICLE_BASE["accuracy"], linestyle="--")
plt.axhline(ARTICLE_FS["accuracy"], linestyle="--")
plt.axhline(ARTICLE_BASE["sensitivity"], linestyle=":")
plt.axhline(ARTICLE_FS["sensitivity"], linestyle=":")
plt.axhline(ARTICLE_BASE["specificity"], linestyle="-." )
plt.axhline(ARTICLE_FS["specificity"], linestyle="-." )

plt.ylim(0.85, 1.0)
plt.title(f"Distribuição (30 repetições) + linhas do artigo (SWAP={SWAP})")
plt.ylabel("Valor (0–1)")
plt.tight_layout()
plt.savefig(f"{MY_OUTDIR}\\compare_boxplots.png", dpi=200)

print("Gerado:")
print(f" - {MY_OUTDIR}\\compare_bars.png")
print(f" - {MY_OUTDIR}\\compare_boxplots.png")

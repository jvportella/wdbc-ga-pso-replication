import pandas as pd
import matplotlib.pyplot as plt

# ===== ajuste aqui =====
OUTDIR = r".\final_jobs1"
SWAP_SENS_SPEC = True  # True se quiser inverter sens<->spec (convenção de classe positiva)
# =======================

# Valores do artigo (WDBC, Tabela 5) para PSO/PS-classifier
ART_BASE = {"accuracy": 0.964, "sensitivity": 0.986, "specificity": 0.931}
ART_FS   = {"accuracy": 0.972, "sensitivity": 0.980, "specificity": 0.956}

def norm(v):
    return v/100.0 if v > 1 else v

ART_BASE = {k: norm(v) for k, v in ART_BASE.items()}
ART_FS   = {k: norm(v) for k, v in ART_FS.items()}

if SWAP_SENS_SPEC:
    ART_BASE["sensitivity"], ART_BASE["specificity"] = ART_BASE["specificity"], ART_BASE["sensitivity"]
    ART_FS["sensitivity"], ART_FS["specificity"] = ART_FS["specificity"], ART_FS["sensitivity"]

df = pd.read_csv(f"{OUTDIR}\\wdbc_results_per_repeat.csv")
x = range(1, len(df) + 1)

def lineplot(metric, fname):
    plt.figure(figsize=(10, 5))

    # suas curvas por repetição
    plt.plot(x, df[f"base_{metric}"], marker="o", label=f"Seu BASE {metric}")
    plt.plot(x, df[f"fs_{metric}"], marker="x", label=f"Seu FS {metric}")

    # médias dos seus resultados (linhas horizontais em laranja)
    my_base_mean = df[f"base_{metric}"].mean()
    my_fs_mean   = df[f"fs_{metric}"].mean()

    plt.axhline(my_base_mean, linestyle="--", linewidth=2, color="orange",
                label=f"Sua média BASE ({my_base_mean:.4f})")
    plt.axhline(my_fs_mean, linestyle=":", linewidth=2, color="orange",
                label=f"Sua média FS ({my_fs_mean:.4f})")

    # linhas do artigo (estilos diferentes)
    plt.axhline(ART_BASE[metric], linestyle="--", linewidth=2,
                label=f"Artigo BASE {metric}")
    plt.axhline(ART_FS[metric], linestyle=":", linewidth=2,
                label=f"Artigo FS {metric}")

    plt.title(f"WDBC por repetição (SWAP={SWAP_SENS_SPEC}) — {metric}")
    plt.xlabel("Repetição")
    plt.ylabel(metric)
    plt.ylim(0.85, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}\\{fname}", dpi=200)
    plt.close()

lineplot("accuracy", "line_accuracy.png")
lineplot("sensitivity", "line_sensitivity.png")
lineplot("specificity", "line_specificity.png")

print("Gerados:")
print(f" - {OUTDIR}\\line_accuracy.png")
print(f" - {OUTDIR}\\line_sensitivity.png")
print(f" - {OUTDIR}\\line_specificity.png")

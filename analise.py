import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("mall-customers.csv")
print(df.describe())

# 1. Análise Geral do Perfil dos Clientes

# Distribuição de Idade, Renda e Pontuação de Gastos.
plt.hist(df["age"], edgecolor="black", color="skyblue")
plt.title("Distribuição de Idades dos Clientes")
plt.xlabel("Idade")
plt.ylabel("Frequência")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
plt.clf()

plt.hist(df["annual-income"], bins=10, edgecolor='black', color="lightgreen")
plt.title("Distribuição de Renda Anual")
plt.xlabel("Renda Anual (em milhares de dólares)")
plt.ylabel("Número de Clientes")
plt.show()
plt.clf()

plt.hist(df["spending-score"], bins=10, edgecolor="black", color="salmon")
plt.title("Distribuição da Pontuação de Gastos")
plt.xlabel("Pontuação de Gastos (1-100)")
plt.ylabel("Número de Clientes")
plt.show()
plt.clf()

# Relação entre Renda e Gastos.
plt.scatter(df["annual-income"], df["spending-score"], alpha=0.7)
plt.title('Relação entre Renda Anual e Pontuação de Gastos')
plt.xlabel('Renda Anual (em milhares de dólares)')
plt.ylabel('Pontuação de Gastos (1-100)')
plt.show()
plt.clf()

# 2. Segmentação e Comparação dos Clientes

# Funções para criar as categorias de segmentação.
def categorizar_idade(idade):
    if idade < 30:
        return "Jovem"
    elif idade < 50:
        return "Adulto"
    else:
        return "Maduro"

def categorizar_renda(renda):
    if renda < 40:
        return "Baixa"
    elif renda < 70:
        return "Média"
    else:
        return "Alta"

df['faixa_etaria'] = df['age'].apply(categorizar_idade)
df['faixa_renda'] = df['annual-income'].apply(categorizar_renda)

# Boxplot de gastos por Gênero.
mulheres = df[df["is_male"] == 0]["spending-score"]
homens = df[df["is_male"] == 1]["spending-score"]

plt.boxplot([mulheres, homens], patch_artist=True,
            boxprops=dict(facecolor="skyblue", color="black"),
            medianprops=dict(color="red"))
plt.title("Pontuação de Gastos por Gênero")
plt.xlabel("Gênero")
plt.ylabel("Pontuação de Gastos (1-100)")
plt.xticks([1, 2], ["Mulheres", "Homens"])
plt.show()
plt.clf()

# Boxplot de gastos por Faixa Etária.
jovens = df[df['faixa_etaria'] == "Jovem"]["spending-score"]
adultos = df[df['faixa_etaria'] == "Adulto"]["spending-score"]
maduros = df[df['faixa_etaria'] == "Maduro"]["spending-score"]

plt.boxplot([jovens, adultos, maduros], patch_artist=True,
            boxprops=dict(facecolor="lightgreen", color="black"),
            medianprops=dict(color="red"))
plt.title("Pontuação de Gastos por Faixa Etária")
plt.xlabel("Faixa Etária")
plt.ylabel("Pontuação de Gastos (1-100)")
plt.xticks([1, 2, 3], ["Jovens (18-29)", "Adultos (30-49)", "Maduros (50+)"])
plt.show()
plt.clf()

# Boxplot de gastos por Faixa de Renda.
renda_baixa = df[df['faixa_renda'] == "Baixa"]["spending-score"]
renda_media = df[df['faixa_renda'] == "Média"]["spending-score"]
renda_alta = df[df['faixa_renda'] == "Alta"]["spending-score"]

plt.boxplot([renda_baixa, renda_media, renda_alta], patch_artist=True,
            boxprops=dict(facecolor="gold", color="black"),
            medianprops=dict(color="red"))
plt.title("Pontuação de Gastos por Faixa de Renda")
plt.xlabel("Faixa de Renda")
plt.ylabel("Pontuação de Gastos (1-100)")
plt.xticks([1, 2, 3], ["Baixa (<40k)", "Média (40-70k)", "Alta (70k+)"])
plt.show()
plt.clf()

# 3. Análise Combinada: Gênero e Idade

# Cruzando os dados para uma visão mais detalhada.
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Mulheres por idade.
mulheres_jovens = df[(df["is_male"] == 0) & (df['faixa_etaria'] == "Jovem")]["spending-score"]
mulheres_adultas = df[(df["is_male"] == 0) & (df['faixa_etaria'] == "Adulto")]["spending-score"]
mulheres_maduras = df[(df["is_male"] == 0) & (df['faixa_etaria'] == "Maduro")]["spending-score"]

axs[0].boxplot([mulheres_jovens, mulheres_adultas, mulheres_maduras], patch_artist=True,
               boxprops=dict(facecolor="lightpink", color="black"),
               medianprops=dict(color="red"))
axs[0].set_title("Mulheres por Faixa Etária")
axs[0].set_xlabel("Faixa Etária")
axs[0].set_ylabel("Pontuação de Gastos (1-100)")
axs[0].set_xticklabels(["Jovens", "Adultas", "Maduras"])

# Homens por idade.
homens_jovens = df[(df["is_male"] == 1) & (df['faixa_etaria'] == "Jovem")]["spending-score"]
homens_adultos = df[(df["is_male"] == 1) & (df['faixa_etaria'] == "Adulto")]["spending-score"]
homens_maduros = df[(df["is_male"] == 1) & (df['faixa_etaria'] == "Maduro")]["spending-score"]

axs[1].boxplot([homens_jovens, homens_adultos, homens_maduros], patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="black"),
               medianprops=dict(color="red"))
axs[1].set_title("Homens por Faixa Etária")
axs[1].set_xlabel("Faixa Etária")
axs[1].set_xticklabels(["Jovens", "Adultos", "Maduros"])

plt.tight_layout()
plt.show()
plt.clf()
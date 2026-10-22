# ============================================================
# Chargement et nettoyage de 16_MDG_EPM2122_AGRC.dta
# Enquête : EPM 2021-2022 (INSTAT Madagascar) — module Agriculture
# ============================================================

import pandas as pd
import pyreadstat

# ---- 0. Options d'affichage pandas (utiles pour inspecter) ----
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)

# ---- 1. Lecture du fichier .dta ----
FICHIER = "16_MDG_EPM2122_AGRC.dta"
df, meta = pyreadstat.read_dta(FICHIER)

print(f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes\n")

# ---- 2. Récupération des libellés (tels quels, avec mojibake) ----
labels_bruts = meta.column_names_to_labels   # dict {code: libellé}

# ---- 3. Réparation des accents (mojibake cp932 → utf-8 → latin-1 → utf-8) ----
def reparer(s):
    """Corrige un mojibake du type 'Mﾃδｩnage' → 'Ménage'."""
    if not isinstance(s, str) or not s:
        return s
    try:
        return (
            s.encode('cp932')
             .decode('utf-8')
             .encode('latin-1')
             .decode('utf-8')
        )
    except (UnicodeError, UnicodeDecodeError):
        return s

labels = {code: reparer(lib) for code, lib in labels_bruts.items()}

# Si vous avez installé ftfy (pip install ftfy), la version "pro" :
# import ftfy
# labels = {code: ftfy.fix_text(lib) for code, lib in labels_bruts.items()}

# ---- 4. Aperçu des libellés réparés ----
print("=== Libellés réparés (extrait) ===")
for code, lib in list(labels.items())[:15]:
    print(f"{code:25s} → {lib}")

# ---- 5. On garde une copie avec les CODES d'origine (pour fusions) ----
df_codes = df.copy()

# ---- 6. Renommage des colonnes en "code — libellé" ----
# (on garde le code pour ne pas perdre la traçabilité)
df.columns = [
    f"{c} — {labels.get(c, '').strip()}" if labels.get(c) else c
    for c in df.columns
]

# Variante : remplacer totalement par le libellé (plus court, mais risque de doublons)
# df = df.rename(columns=labels)

# ---- 7. Aperçu des données ----
print("\n=== 5 premières lignes ===")
print(df.head())
pd.Series(df.columns).to_csv("xxxx.csv", index=False, header=["colonne"], encoding="utf-8-sig")

print("\n=== Colonnes finales (extrait) ===")
for c in list(df.columns)[:15]:
    print(" •", c)

# ---- 8. Structure et statistiques rapides ----
print("\n=== df.info() ===")
df.info()

# print("\n=== df.describe() (colonnes numériques) ===")
# print(df.describe().T)

# ---- 9. Vérification des clés ménage ----
# print("\n=== Valeurs manquantes sur les clés ménage ===")
# print(df[['idHH', 'hhgrap', 'hhnum']].isna().sum())

# ---- 10. Export (optionnel) pour inspection dans Excel/LibreOffice ----
# df_codes.to_csv("epm2122_agrc_codes.csv", index=False, encoding="utf-8-sig")
# df.to_csv("epm2122_agrc_lisible.csv", index=False, encoding="utf-8-sig")
# df.to_excel("epm2122_agrc.xlsx", index=False)
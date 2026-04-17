import pandas as pd
import numpy as np
import os
import re
import warnings
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. ΡΥΘΜΙΣΕΙΣ & ΦΑΚΕΛΟΙ
# ==============================================================================
INPUT_DATA_EXCEL = r"C:\Users\temp\Desktop\05_03_2025 data comparison.xlsx"
GLOVOCS_EXCEL    = r"C:\Users\temp\Downloads\GLOVOCSupdate1_16112020.xlsx"

BASE_DIR = r"C:\Users\temp\Desktop\Sensors_Improved_Analysis"
os.makedirs(BASE_DIR, exist_ok=True)

OUT_S1          = os.path.join(BASE_DIR, "Sensor_1_Improved.xlsx")
OUT_S2          = os.path.join(BASE_DIR, "Sensor_2_Improved.xlsx")
OUT_FINGERPRINT = os.path.join(BASE_DIR, "Fingerprint_Matrix_Improved.xlsx")
OUT_SUMMARY     = os.path.join(BASE_DIR, "Analysis_Summary.xlsx")

ABSOLUTE_INCREASE = 10.0
SENSOR_1_LIMIT    = 8800
SENSOR_2_LIMIT    = 10300

# Blacklist παραμέτρων οργάνου
INSTRUMENTAL_BLACKLIST = [
    "instrumental parameter", "primary ion", "isotope", "water cluster",
    "oxygen cluster", "hydronium", "water cluster isotope", "oxygen isotope",
    "oxygen-17", "oxygen-18", "m.+", "shall be", "primary ion isotope",
    "unknown mass", "unknown"
]

# Όρια αξιοπιστίας ppm
PPM_RELIABLE_THRESHOLD = 20.0   
PPM_UNCERTAIN_THRESHOLD = 50.0  

EVENTS_DICT = {
    "Markers":          ("10:15:00", "12:13:00"),
    "Blanko_Kores":     ("12:13:00", "12:22:00"),
    "Blanko_President": ("12:22:00", "13:17:00"),
    "Acrylic_Paint":    ("13:17:00", "14:17:00"),
    "Paint_Markers":    ("14:17:00", "14:54:00"),
    "Uhu_Stick":        ("14:54:00", "15:45:00"),
    "Pega_Pen_Glue":    ("15:45:00", "16:16:00"),
    "Liquid_Uhu":       ("16:16:00", "17:03:00"),
}

# ==============================================================================
# 2. ΦΟΡΤΩΣΗ GLOVOCS
# ==============================================================================
print("⏳ Φόρτωση GLOVOCS...")
try:
    df_glovocs = pd.read_excel(GLOVOCS_EXCEL)
    df_glovocs["m.H+"] = pd.to_numeric(df_glovocs["m.H+"], errors="coerce")
    df_glovocs = df_glovocs.dropna(subset=["m.H+"])
    print(f"✅ {len(df_glovocs)} ενώσεις φορτώθηκαν.")
except Exception as e:
    print(f"❌ Σφάλμα φόρτωσης: {e}"); input("Enter για έξοδο..."); exit()

# ==============================================================================
# 3. ΣΥΝΑΡΤΗΣΕΙΣ ΤΑΥΤΟΠΟΙΗΣΗΣ
# ==============================================================================
def get_ppm_reliability(ppm_val):
    abs_ppm = abs(ppm_val)
    if abs_ppm <= PPM_RELIABLE_THRESHOLD:
        return "✅ Reliable"
    elif abs_ppm <= PPM_UNCERTAIN_THRESHOLD:
        return "⚠️ Uncertain"
    else:
        return "❌ Unreliable"

def identify_mz(mz_column_name):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(mz_column_name))
    if not numbers:
        return "Unknown", None, "Unidentified", "Unknown", None, "❌ Unreliable"

    target_mz = float(numbers[0])

    # GLOVOCS Match
    df_glovocs["_diff"] = abs(df_glovocs["m.H+"] - target_mz)
    closest = df_glovocs.loc[df_glovocs["_diff"].idxmin()]
    glov_name, glov_ppm, glov_cat = "Not in GLOVOCS", None, "Unidentified"

    if closest["_diff"] <= 0.005:
        theo = closest["m.H+"]
        glov_ppm = round(((target_mz - theo) / theo) * 1e6, 1)
        glov_name = str(closest.get("Name", "Unknown")).strip()
        c_str = str(closest.get("Class", "")).lower()
        if "oxygen" in c_str or "carbonyl" in c_str:   glov_cat = "Oxygenated"
        elif "hydrocarbon" in c_str or "benzenoid" in c_str: glov_cat = "Hydrocarbons"
        elif "nitrogen" in c_str:                      glov_cat = "Nitrogenous"
        elif "sulfur" in c_str:                        glov_cat = "Sulfur"
        elif "halogen" in c_str:                       glov_cat = "Halogenated"
        else:                                          glov_cat = "Other"

    # Math Match
    neutral_mass = target_mz - 1.007276
    m_C, m_H, m_O, m_N, m_Si = 12.0, 1.007825, 15.994914, 14.003074, 27.976926
    candidates = []
    for c in range(21):
        for o in range(9):
            for n in range(5):
                for si in range(6):
                    base_m = c * m_C + o * m_O + n * m_N + si * m_Si
                    if base_m > neutral_mass + 0.010: break
                    h_est = int(round((neutral_mass - base_m) / m_H))
                    if not (0 <= h_est <= (2 * c + 2 * si + n + 2)): continue
                    dbe = c + si + 1 - (h_est / 2.0) + (n / 2.0)
                    if dbe < 0 or not float(dbe).is_integer(): continue
                    theo_m = base_m + h_est * m_H
                    diff = abs(neutral_mass - theo_m)
                    if diff <= 0.010:
                        ppm = round(((neutral_mass - theo_m) / theo_m) * 1e6, 1)
                        f = (f"C{c if c > 1 else ''}" if c > 0 else "") + \
                            (f"H{h_est if h_est > 1 else ''}" if h_est > 0 else "") + \
                            (f"N{n if n > 1 else ''}" if n > 0 else "") + \
                            (f"O{o if o > 1 else ''}" if o > 0 else "") + \
                            (f"Si{si if si > 1 else ''}" if si > 0 else "")
                        candidates.append((f, diff, ppm))

    math_formula, math_ppm = "No Match", None
    if candidates:
        candidates.sort(key=lambda x: x[1])
        math_formula, math_ppm = candidates[0][0], candidates[0][2]

    best_ppm = glov_ppm if glov_ppm is not None else math_ppm
    reliability = get_ppm_reliability(best_ppm) if best_ppm is not None else "❌ Unreliable"

    return glov_name, glov_ppm, glov_cat, math_formula, math_ppm, reliability

def is_instrumental(glov_name, math_formula):
    combined = (str(glov_name) + " " + str(math_formula)).lower()
    return any(word in combined for word in INSTRUMENTAL_BLACKLIST)

# ==============================================================================
# 4. ΦΟΡΤΩΣΗ & ΣΥΓΧΡΟΝΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ
# ==============================================================================
print("⏳ Φόρτωση και συγχρονισμός δεδομένων...")
df_ptr_raw = pd.read_excel(INPUT_DATA_EXCEL, sheet_name='PTR DATA')
df_ptr_raw['Timestamp'] = pd.to_datetime(df_ptr_raw['Timestamp']) + pd.Timedelta(minutes=6)
df_ptr = df_ptr_raw.set_index('Timestamp').resample('2min').mean(numeric_only=True).sort_index()

df_ens_raw = pd.read_excel(INPUT_DATA_EXCEL, sheet_name='ENS DATA', header=1)
df_ens = pd.DataFrame()
df_ens['Timestamp'] = pd.to_datetime(df_ens_raw.iloc[:, 0], errors='coerce')
df_ens['Sensor_1'] = pd.to_numeric(
    df_ens_raw['vocec_al_1'] if 'vocec_al_1' in df_ens_raw.columns else df_ens_raw.iloc[:, 11],
    errors='coerce')
df_ens['Sensor_2'] = pd.to_numeric(
    df_ens_raw['vocec_al_2'] if 'vocec_al_2' in df_ens_raw.columns else df_ens_raw.iloc[:, 26],
    errors='coerce')
df_ens.dropna(subset=['Timestamp'], inplace=True)
df_ens.sort_values('Timestamp', inplace=True)

df_full  = pd.merge_asof(df_ptr.reset_index(), df_ens, on='Timestamp', direction='nearest', tolerance=pd.Timedelta(minutes=1)).set_index('Timestamp')
df_ml = df_full.copy()
df_ml.loc[df_ml['Sensor_1'] >= SENSOR_1_LIMIT, 'Sensor_1'] = np.nan
df_ml.loc[df_ml['Sensor_2'] >= SENSOR_2_LIMIT, 'Sensor_2'] = np.nan

mz_cols  = [c for c in df_full.columns if 'm/z' in c]
year_str = df_full.index[0].strftime('%Y-%m-%d')
print(f"✅ {len(mz_cols)} m/z channels, {len(df_full)} timepoints.")

# ==============================================================================
# 5. ΚΥΡΙΑ ΑΝΑΛΥΣΗ
# ==============================================================================
s1_list, s2_list = [], []
event_summaries = []

print("🚀 Εκτέλεση ανάλυσης...")

for event, (t_start, t_end) in EVENTS_DICT.items():
    sdt = pd.to_datetime(f"{year_str} {t_start}")
    edt = pd.to_datetime(f"{year_str} {t_end}")
    ev_full = df_full[(df_full.index >= sdt) & (df_full.index <= edt)]
    ev_ml   = df_ml  [(df_ml.index   >= sdt) & (df_ml.index   <= edt)]
    if ev_full.empty: continue

    active_mzs = [c for c in mz_cols if (ev_full[c].max() - ev_full[c].iloc[0]) >= ABSOLUTE_INCREASE]
    if not active_mzs: continue

    for s_col, master in [('Sensor_1', s1_list), ('Sensor_2', s2_list)]:
        temp_df = ev_ml[active_mzs + [s_col]].dropna()
        n, p    = len(temp_df), len(active_mzs)

        loo_cv_r2 = None
        rf_insample_r2 = None
        if n >= 10:
            X_raw = temp_df[active_mzs].values
            y_raw = temp_df[s_col].values
            rf_full = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_raw, y_raw)
            rf_insample_r2 = round(float(r2_score(y_raw, rf_full.predict(X_raw))), 3)

            if n <= 80:  
                preds = []
                for tri, tei in LeaveOneOut().split(X_raw):
                    rf_l = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_l.fit(X_raw[tri], y_raw[tri])
                    preds.append(float(rf_l.predict(X_raw[tei])[0]))
                ss_res = np.sum((y_raw - np.array(preds)) ** 2)
                ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
                loo_cv_r2 = round(float(1 - ss_res / ss_tot), 3) if ss_tot > 0 else None

        artifact_risk = ("HIGH"   if n < p or n < 10 else
                         "MEDIUM" if n / p < 2       else "LOW")

        event_summaries.append({
            "Event": event, "Sensor": s_col,
            "n (samples)": n, "p (active m/z)": p, "n/p ratio": round(n / p, 2) if p > 0 else 0,
            "RF In-sample R²": rf_insample_r2,
            "LOO-CV R²": loo_cv_r2,
            "Artifact Risk": artifact_risk,
            "Warning": ("⛔ n < p: μοντέλα αναξιόπιστα" if n < p else
                        "⚠️ Οριακό n/p" if n / p < 2 else "✅ OK")
        })

        ridge_coef, rf_imp, gbm_imp = {}, {}, {}
        if n > 4:
            X_sc = StandardScaler().fit_transform(temp_df[active_mzs])
            y_sc = StandardScaler().fit_transform(temp_df[[s_col]]).flatten()
            ridge = Ridge(alpha=1.0).fit(X_sc, y_sc)
            ridge_coef = {active_mzs[i]: ridge.coef_[i] for i in range(p)}

            X_raw2 = temp_df[active_mzs].values
            y_raw2 = temp_df[s_col].values
            rf  = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_raw2, y_raw2)
            gbm = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_raw2, y_raw2)
            rf_imp  = {active_mzs[i]: rf.feature_importances_[i] * 100  for i in range(p)}
            gbm_imp = {active_mzs[i]: gbm.feature_importances_[i] * 100 for i in range(p)}

        pearson_r2_map, pearson_p_map, spearman_r_map, spearman_p_map = {}, {}, {}, {}
        for col in active_mzs:
            vidx = ~np.isnan(ev_ml[col]) & ~np.isnan(ev_ml[s_col])
            if vidx.sum() > 3:
                r_p, p_p = pearsonr(ev_ml[col][vidx], ev_ml[s_col][vidx])
                r_s, p_s = spearmanr(ev_ml[col][vidx], ev_ml[s_col][vidx])
                pearson_r2_map[col]  = r_p ** 2
                pearson_p_map[col]   = float(p_p)
                spearman_r_map[col]  = float(r_s)
                spearman_p_map[col]  = float(p_s)

        valid_cols = [c for c in active_mzs if c in pearson_p_map]
        fdr_map = {}
        if valid_cols:
            raw_ps = [pearson_p_map[c] for c in valid_cols]
            _, p_adj, _, _ = multipletests(raw_ps, alpha=0.05, method='fdr_bh')
            fdr_map = {c: float(p_adj[i]) for i, c in enumerate(valid_cols)}

        for col in active_mzs:
            fdr_p = fdr_map.get(col, 1.0)
            if fdr_p > 0.05:
                continue

            glov_name, glov_ppm, glov_cat, math_form, math_ppm, reliability = identify_mz(col)

            if is_instrumental(glov_name, math_form):
                continue

            p_r2  = pearson_r2_map.get(col, 0)
            s_r   = spearman_r_map.get(col, 0)
            s_r2  = s_r ** 2
            div   = abs(p_r2 - s_r2)
            if div > 0.3:
                divergence_flag = "⚠️ Outlier-driven"
            elif div > 0.15:
                divergence_flag = "🟡 Check"
            else:
                divergence_flag = "✅ Robust"

            ptr_increase = int(round(ev_full[col].max() - ev_full[col].iloc[0], 0))
            glov_str = f"{glov_name} ({glov_ppm}ppm)" if glov_ppm is not None else glov_name
            math_str = f"{math_form} ({math_ppm}ppm)" if math_ppm is not None else math_form

            # 🛠 ΕΔΩ ΠΡΟΣΤΕΘΗΚΕ ΤΟ "Sensor": s_col ΓΙΑ ΝΑ ΛΥΘΕΙ ΤΟ KEY ERROR 🛠
            row = {
                "Experiment":           event,
                "Sensor":               s_col,   
                "m/z":                  col,
                "GLOVOCS Match":        glov_str,
                "Math Formula":         math_str,
                "Category":             glov_cat,
                "ID Reliability":       reliability,
                "Pearson R²":           round(p_r2, 3),
                "Spearman R":           round(s_r, 3),
                "Pearson/Spearman Flag": divergence_flag,
                "FDR p-value":          round(fdr_p, 4),
                "RF Importance (%)":    round(rf_imp.get(col, 0), 1),
                "GBM Importance (%)":   round(gbm_imp.get(col, 0), 1),
                "Ridge Beta":           round(ridge_coef.get(col, 0), 3),
                "PTR Increase (ppb)":   ptr_increase,
                "Artifact Risk":        artifact_risk,
                "LOO-CV R² (event)":    loo_cv_r2 if loo_cv_r2 is not None else "n/a",
            }
            master.append(row)

# ==============================================================================
# 6. ΕΞΑΓΩΓΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
# ==============================================================================
print("📊 Εξαγωγή αποτελεσμάτων...")

def export_by_event(data, path):
    if not data: return
    df = pd.DataFrame(data)
    df = df.sort_values(["Experiment", "Pearson R²"], ascending=[True, False])
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for exp in df["Experiment"].unique():
            sheet = df[df["Experiment"] == exp].drop(columns=["Experiment", "Sensor"])
            sheet.to_excel(writer, sheet_name=exp[:31], index=False)

export_by_event([d for d in s1_list if d['Sensor'] == 'Sensor_1'], OUT_S1)
export_by_event([d for d in s2_list if d['Sensor'] == 'Sensor_2'], OUT_S2)

all_df = pd.DataFrame(s1_list + s2_list)
if not all_df.empty:
    all_df["Base Name"] = all_df["GLOVOCS Match"].apply(
        lambda x: re.sub(r"\s*\([-+]?\d+\.?\d*ppm\)", "", str(x)).strip()
    )
    all_df["Compound Label"] = all_df.apply(
        lambda r: r["Math Formula"].split(" (")[0] if r["Base Name"] == "Not in GLOVOCS"
                  else r["Base Name"], axis=1
    )

    all_df["Present"] = np.where(
        (all_df["FDR p-value"] <= 0.05) & (all_df["Pearson/Spearman Flag"] != "⚠️ Outlier-driven"), 1, 0
    )

    # 🛠 ΤΟ PIVOT TABLE ΤΩΡΑ ΘΑ ΔΟΥΛΕΨΕΙ ΚΑΝΟΝΙΚΑ 🛠
    finger = pd.pivot_table(
        all_df, values="Present",
        index="Compound Label",
        columns=["Sensor", "Experiment"],
        aggfunc="max", fill_value=0
    )
    finger.to_excel(OUT_FINGERPRINT)

    summary_df = pd.DataFrame(event_summaries)
    with pd.ExcelWriter(OUT_SUMMARY, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name="Event Summary", index=False)
        counts = all_df[all_df["Present"] == 1].groupby(["Experiment", "Sensor"]).size().reset_index(name="Significant Compounds")
        counts.to_excel(writer, sheet_name="Compound Counts", index=False)

print(f"""
🎉 Ανάλυση ολοκληρώθηκε χωρίς σφάλματα!

📁 Αρχεία: {BASE_DIR}
""")
input("Πάτα Enter για έξοδο...")
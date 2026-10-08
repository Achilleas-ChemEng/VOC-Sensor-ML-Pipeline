
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:47:07 2026

@author: temp
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.signal import correlate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import os
import re
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# ΡΥΘΜΙΣΕΙΣ ΧΡΗΣΤΗ
# ==============================================================================
FILES_TO_PROCESS = [
    #r"C:\Users\temp\Desktop\01_12_2025 data comparison.xlsx",
    #r"C:\Users\temp\Desktop\08_12_2025 data comparison.xlsx",
   # r"C:\Users\temp\Desktop\07_12_2025 data comparison.xlsx",
    #r"C:\Users\temp\Desktop\23_01_2026 data comparison.xlsx",
   # r"C:\Users\temp\Desktop\30_01_2026 data comparison.xlsx",
  # r"C:\Users\temp\Desktop\20_02_2026 data comparison.xlsx",
  # r"C:\Users\temp\Desktop\24_02_2026 data comparison.xlsx",
   ### r"C:\Users\temp\Desktop\28_02_2026 data comparison.xlsx",
   ## r"C:\Users\temp\Desktop\01_03_2026 data comparison.xlsx" ,
  # r"C:\Users\temp\Desktop\BENZALDEHYDE.xlsx" 
   r"C:\Users\temp\Desktop\additional_marker_tol_phen_form.xlsx"
  # r"C:\Users\temp\Desktop\08_12_2025 data comparison.xlsx"
]

OUTPUT_EXCEL = r"C:\Users\temp\Desktop\new plots223.xlsx"
OUTPUT_PLOTS_TWIN_DIR = r"C:\Users\temp\Desktop\new plotsL223"
OUTPUT_PLOTS_SINGLE_DIR = r"C:\Users\temp\Desktop\new plotsL223"

# ==============================================================================
# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ==============================================================================
def clean_numeric(series):
    if series.dtype == 'O': 
        series = series.astype(str).str.replace(',', '.')
    return pd.to_numeric(series, errors='coerce')

def calculate_time_lag(s_sns, s_ptr):
    """Ασφαλής υπολογισμός του Lag. Αν πέσει σε διέρεση με το 0, επιστρέφει NaN."""
    mask = ~np.isnan(s_sns) & ~np.isnan(s_ptr)
    c1, c2 = s_sns[mask], s_ptr[mask]
    
    # Αν έχουμε λιγότερα από 2 σημεία ή απόλυτη ευθεία, δεν υπάρχει lag
    if len(c1) < 2 or np.std(c1) == 0 or np.std(c2) == 0:
        return np.nan
        
    try:
        c1_norm = c1 - np.mean(c1)
        c2_norm = c2 - np.mean(c2)
        correlation = correlate(c1_norm, c2_norm, mode='full')
        lags = np.arange(-len(c1_norm) + 1, len(c1_norm))
        return lags[np.argmax(correlation)]
    except:
        return np.nan

def make_safe_filename(s):
    """Καθαρίζει το string από απαγορευμένους χαρακτήρες για τα Windows."""
    # Αντικαθιστά \ / : * ? " < > | και κρυφές αλλαγές γραμμής
    return re.sub(r'[\\/*?:"<>|\n\r\t\0]', '_', str(s)).strip()

# ==============================================================================
# ΚΥΡΙΑ ΕΠΕΞΕΡΓΑΣΙΑ
# ==============================================================================
def process_batch_files(file_paths, output_excel_path, out_twin_dir, out_single_dir):
    os.makedirs(out_twin_dir, exist_ok=True)
    os.makedirs(out_single_dir, exist_ok=True)

    stats_dict = {
        ('Raw Timing', True, 'Full'): [], ('Raw Timing', True, 'Peak_Only'): [],
        ('Lag Shifted', True, 'Full'): [], ('Lag Shifted', True, 'Peak_Only'): [],
        ('Raw Timing', False, 'Full'): [], ('Raw Timing', False, 'Peak_Only'): [],
        ('Lag Shifted', False, 'Full'): [], ('Lag Shifted', False, 'Peak_Only'): [],
    }

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        print(f"\n========================================================")
        print(f"ΕΠΕΞΕΡΓΑΣΙΑ ΑΡΧΕΙΟΥ: {filename}")
        print(f"========================================================")
        
        date_match = re.search(r'(\d{1,2}_\d{1,2}_\d{4})', filename)
        date_str = date_match.group(1) if date_match else os.path.splitext(filename)[0]
        
        try:
            data = pd.read_excel(file_path, sheet_name="SELECTED DATA")
        except Exception as e:
            print(f" Σφάλμα ανάγνωσης στο αρχείο {filename}: {e}")
            continue

        n = 1 
        while n + 2 < len(data.columns):
            try:
                subset = data.iloc[:, n:n+3].dropna(how='all')
                if subset.empty: 
                    n += 5
                    continue

                substance_name = str(data.columns[n-1]).strip()
                if substance_name.lower() in ['nan', '']:
                    n += 5
                    continue
                
                # Χρήση της νέας συνάρτησης για ασφαλή ονόματα φακέλων/αρχείων
                safe_substance_name = make_safe_filename(substance_name)
                
                print(f"\n--- Αναλύεται η ουσία: {substance_name} ---")
                
                base_columns, comparison_columns = [n, n+1], [n+2]     
                
                baselines = {}
                for col_idx in base_columns + comparison_columns:
                    col_data_clean = clean_numeric(data.iloc[:, col_idx]).dropna()
                    baselines[col_idx] = col_data_clean.min() if not col_data_clean.empty else 0.0

                time_column = data.iloc[:, n-1]
                time_dt_full = pd.to_datetime(time_column.astype(str), errors='coerce')
                is_time = not time_dt_full.isna().all()
                if not is_time: time_dt_full = time_column.astype(str)

                ptr_raw = clean_numeric(data.iloc[:, comparison_columns[0]])
                s1_raw = clean_numeric(data.iloc[:, base_columns[0]])
                s2_raw = clean_numeric(data.iloc[:, base_columns[1]])
                
                # Υπολογισμός End Index για το Peak_Only
                ptr_bg_sub = ptr_raw - baselines[comparison_columns[0]]
                if not ptr_bg_sub.dropna().empty:
                    peak_idx = ptr_bg_sub.argmax()
                    peak_val = ptr_bg_sub.iloc[peak_idx]
                    threshold = 0.05 * peak_val 
                    after_peak = ptr_bg_sub.iloc[peak_idx:]
                    if len(after_peak) > 1:
                        drop_locs = np.where(after_peak < threshold)[0]
                        end_idx = peak_idx + drop_locs[0] if len(drop_locs) > 0 else len(ptr_bg_sub) - 1
                    else:
                        end_idx = len(ptr_bg_sub) - 1
                else:
                    end_idx = len(ptr_bg_sub) - 1

                for subtract_bg in [True, False]:
                    bg_suffix = "With_BG_Sub" if subtract_bg else "No_BG_Sub"

                    ptr_full = ptr_raw - baselines[comparison_columns[0]] if subtract_bg else ptr_raw
                    s1_full = s1_raw - baselines[base_columns[0]] if subtract_bg else s1_raw
                    s2_full = s2_raw - baselines[base_columns[1]] if subtract_bg else s2_raw
                    
                    #  ΥΠΟΛΟΓΙΣΜΟΣ LAG (Ασφαλής μέθοδος)
                    lag1 = calculate_time_lag(s1_full, ptr_full)
                    lag2 = calculate_time_lag(s2_full, ptr_full)
                    
                    s1_lagged = s1_full.shift(-int(lag1)) if not np.isnan(lag1) else s1_full.copy()
                    s2_lagged = s2_full.shift(-int(lag2)) if not np.isnan(lag2) else s2_full.copy()

                    windows = [("Full", len(ptr_full)), ("Peak_Only", end_idx + 1)]
                    phases = [("Raw Timing", s1_full, s2_full), ("Lag Shifted", s1_lagged, s2_lagged)]

                    # ====================================================================
                    # ΔΗΜΙΟΥΡΓΙΑ TIME SERIES (ΚΑΙ ΓΙΑ RAW ΚΑΙ ΓΙΑ LAGGED)
                    # ====================================================================
                    for phase_name, s1_active, s2_active in phases:
                        
                        ts_twin_dir = os.path.join(out_twin_dir, date_str, safe_substance_name, bg_suffix, "Time_Series", phase_name.replace(' ', '_'))
                        ts_single_dir = os.path.join(out_single_dir, date_str, safe_substance_name, bg_suffix, "Time_Series", phase_name.replace(' ', '_'))
                        os.makedirs(ts_twin_dir, exist_ok=True)
                        os.makedirs(ts_single_dir, exist_ok=True)

                        for window_name, e_idx in windows:
                            t_win = time_dt_full.iloc[:e_idx]
                            p_win = ptr_full.iloc[:e_idx]
                            s1_win = s1_active.iloc[:e_idx]
                            s2_win = s2_active.iloc[:e_idx]
                            
                            base_title = f"{substance_name} - {date_str} - {window_name} - {phase_name}"
                            
                            # Χρήση της νέας συνάρτησης για το όνομα του αρχείου
                            base_filename = make_safe_filename(base_title)
                            
                            # ΝΕΑ ΟΝΟΜΑΤΑ ΑΡΧΕΙΩΝ ΓΙΑ ΝΑ ΜΗΝ ΓΙΝΕΤΑΙ OVERWRITE
                            ts_filename_twin = f"{base_filename}_Dual_Axis_TS.png"
                            ts_filename_single = f"{base_filename}_Single_Axis_TS.png"

                            # ΠΛΟΤ 1: TWIN AXIS (Απόλυτες Τιμές με 2 Άξονες)
                            fig_twin, ax1 = plt.subplots(figsize=(12, 7))
                            ax2 = ax1.twinx()
                            l1, = ax1.plot(t_win, s1_win, label='TVOCs Sensor 1', linestyle='-', color='red')
                            l2, = ax1.plot(t_win, s2_win, label='TVOCs Sensor 2', linestyle='-', color='blue')
                            l3, = ax2.plot(t_win, p_win, label=f'PTR-ToF-MS ({substance_name})', linestyle='--', color='black', linewidth=2)
                            
                            ax1.set_title(f"{base_title}\n(Dual-Axis Real Values)", fontsize=16)
                            # Αλλαγή σε κεφαλαία & αύξηση μεγέθους
                            ax1.set_xlabel('TIME (UTC + 02:00)', fontsize=16)
                            ax1.set_ylabel('CONCENTRATION (ppb)', color='black', fontsize=16)
                            ax2.set_ylabel('PTR-ToF-MS', color='black', fontsize=16)
                            
                            # Αύξηση μεγέθους στους αριθμούς των αξόνων
                            ax1.tick_params(axis='both', which='major', labelsize=14)
                            ax2.tick_params(axis='both', which='major', labelsize=14)
                            
                            ax1.set_ylim(bottom=0)
                            ax2.set_ylim(bottom=0)
                            
                            # ΑΣΦΑΛΕΣ ΚΛΕΙΔΩΜΑ ΑΞΟΝΑ Χ (Χωρίς περιθώρια)
                            if is_time:
                                t_clean = t_win.dropna()
                                if not t_clean.empty:
                                    ax1.set_xlim(t_clean.min(), t_clean.max())
                            else:
                                ax1.set_xlim(0, max(0, len(t_win) - 1))
                            
                            lines = [l1, l2, l3]
                            # Αύξηση μεγέθους υπομνήματος
                            ax1.legend(lines, [l.get_label() for l in lines], loc='best', framealpha=0.0, fontsize=14)
                            if is_time: ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            else: ax1.set_xticks(np.arange(0, len(t_win), max(1, len(t_win)//8)))
                            
                            fig_twin.savefig(os.path.join(ts_twin_dir, ts_filename_twin), bbox_inches='tight')
                            plt.close(fig_twin)

                            # ΠΛΟΤ 2: SINGLE AXIS (Απόλυτες Τιμές στον ίδιο άξονα)
                            fig_single, ax_s = plt.subplots(figsize=(12, 7))
                            ax_s.plot(t_win, s1_win, label='TVOCs Sensor 1', linestyle='-', color='red')
                            ax_s.plot(t_win, s2_win, label='TVOCs Sensor 2', linestyle='-', color='blue')
                            ax_s.plot(t_win, p_win, label=f'PTR-ToF-MS ({substance_name})', linestyle='--', color='black', linewidth=2)
                            
                            ax_s.set_title(f"{base_title}\n(Single-Axis Absolute Values)", fontsize=16)
                            
                            # Αλλαγή σε κεφαλαία & αύξηση μεγέθους
                            ax_s.set_xlabel('TIME (UTC + 02:00)', fontsize=16)
                            ax_s.set_ylabel('CONCENTRATION (ppb)', fontsize=16)
                            
                            # Αύξηση μεγέθους στους αριθμούς του άξονα
                            ax_s.tick_params(axis='both', which='major', labelsize=14)
                            
                            ax_s.set_ylim(bottom=0)
                            
                            # ΑΣΦΑΛΕΣ ΚΛΕΙΔΩΜΑ ΑΞΟΝΑ Χ (Χωρίς περιθώρια)
                            if is_time:
                                t_clean = t_win.dropna()
                                if not t_clean.empty:
                                    ax_s.set_xlim(t_clean.min(), t_clean.max())
                            else:
                                ax_s.set_xlim(0, max(0, len(t_win) - 1))
                            
                            # Αύξηση μεγέθους υπομνήματος
                            ax_s.legend(loc='best', framealpha=0.0, fontsize=14)
                            
                            if is_time: ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            else: ax_s.set_xticks(np.arange(0, len(t_win), max(1, len(t_win)//8)))
                            
                            fig_single.savefig(os.path.join(ts_single_dir, ts_filename_single), bbox_inches='tight')
                            plt.close(fig_single)


                    # ====================================================================
                    # ΣΤΑΤΙΣΤΙΚΑ ΚΑΙ SCATTERS
                    # ====================================================================
                    sc_twin_dir = os.path.join(out_twin_dir, date_str, safe_substance_name, bg_suffix, "Scatter_Plots")
                    sc_single_dir = os.path.join(out_single_dir, date_str, safe_substance_name, bg_suffix, "Scatter_Plots")
                    
                    for cols in [(base_columns[0], comparison_columns[0]), (base_columns[1], comparison_columns[0])]:
                        sensor_num = "1" if cols[0] == base_columns[0] else "2"
                        sensor_name = f"TVOCs Sensor {sensor_num}"
                        
                        col1_full = s1_full if sensor_num == "1" else s2_full
                        col2_full = ptr_full
                        active_lag = lag1 if sensor_num == "1" else lag2

                        datasets_to_run = []
                        for window_name, e_idx in windows:
                            c1_win = col1_full.iloc[:e_idx]
                            c2_win = col2_full.iloc[:e_idx]
                            
                            mask_win = ~np.isnan(c1_win) & ~np.isnan(c2_win)
                            c1_raw_m = c1_win[mask_win]
                            c2_raw_m = c2_win[mask_win]
                            
                            for filter_name in ["None", "5_95_Filter"]:
                                if filter_name == "5_95_Filter":
                                    if len(c1_raw_m) < 2: continue
                                    p5_1, p95_1 = np.percentile(c1_raw_m, 5), np.percentile(c1_raw_m, 95)
                                    p5_2, p95_2 = np.percentile(c2_raw_m, 5), np.percentile(c2_raw_m, 95)
                                    f_mask = (c1_raw_m >= p5_1) & (c1_raw_m <= p95_1) & (c2_raw_m >= p5_2) & (c2_raw_m <= p95_2)
                                    stat_c1, stat_c2 = c1_raw_m[f_mask], c2_raw_m[f_mask]
                                else:
                                    stat_c1, stat_c2 = c1_raw_m, c2_raw_m
                                    
                                datasets_to_run.append({'stat_c1': stat_c1, 'stat_c2': stat_c2, 'phase_name': "Raw Timing", 'window': window_name, 'filter': filter_name, 'bg': subtract_bg, 'lag': active_lag})
                                
                            if not np.isnan(active_lag):
                                c1_win_shift = c1_win.shift(-int(active_lag))
                                mask_lag = ~np.isnan(c1_win_shift) & ~np.isnan(c2_win)
                                c1_lag_m = c1_win_shift[mask_lag]
                                c2_lag_m = c2_win[mask_lag]
                                
                                for filter_name in ["None", "5_95_Filter"]:
                                    if filter_name == "5_95_Filter":
                                        if len(c1_lag_m) < 2: continue
                                        p5_1, p95_1 = np.percentile(c1_lag_m, 5), np.percentile(c1_lag_m, 95)
                                        p5_2, p95_2 = np.percentile(c2_lag_m, 5), np.percentile(c2_lag_m, 95)
                                        f_mask = (c1_lag_m >= p5_1) & (c1_lag_m <= p95_1) & (c2_lag_m >= p5_2) & (c2_lag_m <= p95_2)
                                        stat_c1, stat_c2 = c1_lag_m[f_mask], c2_lag_m[f_mask]
                                    else:
                                        stat_c1, stat_c2 = c1_lag_m, c2_lag_m
                                        
                                    datasets_to_run.append({'stat_c1': stat_c1, 'stat_c2': stat_c2, 'phase_name': "Lag Shifted", 'window': window_name, 'filter': filter_name, 'bg': subtract_bg, 'lag': active_lag})

                        for d in datasets_to_run:
                            c1_vals, c2_vals = d['stat_c1'], d['stat_c2']
                            if len(c1_vals) < 2: continue
                            
                            try:
                                abs_inc_c1 = round(c1_vals.max() - c1_vals.min(), 2)
                                abs_inc_c2 = round(c2_vals.max() - c2_vals.min(), 2)
                                mean_col1, std_col1 = round(np.mean(c1_vals), 2), round(np.std(c1_vals), 2)
                                mean_col2, std_col2 = round(np.mean(c2_vals), 2), round(np.std(c2_vals), 2)
                                
                                X = sm.add_constant(c2_vals) 
                                model = sm.OLS(c1_vals, X).fit()
                                r_squared = round(model.rsquared, 2)
                                
                                # ΑΣΦΑΛΗΣ ΥΠΟΛΟΓΙΣΜΟΣ ΚΛΙΣΗΣ (αν λείπουν τιμές)
                                if len(model.params) > 1:
                                    slope = round(model.params.iloc[1], 2)
                                    intercept = model.params.iloc[0] 
                                else:
                                    slope = np.nan
                                    intercept = model.params.iloc[0]
                                
                                rmse = round(np.sqrt(np.mean((c1_vals - c2_vals)**2)), 2)
                                mae = round(np.mean(np.abs(c1_vals - c2_vals)), 2)
                                mbe = round(np.mean(c1_vals - c2_vals), 2)
                                rae = round((np.sum(np.abs(c1_vals - c2_vals)) / np.sum(np.abs(c1_vals - np.mean(c1_vals)))), 2)
                                rbe = round((np.sum(c1_vals - c2_vals) / np.sum(np.abs(c1_vals - np.mean(c1_vals)))), 2)
                                nme = round((np.sum(np.abs(c1_vals - c2_vals)) / np.sum(c2_vals)), 2)
                                nmb = round((np.sum((c1_vals - c2_vals)) / np.sum(c2_vals)), 2)
                                
                                if d['filter'] == "None":
                                    stats_row = [
                                        date_str, substance_name, sensor_name, d['window'], abs_inc_c1, abs_inc_c2,
                                        mean_col1, std_col1, mean_col2, std_col2,
                                        r_squared, slope, d['lag'], rmse, mae, mbe, nme, nmb, rae, rbe
                                    ]
                                    stats_dict[(d['phase_name'], d['bg'], d['window'])].append(stats_row)

                                scatter_title = f"{substance_name} - {sensor_name} - {date_str} - {d['phase_name']} - {d['window']} - {d['filter']}"
                                fig_sc, ax_sc = plt.subplots(figsize=(10, 7))
                                ax_sc.scatter(c2_vals, c1_vals, color='blue', alpha=0.6, label='Data points')
                                if not np.isnan(slope):
                                    x_line = np.linspace(c2_vals.min(), c2_vals.max(), 100)
                                    y_line = slope * x_line + intercept
                                    ax_sc.plot(x_line, y_line, color='red', label=f'Fit: y = {slope}x + {round(intercept, 1)}\n$R^2$ = {r_squared}')
                                
                                ax_sc.set_title(f"{scatter_title}\n(Scatter Plot - {d['window']})", fontsize=16)
                                
                                # Αύξηση μεγέθους
                                ax_sc.set_xlabel('PTR-ToF-MS', fontsize=16)
                                ax_sc.set_ylabel(sensor_name, fontsize=16)
                                
                                # Αύξηση μεγέθους στους αριθμούς του άξονα
                                ax_sc.tick_params(axis='both', which='major', labelsize=14)
                                
                                ax_sc.set_ylim(bottom=0)
                                ax_sc.set_xlim(left=0)
                                
                                # Αύξηση μεγέθους υπομνήματος
                                ax_sc.legend(loc='best', framealpha=0.0, fontsize=14)
                                
                                timing_folder = d['phase_name'].replace(' ', '_')
                                filter_folder = "All_Data" if d['filter'] == "None" else "5_95_Filter"
                                
                                # Χρήση της νέας συνάρτησης για το όνομα του scatter plot
                                sc_filename = f"{make_safe_filename(scatter_title)}_Scatter.png"
                                
                                for base_sc_dir in [sc_twin_dir, sc_single_dir]:
                                    save_dir = os.path.join(base_sc_dir, d['window'], timing_folder, filter_folder)
                                    os.makedirs(save_dir, exist_ok=True)
                                    fig_sc.savefig(os.path.join(save_dir, sc_filename), bbox_inches='tight')
                                    
                                plt.close(fig_sc)
                            except Exception as stat_err:
                                print(f" Σφάλμα στα στατιστικά του Sensor {sensor_num}: {stat_err}")
                                continue
            except Exception as e_inner:
                print(f" Σφάλμα στην ανάλυση αυτού του block, προσπέραση: {e_inner}")
                pass
            
            n += 5

        empty_row = [np.nan] * 20 
        for key in stats_dict:
            stats_dict[key].append(empty_row)

    columns_list = [
        "Date", "Substance", "Sensor", "Window", "Abs Inc Sensor", "Abs Inc PTR-ToF-MS", 
        "Mean Sensor", "Std Sensor", "Mean PTR-ToF-MS", "Std PTR-ToF-MS",
        "R-squared", "Slope", "Time Lag", "RMSE", "MAE", "MBE", "nME", "nMB", "RAE", "RBE"
    ]
    
    sheet_mapping = {
        ('Raw Timing', True, 'Full'): 'Raw_BG_Full', ('Raw Timing', True, 'Peak_Only'): 'Raw_BG_Peak',
        ('Lag Shifted', True, 'Full'): 'Lagged_BG_Full', ('Lag Shifted', True, 'Peak_Only'): 'Lagged_BG_Peak',
        ('Raw Timing', False, 'Full'): 'Raw_NoBG_Full', ('Raw Timing', False, 'Peak_Only'): 'Raw_NoBG_Peak',
        ('Lag Shifted', False, 'Full'): 'Lagged_NoBG_Full', ('Lag Shifted', False, 'Peak_Only'): 'Lagged_NoBG_Peak',
    }
    
    with pd.ExcelWriter(output_excel_path) as writer:
        for key, sheet_name in sheet_mapping.items():
            df = pd.DataFrame(stats_dict[key], columns=columns_list)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
    print(f"\n Η επεξεργασία ολοκληρώθηκε! Το Excel αποθηκεύτηκε στο: {output_excel_path}")

# ==============================================================================
# ΕΚΤΕΛΕΣΗ
# ==============================================================================
process_batch_files(FILES_TO_PROCESS, OUTPUT_EXCEL, OUTPUT_PLOTS_TWIN_DIR, OUTPUT_PLOTS_SINGLE_DIR)

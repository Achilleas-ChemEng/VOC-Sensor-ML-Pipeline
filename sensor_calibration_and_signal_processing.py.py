# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.signal import correlate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import os
import re

def process_batch_files(file_paths, output_excel_path, output_plots_dir):
    if not os.path.exists(output_plots_dir):
        os.makedirs(output_plots_dir)

    # Λεξικό που θα κρατάει τα στατιστικά για τα 8 διαφορετικά σενάρια (Φύλλα Excel)
    stats_dict = {
        ('Raw Timing', True, 'Full'): [],
        ('Raw Timing', True, 'Peak_Only'): [],
        ('Lag Shifted', True, 'Full'): [],
        ('Lag Shifted', True, 'Peak_Only'): [],
        ('Raw Timing', False, 'Full'): [],
        ('Raw Timing', False, 'Peak_Only'): [],
        ('Lag Shifted', False, 'Full'): [],
        ('Lag Shifted', False, 'Peak_Only'): [],
    }

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        print(f"\n========================================================")
        print(f"ΕΠΕΞΕΡΓΑΣΙΑ ΑΡΧΕΙΟΥ: {filename}")
        print(f"========================================================")
        
        date_match = re.search(r'(\d{1,2}_\d{1,2}_\d{4})', filename)
        date_str = date_match.group(1) if date_match else filename.split(' data')[0]
        
        date_plots_dir = os.path.join(output_plots_dir, date_str)
        if not os.path.exists(date_plots_dir):
            os.makedirs(date_plots_dir)
            
        try:
            data = pd.read_excel(file_path, sheet_name='SELECTED DATA')
        except Exception as e:
            print(f"Σφάλμα ανάγνωσης αρχείου {filename}: {e}")
            continue

        n = 1 
        
        while n + 2 < len(data.columns):
            subset = data.iloc[:, n:n+3].dropna(how='all')
            if subset.empty:
                break

            substance_name = str(data.columns[n-1]).strip()
            
            safe_substance_name = substance_name.replace('/', '_').replace('\\', '_').replace(':', '')
            substance_plots_dir = os.path.join(date_plots_dir, safe_substance_name)
            if not os.path.exists(substance_plots_dir):
                os.makedirs(substance_plots_dir)
            
            base_columns = [n, n+1]
            comparison_columns = [n+2]
            
            print(f"\n--- Αναλύεται η ουσία: {substance_name} ---")
            
            baselines = {}
            unique_cols = base_columns + comparison_columns
            
            # Εύρεση Background (Minimum)
            for col_idx in unique_cols:
                col_data_clean = data.iloc[:, col_idx].dropna()
                baselines[col_idx] = col_data_clean.min() if not col_data_clean.empty else 0.0

            # Αληθινός άξονας χρόνου
            time_column = data.iloc[:, n-1]
            time_dt_full = pd.to_datetime(time_column.astype(str), errors='coerce')
            is_time = not time_dt_full.isna().all()
            if not is_time:
                time_dt_full = time_column.astype(str)

            # Εύρεση του "Peak" (για να βρούμε που τελειώνει η κορυφή)
            ptr_bg_sub = data.iloc[:, comparison_columns[0]] - baselines[comparison_columns[0]]
            if not ptr_bg_sub.dropna().empty:
                peak_idx = ptr_bg_sub.argmax()
                peak_val = ptr_bg_sub.iloc[peak_idx]
                threshold = 0.15 * peak_val 
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
                
                # ΟΡΓΑΝΩΣΗ ΒΑΣΙΚΩΝ ΥΠΟΦΑΚΕΛΩΝ
                ts_out_dir = os.path.join(substance_plots_dir, bg_suffix, "Time_Series")
                sc_out_dir = os.path.join(substance_plots_dir, bg_suffix, "Scatter_Plots")
                os.makedirs(ts_out_dir, exist_ok=True)
                os.makedirs(sc_out_dir, exist_ok=True)

                ptr_full = data.iloc[:, comparison_columns[0]] - baselines[comparison_columns[0]] if subtract_bg else data.iloc[:, comparison_columns[0]]
                s1_full = data.iloc[:, base_columns[0]] - baselines[base_columns[0]] if subtract_bg else data.iloc[:, base_columns[0]]
                s2_full = data.iloc[:, base_columns[1]] - baselines[base_columns[1]] if subtract_bg else data.iloc[:, base_columns[1]]
                
                windows = [("Full", len(ptr_full)), ("Peak_Only", end_idx + 1)]

                # ====================================================================
                # 1. ΔΗΜΙΟΥΡΓΙΑ ΚΟΙΝΩΝ TIME SERIES (Και οι 2 αισθητήρες + PTR)
                # ====================================================================
                for window_name, e_idx in windows:
                    t_win = time_dt_full.iloc[:e_idx]
                    p_win = ptr_full.iloc[:e_idx]
                    s1_win = s1_full.iloc[:e_idx]
                    s2_win = s2_full.iloc[:e_idx]
                    
                    fig_ts, ax_ts = plt.subplots(figsize=(10, 6))
                    
                    ax_ts.plot(t_win, s1_win, label='TVOCs Sensor 1', linestyle='-', color='blue')
                    ax_ts.plot(t_win, s2_win, label='TVOCs Sensor 2', linestyle='-', color='orange')
                    # ΑΛΛΑΓΗ ΕΔΩ: Η ουσία μπαίνει στο Legend της χρονοσειράς
                    ax_ts.plot(t_win, p_win, label=f'PTR-ToF-MS ({substance_name})', linestyle='--', color='green')
                    
                    base_title = f"{substance_name} - {date_str} - {window_name}"
                    ax_ts.set_title(f"{base_title}\n(Combined Time Series)")
                    ax_ts.set_xlabel('TIMESTAMP (UTC +02:00)')
                    y_label = 'Increase in Mixing Ratio (ppb)' if subtract_bg else 'Mixing Ratio (ppb)'
                    ax_ts.set_ylabel(y_label)
                    
                    # No Axis Standoff
                    ax_ts.margins(x=0) 
                    
                    ax_ts.legend()
                    
                    if is_time:
                        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    else:
                        tick_spacing = max(1, len(t_win) // 8)
                        ax_ts.set_xticks(np.arange(0, len(t_win), tick_spacing))
                        
                    fig_ts.autofmt_xdate() 
                    ts_filename = f"{base_title.replace('/', '_').replace(':', '')}_Combined_TS.png"
                    fig_ts.savefig(os.path.join(ts_out_dir, ts_filename), bbox_inches='tight')
                    plt.close(fig_ts)

                # ====================================================================
                # 2. ΥΠΟΛΟΓΙΣΜΟΣ ΣΤΑΤΙΣΤΙΚΩΝ ΚΑΙ SCATTER PLOTS
                # ====================================================================
                for cols in [(base_columns[0], comparison_columns[0]), (base_columns[1], comparison_columns[0])]:
                    sensor_num = "1" if cols[0] == base_columns[0] else "2"
                    sensor_name = f"TVOCs Sensor {sensor_num}"
                    
                    # ΑΛΛΑΓΗ ΕΔΩ: Καθαρό όνομα για τον άξονα Χ του Scatter (χωρίς την ουσία)
                    ref_name = "PTR-ToF-MS"

                    col1_full = s1_full if sensor_num == "1" else s2_full
                    col2_full = ptr_full
                    
                    mask_full = ~np.isnan(col1_full) & ~np.isnan(col2_full)
                    try:
                        c1_for_lag = col1_full[mask_full] - np.mean(col1_full[mask_full])
                        c2_for_lag = col2_full[mask_full] - np.mean(col2_full[mask_full])
                        correlation = correlate(c1_for_lag, c2_for_lag, mode='full')
                        time_lag = np.arange(-len(c1_for_lag) + 1, len(c1_for_lag))[np.argmax(correlation)]
                    except:
                        time_lag = np.nan

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
                                
                            datasets_to_run.append({
                                'stat_c1': stat_c1, 'stat_c2': stat_c2,
                                'phase_name': "Raw Timing",
                                'window': window_name, 'filter': filter_name, 'bg': subtract_bg
                            })
                            
                        if not np.isnan(time_lag):
                            c1_win_shift = c1_win.shift(-int(time_lag))
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
                                    
                                datasets_to_run.append({
                                    'stat_c1': stat_c1, 'stat_c2': stat_c2,
                                    'phase_name': "Lag Shifted",
                                    'window': window_name, 'filter': filter_name, 'bg': subtract_bg
                                })

                    for d in datasets_to_run:
                        c1_vals, c2_vals = d['stat_c1'], d['stat_c2']
                        if len(c1_vals) < 2: continue
                        
                        abs_inc_c1 = round(c1_vals.max() - c1_vals.min(), 2)
                        abs_inc_c2 = round(c2_vals.max() - c2_vals.min(), 2)
                        
                        mean_col1, std_col1 = round(np.mean(c1_vals), 2), round(np.std(c1_vals), 2)
                        mean_col2, std_col2 = round(np.mean(c2_vals), 2), round(np.std(c2_vals), 2)
                        
                        try:
                            X = sm.add_constant(c2_vals) 
                            model = sm.OLS(c1_vals, X).fit()
                            r_squared, slope = round(model.rsquared, 2), round(model.params.iloc[1], 2)
                            intercept = model.params.iloc[0] 
                        except:
                            r_squared, slope, intercept = np.nan, np.nan, np.nan
                        
                        rmse = round(np.sqrt(np.mean((c1_vals - c2_vals)**2)), 2)
                        mae = round(np.mean(np.abs(c1_vals - c2_vals)), 2)
                        mbe = round(np.mean(c1_vals - c2_vals), 2)
                        rae = round((np.sum(np.abs(c1_vals - c2_vals)) / np.sum(np.abs(c1_vals - np.mean(c1_vals)))), 2)
                        rbe = round((np.sum(c1_vals - c2_vals) / np.sum(np.abs(c1_vals - np.mean(c1_vals)))), 2)
                        nme = round((np.sum(np.abs(c1_vals - c2_vals)) / np.sum(c2_vals)), 2)
                        nmb = round((np.sum((c1_vals - c2_vals)) / np.sum(c2_vals)), 2)
                        
                        # Προσθέτουμε στο Excel ΜΟΝΟ αν το filter είναι "None" (όλα τα δεδομένα)
                        if d['filter'] == "None":
                            stats_row = [
                                date_str, substance_name, sensor_name, d['window'],
                                abs_inc_c1, abs_inc_c2,
                                mean_col1, std_col1, mean_col2, std_col2,
                                r_squared, slope, time_lag, rmse, mae, mbe, nme, nmb, rae, rbe
                            ]
                            
                            dict_key = (d['phase_name'], d['bg'], d['window'])
                            stats_dict[dict_key].append(stats_row)

                        # Δημιουργία SCATTER PLOT
                        scatter_title = f"{substance_name} - {sensor_name} - {date_str} - {d['phase_name']} - {d['window']} - {d['filter']}"
                        fig_sc, ax_sc = plt.subplots(figsize=(8, 6))
                        ax_sc.scatter(c2_vals, c1_vals, color='blue', alpha=0.6, label='Data points')
                        if not np.isnan(slope):
                            x_line = np.linspace(c2_vals.min(), c2_vals.max(), 100)
                            y_line = slope * x_line + intercept
                            ax_sc.plot(x_line, y_line, color='red', label=f'Fit: y = {slope}x + {round(intercept, 1)}\n$R^2$ = {r_squared}')
                        
                        ax_sc.set_title(f"{scatter_title}\n(Scatter Plot - {d['window']})")
                        ax_sc.set_xlabel(f'{ref_name} Mixing Ratio (ppb)')
                        ax_sc.set_ylabel(f'{sensor_name} Mixing Ratio (ppb)')
                        
                        # No Axis Standoff στα scatter
                        ax_sc.margins(0)
                        
                        ax_sc.legend()
                        
                        window_folder = d['window']
                        timing_folder = d['phase_name'].replace(' ', '_')
                        filter_folder = "All_Data" if d['filter'] == "None" else "5_95_Filter"
                        
                        save_dir = os.path.join(sc_out_dir, window_folder, timing_folder, filter_folder)
                        os.makedirs(save_dir, exist_ok=True)
                        
                        sc_filename = f"{scatter_title.replace('/', '_').replace(':', '')}_Scatter.png"
                        fig_sc.savefig(os.path.join(save_dir, sc_filename), bbox_inches='tight')
                        plt.close(fig_sc)

            n += 5

        empty_row = [np.nan] * 20 
        for key in stats_dict:
            stats_dict[key].append(empty_row)

    columns_list = [
        "Date", "Substance", "Sensor", "Window", 
        "Abs Inc Sensor", "Abs Inc PTR", 
        "Mean Sensor", "Std Sensor", "Mean PTRMS", "Std PTRMS",
        "R-squared", "Slope", "Time Lag", "RMSE", "MAE", "MBE", "nME", "nMB", "RAE", "RBE"
    ]
    
    sheet_mapping = {
        ('Raw Timing', True, 'Full'): 'Raw_BG_Full',
        ('Raw Timing', True, 'Peak_Only'): 'Raw_BG_Peak',
        ('Lag Shifted', True, 'Full'): 'Lagged_BG_Full',
        ('Lag Shifted', True, 'Peak_Only'): 'Lagged_BG_Peak',
        ('Raw Timing', False, 'Full'): 'Raw_NoBG_Full',
        ('Raw Timing', False, 'Peak_Only'): 'Raw_NoBG_Peak',
        ('Lag Shifted', False, 'Full'): 'Lagged_NoBG_Full',
        ('Lag Shifted', False, 'Peak_Only'): 'Lagged_NoBG_Peak',
    }
    
    with pd.ExcelWriter(output_excel_path) as writer:
        for key, sheet_name in sheet_mapping.items():
            df = pd.DataFrame(stats_dict[key], columns=columns_list)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
    print(f"\n✅ Η επεξεργασία ολοκληρώθηκε! Το Excel αποθηκεύτηκε στο: {output_excel_path}")

# ========================== USER SETTINGS ==========================
FILES_TO_PROCESS = [
    r"C:\Users\temp\Desktop\01_12_2025 data comparison.xlsx",
    r"C:\Users\temp\Desktop\08_12_2025 data comparison.xlsx",
    r"C:\Users\temp\Desktop\23_01_2026 data comparison.xlsx",
    r"C:\Users\temp\Desktop\30_01_2026 data comparison.xlsx",
    r"C:\Users\temp\Desktop\20_02_2026 data comparison.xlsx",
    r"C:\Users\temp\Desktop\24_02_2026 data comparison.xlsx",
    r"C:\Users\temp\Desktop\28_02_2026 data comparison.xlsx",
    r"C:\Users\temp\Desktop\01_03_2026 data comparison.xlsx"
]

OUTPUT_EXCEL = r"C:\Users\temp\Desktop\All_Experiments_Results4.xlsx"
OUTPUT_PLOTS_FOLDER = r"C:\Users\temp\Desktop\Sensor_Plots_Organized"
# ===================================================================

process_batch_files(FILES_TO_PROCESS, OUTPUT_EXCEL, OUTPUT_PLOTS_FOLDER)
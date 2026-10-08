# VOC Sensor Evaluation Pipeline (Machine Learning & Signal Processing)

Welcome to my repository! This codebase contains the analytical pipeline I developed for my Chemical Engineering Diploma Thesis. 

#What is this project about?
I wanted to evaluate how reliable low-cost Volatile Organic Compound (VOC) sensors actually are when compared to a highly accurate (and very expensive) reference instrument, the PTR-ToF-MS. 

Since data were incredibly noisy and messy, I couldn't just compare the numbers in Excel. I built this custom Python pipeline to clean the signals, mathematically align the time delays, and use Machine Learning to figure out exactly which chemicals were triggering the sensors.

# The Code

## 1. `sensor_calibration_and_signal_processing.py`
This script handles the raw data, signal processing, and statistical validation:
* **Peak Detection :** It automatically finds the maximum concentration peaks and dynamically trims the signal tail (below a 5% threshold).
* **Fixing Time Delays:** It uses cross-correlation (`scipy.signal.correlate`) to automatically calculate and correct the time lag between the PTR-MS and the cheap sensors.
* **Metrics:** After aligning the data, it calculates key validation metrics like R^2, RMSE, and NME.

## 2. `ml_compound_importance_and_identification.py`
This is the more advanced part of the pipeline. I wanted to know *which* specific VOCs (m/z channels) were driving the sensor responses.
* **Feature Importance:** I trained Random Forest and Gradient Boosting models to determine the non-linear impact of specific chemical compounds on the sensors.
* **Avoiding Overfitting (LOO-CV):** Because experimental sample sizes can be small, I implemented Leave-One-Out Cross-Validation to keep the models robust and avoid overfitting. I also used False Discovery Rate (FDR) correction to ensure the correlations were statistically significant and not just luck.
* **Automated Identification:** The script takes the significant m/z values, calculates potential chemical formulas, and maps them against the GLOVOCS database to identify the actual compounds.

## Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `scipy`, `matplotlib`

"""
PSD analysis of a KineCal recording.

Loads a recording CSV (all joint positions) and plots power spectral
density for:
  • ML axis  — hip, shoulder, ankle midpoints + Dempster CoM
  • AP axis  — current trunk-length proxy + hip/shoulder z_norm (MediaPipe
               monocular depth estimate — potential free AP improvement)

Use this to:
  1. Validate that the Butterworth cutoff sits at the noise knee
  2. Compare signal quality across joints
  3. Assess whether z_norm is a better AP proxy than trunk length

Usage
-----
    python analyse_psd.py                      # picks most recent CSV
    python analyse_psd.py recording_XYZ.csv   # specific file
"""

import sys
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt

# ── Filter settings (must match live_preview.py) ─────────────────────────────
FC_ML  = 6.0
FC_AP  = 3.0
ORDER  = 2


def apply_butter(signal, fc, fs):
    Wn   = min(2.0 * fc / fs, 0.99)
    b, a = butter(ORDER, Wn, btype='low')
    out  = signal.copy().astype(float)
    nans = np.isnan(out)
    if nans.any() and not nans.all():
        out[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], out[~nans])
    if len(out) >= 15:
        out = filtfilt(b, a, out)
    return out


def psd(signal, fs):
    clean = signal.copy().astype(float)
    nans  = np.isnan(clean)
    if nans.any() and not nans.all():
        clean[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], clean[~nans])
    elif nans.all():
        return np.array([0.0]), np.array([0.0])
    nperseg = min(len(clean), int(fs * 4))
    f, p    = welch(clean, fs=fs, nperseg=nperseg, scaling='density')
    return f, p


def midpoint(df, col_a, col_b):
    """Mean of two columns, NaN where both are NaN."""
    return df[[col_a, col_b]].mean(axis=1)


def mean_centre(s):
    return s - np.nanmean(s)


def main():
    # ── Load CSV ──────────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        csvs = sorted(glob.glob('recording_*.csv'), key=os.path.getmtime)
        if not csvs:
            print("No recording_*.csv files found.  Run a trial first.")
            sys.exit(1)
        path = csvs[-1]
        print(f"Loading: {path}")

    df = pd.read_csv(path)
    print(f"  {len(df)} frames   {len(df.columns)} columns")

    # ── Sample rate ───────────────────────────────────────────────────────────
    dt  = np.diff(df['time_s'].values)
    fps = 1.0 / np.median(dt)
    dur = df['time_s'].iloc[-1]
    print(f"  Median FPS: {fps:.1f}   Duration: {dur:.1f}s   "
          f"Max gap: {np.max(dt)*1000:.1f}ms")

    # ── Build signals ─────────────────────────────────────────────────────────
    def col(name):
        return df[name].values if name in df.columns else np.full(len(df), np.nan)

    # ML signals (x = left-right in image, cm)
    hip_ml    = mean_centre(midpoint(df, 'left_hip_x_cm',      'right_hip_x_cm'))
    sho_ml    = mean_centre(midpoint(df, 'left_shoulder_x_cm', 'right_shoulder_x_cm'))
    ank_ml    = mean_centre(midpoint(df, 'left_ankle_x_cm',    'right_ankle_x_cm'))
    com_ml    = mean_centre(col('com_ml_cm'))

    # AP signals
    ap_proxy  = mean_centre(col('ap_cm'))                         # trunk-length proxy
    ap_radar  = mean_centre(col('ap_radar_cm'))                   # IWR6843 radar (NaN if not recorded)
    hip_z     = mean_centre(midpoint(df, 'left_hip_z_norm',      'right_hip_z_norm'))
    sho_z     = mean_centre(midpoint(df, 'left_shoulder_z_norm', 'right_shoulder_z_norm'))
    ank_z     = mean_centre(midpoint(df, 'left_ankle_z_norm',    'right_ankle_z_norm'))

    # ── Plot layout: 2 rows × 2 cols ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"PSD analysis — {os.path.basename(path)}\n"
        f"FPS: {fps:.1f}   Frames: {len(df)}   Duration: {dur:.0f}s",
        fontsize=12
    )

    nyquist  = fps / 2.0
    xlim     = min(nyquist * 0.95, 25.0)
    sway_col = '#4CAF50'

    # ── Top-left: ML signals ──────────────────────────────────────────────────
    ax = axes[0, 0]
    for sig, label, col_hex in [
        (hip_ml,  'Hip midpoint',      '#2196F3'),
        (sho_ml,  'Shoulder midpoint', '#9C27B0'),
        (ank_ml,  'Ankle midpoint',    '#FF9800'),
        (com_ml,  'Dempster CoM',      '#F44336'),
    ]:
        f, p = psd(sig, fps)
        ax.semilogy(f, p, color=col_hex, alpha=0.7, linewidth=1.5, label=label)

    ax.axvline(FC_ML, color='red', linestyle='--', linewidth=1.5,
               label=f'Current cutoff {FC_ML} Hz')
    ax.axvspan(0, 3, alpha=0.08, color=sway_col, label='Typical sway (0–3 Hz)')
    ax.set_title('ML axis — joint comparison')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (cm² / Hz)')
    ax.set_xlim(0, xlim)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # ── Top-right: ML filtered vs raw (hip only) ──────────────────────────────
    ax = axes[0, 1]
    f_r, p_r = psd(hip_ml, fps)
    f_f, p_f = psd(apply_butter(hip_ml, FC_ML, fps), fps)
    ax.semilogy(f_r, p_r, color='#2196F3', alpha=0.4, linewidth=1, label='Hip ML raw')
    ax.semilogy(f_f, p_f, color='#2196F3', linewidth=2,             label=f'Hip ML filtered ({FC_ML} Hz)')
    ax.axvline(FC_ML, color='red', linestyle='--', linewidth=1.5, label=f'Cutoff {FC_ML} Hz')
    ax.axvspan(0, 3, alpha=0.08, color=sway_col, label='Typical sway (0–3 Hz)')
    ax.set_title('ML axis — filter effect (hip)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (cm² / Hz)')
    ax.set_xlim(0, xlim)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # ── Bottom-left: AP proxy vs z_norm signals ───────────────────────────────
    ax = axes[1, 0]
    for sig, label, col_hex in [
        (ap_proxy, 'Trunk-length proxy (current)', '#FF9800'),
        (ap_radar, 'IWR6843 radar range',           '#F44336'),
        (hip_z,    'Hip z_norm (MediaPipe depth)',  '#03A9F4'),
        (sho_z,    'Shoulder z_norm',               '#9C27B0'),
        (ank_z,    'Ankle z_norm',                  '#8BC34A'),
    ]:
        f, p = psd(sig, fps)
        ax.semilogy(f, p, alpha=0.7, linewidth=1.5, color=col_hex, label=label)

    ax.axvline(FC_AP, color='red', linestyle='--', linewidth=1.5,
               label=f'Current AP cutoff {FC_AP} Hz')
    ax.axvspan(0, 3, alpha=0.08, color=sway_col, label='Typical sway (0–3 Hz)')
    ax.set_title('AP axis — proxy comparison')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (units² / Hz)')
    ax.set_xlim(0, xlim)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # ── Bottom-right: signal-to-noise summary bar chart ───────────────────────
    ax = axes[1, 1]

    def snr_db(sig, fs, signal_band=(0.1, 3.0), noise_band=(8.0, 20.0)):
        """Ratio of power in sway band vs noise band (dB)."""
        f, p = psd(sig, fps)
        s_mask = (f >= signal_band[0]) & (f <= signal_band[1])
        n_mask = (f >= noise_band[0])  & (f <= noise_band[1])
        if s_mask.sum() == 0 or n_mask.sum() == 0:
            return 0.0
        s_pwr = np.mean(p[s_mask])
        n_pwr = np.mean(p[n_mask])
        return 10 * np.log10(s_pwr / max(n_pwr, 1e-30))

    signals = [
        ('Hip ML',        hip_ml,   '#2196F3'),
        ('Shoulder ML',   sho_ml,   '#9C27B0'),
        ('Ankle ML',      ank_ml,   '#FF9800'),
        ('CoM ML',        com_ml,   '#F44336'),
        ('AP proxy',      ap_proxy, '#FF9800'),
        ('AP radar',      ap_radar, '#F44336'),
        ('Hip z_norm',    hip_z,    '#03A9F4'),
        ('Shoulder z',    sho_z,    '#9C27B0'),
        ('Ankle z',       ank_z,    '#8BC34A'),
    ]

    names  = [s[0] for s in signals]
    snrs   = [snr_db(s[1], fps) for s in signals]
    colors = [s[2] for s in signals]

    bars = ax.barh(names, snrs, color=colors, alpha=0.8)
    ax.axvline(0, color='white', linewidth=0.5)
    ax.set_xlabel('SNR — sway band vs noise floor (dB)')
    ax.set_title('Signal quality comparison\n(higher = more sway content vs noise)')
    ax.grid(True, axis='x', alpha=0.3)

    # Label bars
    for bar, val in zip(bars, snrs):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f} dB', va='center', fontsize=8, color='white')

    plt.tight_layout()

    out_png = path.replace('.csv', '_psd.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot → {out_png}")
    plt.show()


if __name__ == '__main__':
    main()

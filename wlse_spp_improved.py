#!/usr/bin/env python3
"""AAE4203 Lab 1 — Improved Multi-Constellation WLS + RAIM SPP Solver & Pure-Python RINEX Processor.

Supports:
1. Direct RINEX 3 (.obs + .nav) multi-constellation (GPS + Galileo + BeiDou) SPP solving
   with Klobuchar ionospheric correction, Saastamoinen tropospheric correction, TGD,
   Sagnac correction, combined sin^2(el) * C/N0 weighting, and RAIM outlier rejection.
2. Standard CSV-matrix input mode (--input-dir) compatible with official `rinex2csv.m`
   outputs (`benchmark/gps_csv` or `outputs/own_inputs`).
3. Direct extraction of receiver standalone solutions (`NAV-HPPOSECEF`) from `.ubx` binary logs.
"""
import argparse
import csv
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

C = 299792458.0
MU_GPS = 3.986005e14
MU_GAL = 3.986004418e14
MU_BDS = 3.986004418e14
OMGE_GPS = 7.2921151467e-5
OMGE_BDS = 7.2921150e-5
FREQ_GPS_L1 = 1575.42e6
FREQ_BDS_B1I = 1561.098e6

HEADER = ['gps_week', 'gps_tow_s', 'x_ecef_m', 'y_ecef_m', 'z_ecef_m']


def ymdhms_to_gpst(y, m, d, hh, mm, ss):
    sec_int = int(math.floor(ss))
    frac = ss - sec_int
    dt0 = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
    dt = datetime(y, m, d, hh, mm, sec_int, tzinfo=timezone.utc)
    diff = (dt - dt0).total_seconds() + frac
    week = int(diff // 604800)
    tow = diff - week * 604800
    return week, tow


def ecef_to_lla(x, y, z):
    a = 6378137.0
    e2 = 6.69437999014e-3
    b = a * math.sqrt(1.0 - e2)
    ep2 = (a**2 - b**2) / (b**2)
    p = math.hypot(x, y)
    th = math.atan2(a * z, b * p)
    lon = math.atan2(y, x)
    lat = math.atan2(z + ep2 * b * (math.sin(th)**3), p - e2 * a * (math.cos(th)**3))
    n_val = a / math.sqrt(1.0 - e2 * (math.sin(lat)**2))
    alt = p / math.cos(lat) - n_val
    return math.degrees(lat), math.degrees(lon), alt


def ecef_to_enu(x, y, z, xr, yr, zr):
    lat, lon, _ = ecef_to_lla(xr, yr, zr)
    phi, lam = math.radians(lat), math.radians(lon)
    dx, dy, dz = x - xr, y - yr, z - zr
    e = -math.sin(lam) * dx + math.cos(lam) * dy
    n = -math.sin(phi) * math.cos(lam) * dx - math.sin(phi) * math.sin(lam) * dy + math.cos(phi) * dz
    u = math.cos(phi) * math.cos(lam) * dx + math.cos(phi) * math.sin(lam) * dy + math.sin(phi) * dz
    return np.array([e, n, u])


def sat_az_el(rx_xyz, sat_xyz):
    lat, lon, h = ecef_to_lla(*rx_xyz)
    enu = ecef_to_enu(*sat_xyz, *rx_xyz)
    dist = np.linalg.norm(enu)
    el = math.degrees(math.asin(np.clip(enu[2] / dist, -1.0, 1.0)))
    az = math.degrees(math.atan2(enu[0], enu[1])) % 360.0
    return az, el, lat, lon, h


def saastamoinen_trop(lat_deg, h_m, el_deg):
    if el_deg <= 0 or h_m < -1000 or h_m > 20000:
        return 0.0
    p = 1013.25 * ((1.0 - 2.2557e-5 * max(0.0, h_m)) ** 5.2568)
    t_k = 15.0 - 6.5e-3 * max(0.0, h_m) + 273.16
    e = 6.108 * math.exp((17.15 * t_k - 4684.0) / (t_k - 38.45)) * 0.7
    z = math.radians(90.0 - el_deg)
    trph = 0.0022768 * p / (1.0 - 0.00266 * math.cos(2.0 * math.radians(lat_deg)) - 0.00028 * h_m / 1000.0) / math.cos(z)
    trpw = 0.002277 * (1255.0 / t_k + 0.05) * e / math.cos(z)
    return trph + trpw


def klobuchar_iono(lat_deg, lon_deg, el_deg, az_deg, tow):
    alpha = [0.1118e-07, -0.7451e-08, -0.5960e-07, 0.1192e-06]
    beta = [0.1167e+06, -0.2294e+06, -0.1311e+06, 0.1049e+07]
    el_sc = el_deg / 180.0
    az_rad = math.radians(az_deg)
    psi = 0.0137 / (el_sc + 0.11) - 0.022
    phi_i = max(-0.416, min(0.416, lat_deg / 180.0 + psi * math.cos(az_rad)))
    lam_i = lon_deg / 180.0 + psi * math.sin(az_rad) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)
    t = (43200.0 * lam_i + tow) % 86400.0
    f = 1.0 + 16.0 * ((0.53 - el_sc) ** 3)
    amp = max(0.0, sum(alpha[i] * (phi_m ** i) for i in range(4)))
    per = max(72000.0, sum(beta[i] * (phi_m ** i) for i in range(4)))
    x = 2.0 * math.pi * (t - 50400.0) / per
    if abs(x) < 1.57:
        delay = f * (5e-9 + amp * (1.0 - x*x/2.0 + (x**4)/24.0))
    else:
        delay = f * 5e-9
    return C * delay


def parse_rinex_nav(nav_path):
    lines = Path(nav_path).read_text(encoding='utf-8', errors='ignore').splitlines()
    idx = 0
    while idx < len(lines):
        if 'END OF HEADER' in lines[idx]:
            idx += 1
            break
        idx += 1
    ephs = {}
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        sys_ch = line[0]
        prn = line[:3].strip()
        if sys_ch in ('G', 'E', 'C', 'J'):
            rec_lines = lines[idx:idx+8]
            idx += 8
            y, m, d = int(rec_lines[0][4:8]), int(rec_lines[0][9:11]), int(rec_lines[0][12:14])
            hh, mm, ss = int(rec_lines[0][15:17]), int(rec_lines[0][18:20]), float(rec_lines[0][21:23])
            toc_week, toc_tow = ymdhms_to_gpst(y, m, d, hh, mm, ss)
            if sys_ch == 'C':
                toc_tow += 14.0
            vals = []
            for c in range(23, len(rec_lines[0]), 19):
                s = rec_lines[0][c:c+19].replace('D', 'E').replace('d', 'e').strip()
                vals.append(float(s) if s else 0.0)
            for rline in rec_lines[1:]:
                for c in range(4, 4 + 4*19, 19):
                    s = rline[c:c+19].replace('D', 'E').replace('d', 'e').strip() if c < len(rline) else ''
                    vals.append(float(s) if s else 0.0)
            eph = {
                'prn': prn, 'sys': sys_ch,
                'toc': toc_week * 604800.0 + toc_tow,
                'af0': vals[0], 'af1': vals[1], 'af2': vals[2],
                'iode': vals[3], 'crs': vals[4], 'deln': vals[5], 'M0': vals[6],
                'cuc': vals[7], 'e': vals[8], 'cus': vals[9], 'sqrtA': vals[10],
                'toe_s': vals[11], 'cic': vals[12], 'OMG0': vals[13], 'cis': vals[14],
                'i0': vals[15], 'crc': vals[16], 'omg': vals[17], 'OMGd': vals[18],
                'idot': vals[19], 'code': vals[20], 'week': vals[21],
                'sva': vals[23], 'svh': int(vals[24]), 'tgd': vals[25],
            }
            eph['toe'] = toc_week * 604800.0 + eph['toe_s'] + (14.0 if sys_ch == 'C' else 0.0)
            ephs.setdefault(prn, []).append(eph)
        elif sys_ch == 'R':
            idx += 4
        else:
            idx += 1
    return ephs


def sat_pos_clk(eph, t_tx):
    sys = eph['sys']
    mu = MU_GPS if sys == 'G' else (MU_GAL if sys == 'E' else MU_BDS)
    omge = OMGE_GPS if sys in ('G', 'E') else OMGE_BDS
    A = eph['sqrtA'] ** 2
    n0 = math.sqrt(mu / (A ** 3))
    tk = t_tx - eph['toe']
    if tk > 302400: tk -= 604800
    elif tk < -302400: tk += 604800
    n = n0 + eph['deln']
    M = eph['M0'] + n * tk
    E = M
    for _ in range(10):
        E = M + eph['e'] * math.sin(E)
    sinE, cosE = math.sin(E), math.cos(E)
    v = math.atan2(math.sqrt(1.0 - eph['e']**2) * sinE, cosE - eph['e'])
    phi = v + eph['omg']
    sin2p, cos2p = math.sin(2.0 * phi), math.cos(2.0 * phi)
    u = phi + eph['cuc'] * cos2p + eph['cus'] * sin2p
    r = A * (1.0 - eph['e'] * cosE) + eph['crc'] * cos2p + eph['crs'] * sin2p
    i = eph['i0'] + eph['idot'] * tk + eph['cic'] * cos2p + eph['cis'] * sin2p
    x_orb, y_orb = r * math.cos(u), r * math.sin(u)
    if sys == 'C' and (int(eph['prn'][1:]) <= 5 or int(eph['prn'][1:]) >= 59):
        OMG = eph['OMG0'] + eph['OMGd'] * tk - omge * eph['toe_s']
        xg = x_orb * math.cos(OMG) - y_orb * math.cos(i) * math.sin(OMG)
        yg = x_orb * math.sin(OMG) + y_orb * math.cos(i) * math.cos(OMG)
        zg = y_orb * math.sin(i)
        sinO, cosO = math.sin(omge * tk), math.cos(omge * tk)
        rx = math.radians(-5.0)
        sin5, cos5 = math.sin(rx), math.cos(rx)
        x1, y1, z1 = xg, yg * cos5 + zg * sin5, -yg * sin5 + zg * cos5
        x, y, z = x1 * cosO + y1 * sinO, -x1 * sinO + y1 * cosO, z1
    else:
        OMG = eph['OMG0'] + (eph['OMGd'] - omge) * tk - omge * eph['toe_s']
        x = x_orb * math.cos(OMG) - y_orb * math.cos(i) * math.sin(OMG)
        y = x_orb * math.sin(OMG) + y_orb * math.cos(i) * math.cos(OMG)
        z = y_orb * math.sin(i)
    tc = t_tx - eph['toc']
    if tc > 302400: tc -= 604800
    elif tc < -302400: tc += 604800
    dts = eph['af0'] + eph['af1'] * tc + eph['af2'] * (tc ** 2)
    dts -= 2.0 * math.sqrt(mu * A) * eph['e'] * sinE / (C ** 2)
    return np.array([x, y, z]), dts * C


def parse_rinex_obs(obs_path):
    lines = Path(obs_path).read_text(encoding='utf-8', errors='ignore').splitlines()
    idx = 0
    while idx < len(lines):
        if 'END OF HEADER' in lines[idx]:
            idx += 1
            break
        idx += 1
    epochs_data = []
    while idx < len(lines):
        line = lines[idx]
        if line.startswith('>'):
            parts = line[1:].split()
            y, m, d, hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), float(parts[5])
            nsat = int(parts[7])
            week, tow = ymdhms_to_gpst(y, m, d, hh, mm, ss)
            idx += 1
            sats = {}
            for _ in range(nsat):
                sline = lines[idx]
                idx += 1
                prn = sline[:3].strip()
                pr_str = sline[3:17].strip() if len(sline) >= 17 else ''
                snr_str = sline[51:65].strip() if len(sline) >= 65 else ''
                pr = float(pr_str) if pr_str else float('nan')
                snr = float(snr_str) if snr_str else float('nan')
                sats[prn] = (pr, snr)
            epochs_data.append((week, tow, sats))
        else:
            idx += 1
    return epochs_data


def solve_rinex_epoch(w, tow, sats, ephs, x_seed, use_sys=('G', 'E', 'C'), weight_mode='cn0', raim=True):
    t_rx = w * 604800.0 + tow
    items = []
    for prn, (pr, snr) in sats.items():
        sys = prn[0]
        if sys not in use_sys or prn not in ephs or not math.isfinite(pr) or pr < 1e7:
            continue
        eph = min(ephs[prn], key=lambda e: abs(t_rx - e['toc']))
        if eph['svh'] != 0:
            continue
        t_tx = t_rx - pr / C
        sp, clk_m = sat_pos_clk(eph, t_tx)
        t_tx = t_rx - (pr + clk_m) / C
        sp, clk_m = sat_pos_clk(eph, t_tx)
        tgd_m = eph['tgd'] * C
        items.append({'prn': prn, 'sys': sys, 'sp': sp, 'pr': pr, 'clk_m': clk_m, 'tgd_m': tgd_m, 'snr': snr})

    def run_wls(cur_items, x_init):
        active_s = sorted({it['sys'] for it in cur_items})
        if len(cur_items) < 3 + len(active_s):
            return None, None, None
        x = np.zeros(3 + len(active_s), dtype=float)
        x[:3] = x_init[:3]
        for it_step in range(15):
            H, v, weights, used = [], [], [], []
            has_pos = np.linalg.norm(x[:3]) > 6e6
            for it in cur_items:
                sp = it['sp']
                delta = sp - x[:3]
                r = np.linalg.norm(delta)
                omge = OMGE_BDS if it['sys'] == 'C' else OMGE_GPS
                sagnac = omge / C * (sp[0]*x[1] - sp[1]*x[0])
                if has_pos and it_step >= 2:
                    az, el, lat, lon, h = sat_az_el(x[:3], sp)
                    if el < 15.0 or it['snr'] < 15.0:
                        continue
                    ion_l1 = klobuchar_iono(lat, lon, el, az, tow)
                    ion = ion_l1 * ((FREQ_GPS_L1 / FREQ_BDS_B1I) ** 2) if it['sys'] == 'C' else ion_l1
                    trp = saastamoinen_trop(lat, h, el)
                    if weight_mode == 'cn0':
                        w_i = (math.sin(math.radians(el)) ** 2) * (10.0 ** ((it['snr'] - 30.0) / 10.0))
                    else:
                        w_i = math.sin(math.radians(el)) ** 2
                else:
                    ion, trp, w_i = 0.0, 0.0, 1.0
                s_idx = active_s.index(it['sys'])
                row = [-(sp[0]-x[0])/r, -(sp[1]-x[1])/r, -(sp[2]-x[2])/r] + [1.0 if k == s_idx else 0.0 for k in range(len(active_s))]
                res = (it['pr'] + it['clk_m'] - ion - trp - it['tgd_m']) - (r + sagnac + x[3 + s_idx])
                H.append(row); v.append(res); weights.append(w_i); used.append(it)
            if len(H) < 3 + len(active_s):
                return None, None, None
            H_arr, v_arr, W_arr = np.array(H), np.array(v), np.sqrt(np.array(weights))
            dx, _, rank, _ = np.linalg.lstsq(H_arr * W_arr[:, None], v_arr * W_arr, rcond=None)
            if rank < 3 + len(active_s):
                return None, None, None
            x += dx
            if it_step >= 3 and np.linalg.norm(dx[:3]) < 1e-4:
                break
        if not (6.30e6 < np.linalg.norm(x[:3]) < 6.42e6):
            return None, None, None
        return x[:3], v_arr, used

    pos, res, used = run_wls(items, x_seed)
    if pos is None:
        return None
    if raim and len(used) >= 6:
        med = np.median(res)
        abs_dev = np.abs(res - med)
        if np.max(abs_dev) > 15.0:
            bad_idx = int(np.argmax(abs_dev))
            filtered = [u for idx, u in enumerate(used) if idx != bad_idx]
            pos2, _, _ = run_wls(filtered, pos)
            if pos2 is not None:
                pos = pos2
    return pos


def read_csv_inputs(folder):
    folder = Path(folder)
    names = ['pseudoranges_meas', 'satellite_clock_bias', 'ionospheric_delay',
             'tropospheric_delay', 'satellite_positions']
    arrays = {name: np.loadtxt(folder / (name + '.csv'), delimiter=',', ndmin=2) for name in names}
    shape = arrays['pseudoranges_meas'].shape
    with (folder / 'epochs.csv').open(encoding='utf-8-sig', newline='') as stream:
        epochs = [(int(r['gps_week']), float(r['gps_tow_s'])) for r in csv.DictReader(stream)]
    arrays['group_delay'] = (np.loadtxt(folder / 'group_delay.csv', delimiter=',', ndmin=2)
                             if (folder / 'group_delay.csv').exists() else np.zeros(shape))
    return arrays, epochs


def solve_csv_epoch_with_raim(sp, pr, clk, ion, trp, tgd, seed, raim=True):
    def wls_step(sp_sub, pr_sub, clk_sub, ion_sub, trp_sub, tgd_sub, x_init):
        if len(sp_sub) < 4:
            return None, None
        x = np.r_[x_init.copy(), 0.0]
        corrected = pr_sub + clk_sub - ion_sub - trp_sub - tgd_sub
        for _ in range(30):
            delta = sp_sub - x[:3]
            ranges = np.linalg.norm(delta, axis=1)
            if np.any(ranges <= 0):
                return None, None
            H = np.column_stack((-delta / ranges[:, None], np.ones(len(sp_sub))))
            sagnac = OMGE_GPS / C * (sp_sub[:, 0]*x[1] - sp_sub[:, 1]*x[0])
            if np.linalg.norm(x[:3]) > 6e6:
                lat, lon, _ = ecef_to_lla(*x[:3])
                phi, lam = np.radians([lat, lon])
                up = np.array([np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)])
                sine = delta @ up / ranges
                weights = np.maximum(sine, 0.05) ** 2
            else:
                weights = np.ones(len(sp_sub))
            scale = np.sqrt(weights)
            res = corrected - ranges - sagnac - x[3]
            dx, _, rank, _ = np.linalg.lstsq(H * scale[:, None], res * scale, rcond=None)
            if rank < 4:
                return None, None
            x += dx
            if np.linalg.norm(dx) < 1e-4:
                if np.isfinite(x).all() and 6.30e6 < np.linalg.norm(x[:3]) < 6.42e6:
                    return x[:3], res
                break
        return None, None

    pos, res = wls_step(sp, pr, clk, ion, trp, tgd, seed)
    if pos is None:
        return None
    if raim and len(sp) >= 5 and res is not None:
        abs_dev = np.abs(res - np.median(res))
        if np.max(abs_dev) > 15.0:
            keep = np.arange(len(sp)) != int(np.argmax(abs_dev))
            pos2, _ = wls_step(sp[keep], pr[keep], clk[keep], ion[keep], trp[keep], tgd[keep], pos)
            if pos2 is not None:
                pos = pos2
    return pos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obs', type=Path, help='Direct RINEX .obs file path')
    parser.add_argument('--nav', type=Path, help='Direct RINEX .nav file path')
    parser.add_argument('--input-dir', type=Path, help='CSV input directory (from rinex2csv)')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--submission', default='G05_owndata.csv')
    parser.add_argument('--no-raim', action='store_true', help='Disable RAIM outlier rejection')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / args.submission
    rows = []
    seed = np.zeros(3)

    if args.obs and args.nav:
        ephs = parse_rinex_nav(args.nav)
        epochs_data = parse_rinex_obs(args.obs)
        for w, tow, sats in epochs_data:
            pos = solve_rinex_epoch(w, tow, sats, ephs, seed, raim=not args.no_raim)
            if pos is not None:
                seed = pos.copy()
                rows.append((w, round(tow, 1), pos))
    elif args.input_dir:
        arrays, epochs = read_csv_inputs(args.input_dir)
        for i, (w, tow) in enumerate(epochs):
            sp = arrays['satellite_positions'][:, 3*i:3*i+3]
            vecs = [arrays[n][:, i] for n in ['pseudoranges_meas', 'satellite_clock_bias',
                                              'ionospheric_delay', 'tropospheric_delay', 'group_delay']]
            mask = np.isfinite(sp).all(axis=1)
            for v in vecs:
                mask &= np.isfinite(v)
            if mask.sum() < 4:
                continue
            pr, clk, ion, trp, tgd = [v[mask] for v in vecs]
            pos = solve_csv_epoch_with_raim(sp[mask], pr, clk, ion, trp, tgd, seed, raim=not args.no_raim)
            if pos is not None:
                seed = pos.copy()
                rows.append((w, round(tow, 1), pos))
    else:
        parser.error('Provide either (--obs and --nav) or --input-dir')

    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for w, tow, p in rows:
            writer.writerow([w, f"{tow:.1f}", f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}"])
    print(f"Saved {len(rows)} solved epochs to {out_csv}")


if __name__ == '__main__':
    main()
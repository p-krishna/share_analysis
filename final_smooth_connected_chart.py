"""Create a smooth connected crest/trough chart from a stock CSV file."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import find_peaks


def load_price_data(csv_path: str) -> pd.DataFrame:
    """Load price data sorted by date."""
    data = pd.read_csv(csv_path, usecols=['date', 'price_usd'])
    data['date'] = pd.to_datetime(data['date'])
    return data.sort_values('date').reset_index(drop=True)


def build_long_trend(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build long-term trend and band."""
    radius_days = 90
    window = radius_days * 2 + 1
    lower_raw = maximum_filter1d(
        minimum_filter1d(prices, size=window, mode='nearest'),
        size=window,
        mode='nearest',
    )
    upper_raw = minimum_filter1d(
        maximum_filter1d(prices, size=window, mode='nearest'),
        size=window,
        mode='nearest',
    )
    trend_cycle = (lower_raw + upper_raw) / 2.0
    trend_long = (
        pd.Series(trend_cycle)
        .rolling(420, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    band = upper_raw - lower_raw
    return trend_long, band


def detect_turn_points(trend_long: np.ndarray) -> pd.DataFrame:
    """Detect broad crest and trough zones."""
    peak_idx, _ = find_peaks(trend_long, distance=120, prominence=0.04)
    trough_idx, _ = find_peaks(-trend_long, distance=120, prominence=0.04)
    turn_points = pd.DataFrame(
        {
            'index': np.concatenate([peak_idx, trough_idx]),
            'turn_type': ['crest'] * len(peak_idx) + ['trough'] * len(trough_idx),
        }
    )
    return turn_points.sort_values('index').reset_index(drop=True)


def build_segments(
    turn_points: pd.DataFrame,
    dates: pd.Series,
    trend_long: np.ndarray,
    band: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build rising/falling triangle backdrop."""
    segment_rows: List[Dict] = []
    triangle_rows: List[pd.DataFrame] = []

    for segment_id in range(len(turn_points) - 1):
        start_idx = int(turn_points.loc[segment_id, 'index'])
        end_idx = int(turn_points.loc[segment_id + 1, 'index'])
        if end_idx <= start_idx:
            continue

        start_value = float(trend_long[start_idx])
        end_value = float(trend_long[end_idx])
        phase = 'rising' if end_value >= start_value else 'falling'
        segment_rows.append(
            {
                'segment_id': segment_id,
                'start_date': dates.iloc[start_idx],
                'end_date': dates.iloc[end_idx],
                'phase': phase,
            }
        )

        index_range = np.arange(start_idx, end_idx + 1)
        x_fraction = np.linspace(0.0, 1.0, len(index_range))
        center_line = np.linspace(start_value, end_value, len(index_range))
        base_width = np.nanmedian(band[index_range]) * 0.20
        triangle_width = np.sin(np.pi * x_fraction) * max(base_width, 0.08)
        triangle_rows.append(
            pd.DataFrame(
                {
                    'segment_id': segment_id,
                    'date': dates.iloc[index_range].to_numpy(),
                    'tube_low': center_line - triangle_width,
                    'tube_high': center_line + triangle_width,
                    'phase': phase,
                }
            )
        )

    return pd.DataFrame(segment_rows), pd.concat(triangle_rows, ignore_index=True)


def select_zone_points(
    local_prices: np.ndarray,
    turn_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep up to two significant local extrema per zone."""
    if turn_type == 'crest':
        local_idx, props = find_peaks(local_prices, distance=35, prominence=0.12)
        fallback_idx = int(np.argmax(local_prices))
    else:
        local_idx, props = find_peaks(-local_prices, distance=35, prominence=0.12)
        fallback_idx = int(np.argmin(local_prices))

    if len(local_idx) == 0:
        return np.array([fallback_idx]), np.array([np.nan])

    prominences = props['prominences']
    order = np.argsort(-prominences)[:2]
    chosen_idx = local_idx[order]
    chosen_prom = prominences[order]
    ordered = np.argsort(chosen_idx)
    return chosen_idx[ordered], chosen_prom[ordered]


def build_baseline_points(
    turn_points: pd.DataFrame,
    prices: np.ndarray,
    dates: pd.Series,
) -> pd.DataFrame:
    """Build accepted moderate baseline points."""
    rows: List[Dict] = []

    for turn_id, row in turn_points.iterrows():
        center_idx = int(row['index'])
        if turn_id == 0:
            left_bound = max(0, center_idx - 120)
        else:
            left_bound = int((turn_points.loc[turn_id - 1, 'index'] + center_idx) // 2)

        if turn_id == len(turn_points) - 1:
            right_bound = min(len(prices) - 1, center_idx + 120)
        else:
            right_bound = int((center_idx + turn_points.loc[turn_id + 1, 'index']) // 2)

        local_indices = np.arange(left_bound, right_bound + 1)
        local_prices = prices[local_indices]
        chosen_idx, chosen_prom = select_zone_points(local_prices, row['turn_type'])

        for offset, prominence in zip(chosen_idx, chosen_prom):
            actual_index = int(local_indices[int(offset)])
            rows.append(
                {
                    'source': 'baseline',
                    'extreme_kind': 'crest_max' if row['turn_type'] == 'crest' else 'trough_min',
                    'date': dates.iloc[actual_index],
                    'price_usd': float(prices[actual_index]),
                    'actual_index': actual_index,
                    'local_prominence': float(prominence) if not pd.isna(prominence) else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(['date', 'extreme_kind']).reset_index(drop=True)


def build_tail_points(
    baseline_points: pd.DataFrame,
    prices: np.ndarray,
    dates: pd.Series,
) -> pd.DataFrame:
    """Add tail exceptions after the last baseline point only."""
    last_index = int(baseline_points['actual_index'].max())
    tail_indices = np.arange(last_index + 1, len(prices))
    if len(tail_indices) < 40:
        return pd.DataFrame(columns=baseline_points.columns)

    tail_prices = prices[tail_indices]
    trough_idx, trough_props = find_peaks(-tail_prices, distance=35, prominence=0.10)
    crest_idx, crest_props = find_peaks(tail_prices, distance=35, prominence=0.10)
    if len(trough_idx) == 0:
        return pd.DataFrame(columns=baseline_points.columns)

    rows: List[Dict] = []
    first_trough_order = np.argsort(-trough_props['prominences'])[0]
    first_trough = int(trough_idx[first_trough_order])

    def add_point(offset: int, kind: str, prominence: float) -> None:
        actual_index = int(tail_indices[offset])
        rows.append(
            {
                'source': 'tail_exception',
                'extreme_kind': kind,
                'date': dates.iloc[actual_index],
                'price_usd': float(prices[actual_index]),
                'actual_index': actual_index,
                'local_prominence': float(prominence),
            }
        )

    add_point(first_trough, 'trough_min', trough_props['prominences'][first_trough_order])

    later_crest_mask = crest_idx > first_trough
    if later_crest_mask.any():
        later_crest_idx = crest_idx[later_crest_mask]
        later_crest_prom = crest_props['prominences'][later_crest_mask]
        crest_order = np.argsort(-later_crest_prom)[0]
        chosen_crest = int(later_crest_idx[crest_order])
        add_point(chosen_crest, 'crest_max', later_crest_prom[crest_order])

        later_trough_mask = trough_idx > chosen_crest
        if later_trough_mask.any():
            later_trough_idx = trough_idx[later_trough_mask]
            later_trough_prom = trough_props['prominences'][later_trough_mask]
            second_trough_order = np.argsort(-later_trough_prom)[0]
            chosen_trough = int(later_trough_idx[second_trough_order])
            add_point(chosen_trough, 'trough_min', later_trough_prom[second_trough_order])

    return pd.DataFrame(rows).sort_values(['date', 'extreme_kind']).reset_index(drop=True)


def plot_chart(
    price_data: pd.DataFrame,
    segments: pd.DataFrame,
    triangle_tube: pd.DataFrame,
    baseline_points: pd.DataFrame,
    tail_points: pd.DataFrame,
    output_chart_path: str,
) -> None:
    """Plot final smooth connected chart."""
    all_points = pd.concat([baseline_points, tail_points], ignore_index=True)
    all_points = all_points.sort_values(['date', 'extreme_kind', 'source']).reset_index(drop=True)

    x_numeric = mdates.date2num(pd.to_datetime(all_points['date']))
    y_values = all_points['price_usd'].to_numpy(dtype=float)
    unique_mask = np.concatenate(([True], np.diff(x_numeric) > 0))
    x_numeric = x_numeric[unique_mask]
    y_values = y_values[unique_mask]
    interpolator = PchipInterpolator(x_numeric, y_values)
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 1200)
    y_smooth = interpolator(x_smooth)
    smooth_dates = mdates.num2date(x_smooth)

    fig, ax = plt.subplots(figsize=(16, 6), dpi=160)
    ax.plot(
        price_data['date'],
        price_data['price_usd'],
        color='#d6a11f',
        linewidth=0.5,
        alpha=0.5,
        label='price',
    )

    phase_colors = {'rising': '#148f2d', 'falling': '#b02a2a'}
    used_labels = set()
    for _, row in segments.iterrows():
        segment = triangle_tube[triangle_tube['segment_id'] == row['segment_id']]
        label = row['phase'] if row['phase'] not in used_labels else None
        used_labels.add(row['phase'])
        ax.fill_between(
            segment['date'],
            segment['tube_low'],
            segment['tube_high'],
            color=phase_colors[row['phase']],
            alpha=0.26,
            linewidth=0,
            label=label,
        )

    ax.plot(
        smooth_dates,
        y_smooth,
        color='#2f1847',
        linewidth=1.6,
        alpha=0.96,
        label='smoothed connection',
    )

    baseline_crest = baseline_points[baseline_points['extreme_kind'] == 'crest_max']
    baseline_trough = baseline_points[baseline_points['extreme_kind'] == 'trough_min']
    tail_crest = tail_points[tail_points['extreme_kind'] == 'crest_max']
    tail_trough = tail_points[tail_points['extreme_kind'] == 'trough_min']

    ax.scatter(
        baseline_crest['date'], baseline_crest['price_usd'], s=48,
        color='#0b5cab', edgecolor='white', linewidth=0.8,
        zorder=7, label='baseline crest max'
    )
    ax.scatter(
        baseline_trough['date'], baseline_trough['price_usd'], s=48,
        color='#7a1f1f', edgecolor='white', linewidth=0.8,
        zorder=7, label='baseline trough min'
    )

    if not tail_crest.empty:
        ax.scatter(
            tail_crest['date'], tail_crest['price_usd'], s=95, marker='D',
            color='#2a7fff', edgecolor='white', linewidth=1.0,
            zorder=8, label='tail crest add'
        )
    if not tail_trough.empty:
        ax.scatter(
            tail_trough['date'], tail_trough['price_usd'], s=95, marker='D',
            color='#c43b3b', edgecolor='white', linewidth=1.0,
            zorder=8, label='tail trough add'
        )

    ax.set_title('Moderate Crest and Trough Template with Smoothed Connection')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(alpha=0.14)
    ax.legend(loc='upper left', ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(output_chart_path, bbox_inches='tight')
    plt.close(fig)


def run(input_csv_path: str) -> None:
    """Run the full pipeline."""
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    stock_name = Path(input_csv_path).stem.replace('_usd', '')
    price_data = load_price_data(input_csv_path)
    prices = price_data['price_usd'].to_numpy(dtype=float)
    dates = price_data['date']

    trend_long, band = build_long_trend(prices)
    turn_points = detect_turn_points(trend_long)
    segments, triangle_tube = build_segments(turn_points, dates, trend_long, band)
    baseline_points = build_baseline_points(turn_points, prices, dates)
    tail_points = build_tail_points(baseline_points, prices, dates)

    plot_chart(
        price_data=price_data,
        segments=segments,
        triangle_tube=triangle_tube,
        baseline_points=baseline_points,
        tail_points=tail_points,
        output_chart_path=str(output_dir / f'{stock_name}_smooth_connected_chart.png'),
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python final_smooth_connected_chart.py <ticker>')
        print('Example: python final_smooth_connected_chart.py rallis')
        sys.exit(1)
    
    ticker = sys.argv[1]
    csv_path = f'data/{ticker}_usd.csv'
    
    if not Path(csv_path).exists():
        print(f'Error: File {csv_path} not found')
        sys.exit(1)
    
    run(csv_path)

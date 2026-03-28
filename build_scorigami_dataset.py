from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


START_YEAR = 1925
END_YEAR = 2026
RETROSHEET_GAMELOG_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"
GRID_MAX = 30

DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
FRANCHISE_PLOTS_DIR = PLOTS_DIR / "franchises"
VALIDATION_DIR = Path("validation")
RAW_DIR = Path("raw/retrosheet_gamelogs")
RHE_SUMMARY_PATH = VALIDATION_DIR / "rhe_combo_summary.csv"

# Retrosheet team code normalization to franchise label.
TEAM_TO_FRANCHISE = {
    "ANA": "LAA",
    "ARI": "ARI",
    "ATL": "ATL",
    "ATH": "OAK",
    "BAL": "BAL",
    "BOS": "BOS",
    "BRO": "LAD",
    "BSN": "ATL",
    "CAL": "LAA",
    "CHA": "CHW",
    "CHW": "CHW",
    "CHC": "CHC",
    "CHN": "CHC",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "FLO": "MIA",
    "HOU": "HOU",
    "KC1": "OAK",
    "KCA": "KC",
    "KCR": "KC",
    "KC": "KC",
    "LAN": "LAD",
    "LAD": "LAD",
    "LAA": "LAA",
    "MIA": "MIA",
    "MIL": "MIL",
    "MLN": "ATL",
    "MIN": "MIN",
    "MON": "WSN",
    "NY1": "SF",
    "NYA": "NYY",
    "NYY": "NYY",
    "NYN": "NYM",
    "NYM": "NYM",
    "OAK": "OAK",
    "PHA": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SDN": "SD",
    "SD": "SD",
    "SE1": "MIL",
    "SEA": "SEA",
    "SFN": "SF",
    "SFG": "SF",
    "SLA": "BAL",
    "SLN": "STL",
    "STL": "STL",
    "TBA": "TB",
    "TB": "TB",
    "TBR": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WAS": "WSN",
    "WSH": "WSN",
    "WS1": "MIN",
    "WS2": "TEX",
}

STATSAPI_TEAM_TO_RETRO = {
    "AZ": "ARI",
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHN",
    "CWS": "CHA",
    "CHW": "CHA",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KCA",
    "KCR": "KCA",
    "LAA": "ANA",
    "LAD": "LAN",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "ATH": "ATH",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SDN",
    "SEA": "SEA",
    "SF": "SFN",
    "SFG": "SFN",
    "STL": "SLN",
    "TB": "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WAS",
}

FRANCHISE_ORDER = [
    "ARI",
    "ATL",
    "BAL",
    "BOS",
    "CHC",
    "CHW",
    "CIN",
    "CLE",
    "COL",
    "DET",
    "HOU",
    "KC",
    "LAA",
    "LAD",
    "MIA",
    "MIL",
    "MIN",
    "NYM",
    "NYY",
    "OAK",
    "PHI",
    "PIT",
    "SD",
    "SEA",
    "SF",
    "STL",
    "TB",
    "TEX",
    "TOR",
    "WSN",
]


@dataclass
class YearLoadResult:
    year: int
    games: int
    team_games: int
    output_path: Path


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    FRANCHISE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def normalize_franchise(team: str) -> str:
    return TEAM_TO_FRANCHISE.get(team, team)


def current_year() -> int:
    return int(pd.Timestamp.now().year)


def is_current_season(year: int) -> bool:
    return year == current_year()


def map_statsapi_team_code(code: str) -> str:
    return STATSAPI_TEAM_TO_RETRO.get(code, code)


def get_year_raw_zip_path(year: int) -> Path:
    local_zip = RAW_DIR / f"gl{year}.zip"
    if local_zip.exists():
        return local_zip
    url = RETROSHEET_GAMELOG_URL.format(year=year)
    print(f"{year}: downloading Retrosheet gamelog zip")
    urlretrieve(url, local_zip)
    return local_zip


def load_year_team_games_from_statsapi(year: int) -> pd.DataFrame:
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&season={year}&gameType=R&hydrate=linescore,team"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            status = game.get("status", {}).get("abstractGameState")
            if status != "Final":
                continue
            linescore = game.get("linescore", {})
            away_line = linescore.get("teams", {}).get("away", {})
            home_line = linescore.get("teams", {}).get("home", {})
            required = [
                away_line.get("runs"),
                away_line.get("hits"),
                away_line.get("errors"),
                home_line.get("runs"),
                home_line.get("hits"),
                home_line.get("errors"),
            ]
            if any(v is None for v in required):
                continue

            game_date = pd.to_datetime(game.get("gameDate")).tz_localize(None).date()
            away_abbr = map_statsapi_team_code(
                game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
            )
            home_abbr = map_statsapi_team_code(
                game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
            )
            game_number = str(game.get("gameNumber", 0))
            game_id = f"{pd.Timestamp(game_date).strftime('%Y%m%d')}_{away_abbr}_{home_abbr}_G{game_number}"

            rows.append(
                {
                    "year": year,
                    "game_date": pd.Timestamp(game_date),
                    "game_id": game_id,
                    "doubleheader_game_num": game_number,
                    "side": "away",
                    "team": away_abbr,
                    "opponent": home_abbr,
                    "runs": int(away_line["runs"]),
                    "hits": int(away_line["hits"]),
                    "errors": int(away_line["errors"]),
                }
            )
            rows.append(
                {
                    "year": year,
                    "game_date": pd.Timestamp(game_date),
                    "game_id": game_id,
                    "doubleheader_game_num": game_number,
                    "side": "home",
                    "team": home_abbr,
                    "opponent": away_abbr,
                    "runs": int(home_line["runs"]),
                    "hits": int(home_line["hits"]),
                    "errors": int(home_line["errors"]),
                }
            )
    if not rows:
        raise ValueError(f"{year}: Stats API returned no completed regular-season games.")
    team_games = pd.DataFrame(rows)
    team_games["franchise"] = team_games["team"].map(normalize_franchise)
    team_games["opponent_franchise"] = team_games["opponent"].map(normalize_franchise)
    team_games = team_games.sort_values(["game_date", "game_id", "side"]).drop_duplicates(
        subset=["game_id", "side"], keep="last"
    )
    return team_games.reset_index(drop=True)


def load_year_team_games(year: int) -> pd.DataFrame:
    if is_current_season(year):
        current_zip = RAW_DIR / f"gl{year}.zip"
        if current_zip.exists():
            local_zip = current_zip
        else:
            print(f"{year}: ongoing season detected; using MLB Stats API source.")
            return load_year_team_games_from_statsapi(year)
    else:
        local_zip = get_year_raw_zip_path(year)
    # Retrosheet field mapping from https://www.retrosheet.org/gamelogs/glfields.txt
    # (0-based indices)
    usecols = [
        0,   # date
        1,   # doubleheader game number
        3,   # away team
        6,   # home team
        9,   # away runs
        10,  # home runs
        22,  # away hits (field 23)
        45,  # away errors (field 46)
        50,  # home hits (field 51)
        73,  # home errors (field 74)
        21,  # away AB
        23,  # away doubles
        24,  # away triples
        25,  # away HR
        49,  # home AB
        51,  # home doubles
        52,  # home triples
        53,  # home HR
    ]
    raw0 = pd.read_csv(local_zip, compression="zip", header=None, usecols=usecols)
    raw = pd.DataFrame(
        {
            "game_date": raw0[0],
            "doubleheader_game_num": raw0[1],
            "away_team": raw0[3],
            "home_team": raw0[6],
            "away_runs": raw0[9],
            "home_runs": raw0[10],
            "away_hits": raw0[22],
            "away_errors": raw0[45],
            "home_hits": raw0[50],
            "home_errors": raw0[73],
            "away_ab": raw0[21],
            "away_2b": raw0[23],
            "away_3b": raw0[24],
            "away_hr": raw0[25],
            "home_ab": raw0[49],
            "home_2b": raw0[51],
            "home_3b": raw0[52],
            "home_hr": raw0[53],
        }
    )

    numeric_cols = [
        "away_runs",
        "home_runs",
        "away_hits",
        "away_errors",
        "home_hits",
        "home_errors",
        "away_ab",
        "away_2b",
        "away_3b",
        "away_hr",
        "home_ab",
        "home_2b",
        "home_3b",
        "home_hr",
    ]
    for c in numeric_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Defensive checks against wrong field mapping.
    raw["away_singles"] = raw["away_hits"] - raw["away_2b"] - raw["away_3b"] - raw["away_hr"]
    raw["home_singles"] = raw["home_hits"] - raw["home_2b"] - raw["home_3b"] - raw["home_hr"]
    if (raw["away_singles"] < 0).any() or (raw["home_singles"] < 0).any():
        raise ValueError(f"{year}: Negative singles found; likely bad hit-stat field mapping.")
    if (raw["away_ab"] < raw["away_hits"]).any() or (raw["home_ab"] < raw["home_hits"]).any():
        raise ValueError(f"{year}: AB < H detected; likely bad offensive-stat field mapping.")

    raw = raw.drop(columns=["away_ab", "away_2b", "away_3b", "away_hr", "home_ab", "home_2b", "home_3b", "home_hr", "away_singles", "home_singles"])

    raw = raw.dropna(
        subset=["away_runs", "home_runs", "away_hits", "away_errors", "home_hits", "home_errors"]
    )
    raw["game_date"] = pd.to_datetime(raw["game_date"].astype(int).astype(str), format="%Y%m%d")
    raw["game_id"] = (
        raw["game_date"].dt.strftime("%Y%m%d")
        + "_"
        + raw["away_team"]
        + "_"
        + raw["home_team"]
        + "_G"
        + raw["doubleheader_game_num"].astype(str).str.strip()
    )

    away = pd.DataFrame(
        {
            "year": year,
            "game_date": raw["game_date"],
            "game_id": raw["game_id"],
            "doubleheader_game_num": raw["doubleheader_game_num"].astype(str).str.strip(),
            "side": "away",
            "team": raw["away_team"],
            "opponent": raw["home_team"],
            "runs": raw["away_runs"].astype(int),
            "hits": raw["away_hits"].astype(int),
            "errors": raw["away_errors"].astype(int),
        }
    )
    home = pd.DataFrame(
        {
            "year": year,
            "game_date": raw["game_date"],
            "game_id": raw["game_id"],
            "doubleheader_game_num": raw["doubleheader_game_num"].astype(str).str.strip(),
            "side": "home",
            "team": raw["home_team"],
            "opponent": raw["away_team"],
            "runs": raw["home_runs"].astype(int),
            "hits": raw["home_hits"].astype(int),
            "errors": raw["home_errors"].astype(int),
        }
    )

    team_games = pd.concat([away, home], ignore_index=True)
    team_games["franchise"] = team_games["team"].map(normalize_franchise)
    team_games["opponent_franchise"] = team_games["opponent"].map(normalize_franchise)
    return team_games.sort_values(["game_date", "game_id", "side"]).reset_index(drop=True)


def write_year_parquet(team_games: pd.DataFrame, year: int) -> Path:
    output_path = DATA_DIR / f"box_scores_{year}.parquet"
    team_games.to_parquet(output_path, index=False)
    return output_path


def build_all_years(years: Iterable[int]) -> tuple[pd.DataFrame, list[YearLoadResult]]:
    yearly_frames: list[pd.DataFrame] = []
    results: list[YearLoadResult] = []
    for year in years:
        out_path = DATA_DIR / f"box_scores_{year}.parquet"
        if out_path.exists() and not is_current_season(year):
            team_games = pd.read_parquet(out_path)
            print(f"{year}: loaded cached parquet ({len(team_games)} team-games)")
            team_games["franchise"] = team_games["team"].map(normalize_franchise)
            team_games["opponent_franchise"] = team_games["opponent"].map(normalize_franchise)
            team_games.to_parquet(out_path, index=False)
        else:
            if out_path.exists() and is_current_season(year):
                print(f"{year}: refreshing current-season parquet from live source")
            team_games = load_year_team_games(year)
            out_path = write_year_parquet(team_games, year)
        yearly_frames.append(team_games)
        results.append(
            YearLoadResult(
                year=year,
                games=team_games["game_id"].nunique(),
                team_games=len(team_games),
                output_path=out_path,
            )
        )
        print(f"{year}: games={results[-1].games}, team_games={results[-1].team_games}")
    all_team_games = pd.concat(yearly_frames, ignore_index=True)
    return all_team_games, results


def run_validations(df: pd.DataFrame) -> None:
    validation_rows = []

    year_game_counts = (
        df.groupby("year", as_index=False)["game_id"]
        .nunique()
        .rename(columns={"game_id": "games"})
        .sort_values("year")
    )
    year_game_counts["team_games_expected"] = year_game_counts["games"] * 2
    year_team_games = (
        df.groupby("year", as_index=False).size().rename(columns={"size": "team_games"})
    )
    year_summary = year_game_counts.merge(year_team_games, on="year", how="left")
    year_summary["team_games_match"] = (
        year_summary["team_games"] == year_summary["team_games_expected"]
    )

    bad_game_sizes = df.groupby("game_id").size().loc[lambda s: s != 2]
    side_counts = df.groupby("game_id")["side"].nunique()
    bad_sides = side_counts.loc[lambda s: s != 2]
    side_sets = (
        df.groupby("game_id")["side"]
        .apply(lambda s: "|".join(sorted(set(s.tolist()))))
        .rename("side_set")
    )
    bad_side_labels = side_sets.loc[side_sets != "away|home"]
    missing_values = df[["runs", "hits", "errors", "team", "franchise"]].isna().sum()
    negative_counts = (df[["runs", "hits", "errors"]] < 0).sum()

    validation_rows.append(
        {
            "check": "all_games_have_exactly_two_rows",
            "result": bad_game_sizes.empty,
            "details": f"violations={len(bad_game_sizes)}",
        }
    )
    validation_rows.append(
        {
            "check": "year_team_game_counts_match_2x_games",
            "result": bool(year_summary["team_games_match"].all()),
            "details": f"bad_years={int((~year_summary['team_games_match']).sum())}",
        }
    )
    validation_rows.append(
        {
            "check": "all_games_have_home_and_away_rows",
            "result": bad_sides.empty and bad_side_labels.empty,
            "details": f"bad_sides={len(bad_sides)}, bad_side_labels={len(bad_side_labels)}",
        }
    )
    validation_rows.append(
        {
            "check": "no_missing_rhe_team_fields",
            "result": bool((missing_values == 0).all()),
            "details": ", ".join(f"{k}={int(v)}" for k, v in missing_values.items()),
        }
    )
    validation_rows.append(
        {
            "check": "no_negative_rhe_values",
            "result": bool((negative_counts == 0).all()),
            "details": ", ".join(f"{k}={int(v)}" for k, v in negative_counts.items()),
        }
    )

    team_year_counts = (
        df.groupby(["year", "franchise"], as_index=False)
        .size()
        .rename(columns={"size": "team_games"})
    )
    # Support mixed schedule lengths within a single year (e.g., 1961 leagues and strike seasons).
    year_expected_counts = (
        team_year_counts.groupby(["year", "team_games"])
        .size()
        .reset_index(name="n_teams")
        .sort_values(["year", "n_teams", "team_games"], ascending=[True, False, True])
    )
    year_expected_lookup = (
        year_expected_counts.groupby("year")["team_games"].apply(list).to_dict()
    )
    tolerance = 4
    nearest_expected = []
    within_range_vals = []
    expected_min_vals = []
    expected_max_vals = []
    for _, row in team_year_counts.iterrows():
        expected = year_expected_lookup.get(int(row["year"]), [int(row["team_games"])])
        nearest = min(expected, key=lambda x: abs(int(row["team_games"]) - int(x)))
        nearest_expected.append(int(nearest))
        expected_min_vals.append(int(nearest) - tolerance)
        expected_max_vals.append(int(nearest) + tolerance)
        within_range_vals.append(abs(int(row["team_games"]) - int(nearest)) <= tolerance)

    team_year_counts["nearest_expected_team_games"] = nearest_expected
    team_year_counts["expected_min"] = expected_min_vals
    team_year_counts["expected_max"] = expected_max_vals
    team_year_counts["within_range"] = within_range_vals
    outliers = team_year_counts.loc[~team_year_counts["within_range"]]
    validation_rows.append(
        {
            "check": "team_season_game_counts_in_reasonable_range",
            "result": outliers.empty,
            "details": f"outliers={len(outliers)}",
        }
    )

    error_dist = df.groupby("errors").size().rename("team_games").reset_index().sort_values("errors")
    panel_sanity = (
        df.assign(runs_le_hits=df["runs"] <= df["hits"])
        .groupby("errors")
        .agg(
            team_games=("runs", "size"),
            runs_le_hits=("runs_le_hits", "sum"),
        )
        .reset_index()
    )
    panel_sanity["runs_gt_hits"] = panel_sanity["team_games"] - panel_sanity["runs_le_hits"]
    panel_sanity["runs_le_hits_pct"] = panel_sanity["runs_le_hits"] / panel_sanity["team_games"]
    panel_sanity["runs_gt_hits_pct"] = panel_sanity["runs_gt_hits"] / panel_sanity["team_games"]

    cnt1 = int(error_dist.loc[error_dist["errors"] == 1, "team_games"].sum())
    cnt2 = int(error_dist.loc[error_dist["errors"] == 2, "team_games"].sum())
    validation_rows.append(
        {
            "check": "error1_count_greater_than_or_equal_error2_count",
            "result": cnt1 >= cnt2,
            "details": f"errors=1 -> {cnt1}, errors=2 -> {cnt2}",
        }
    )

    pd.DataFrame(validation_rows).to_csv(VALIDATION_DIR / "validation_checks.csv", index=False)
    year_summary.to_csv(VALIDATION_DIR / "games_by_year.csv", index=False)
    team_year_counts.to_csv(VALIDATION_DIR / "team_games_by_year.csv", index=False)
    outliers.to_csv(VALIDATION_DIR / "team_games_outliers.csv", index=False)
    error_dist.to_csv(VALIDATION_DIR / "error_distribution.csv", index=False)
    panel_sanity.to_csv(VALIDATION_DIR / "panel_sanity.csv", index=False)


def _make_count_matrix(panel_df: pd.DataFrame, max_runs: int, max_hits: int) -> np.ndarray:
    matrix = np.zeros((max_runs + 1, max_hits + 1), dtype=float)
    if panel_df.empty:
        return matrix
    counts = panel_df.groupby(["runs", "hits"]).size().reset_index(name="count")
    valid = counts[(counts["runs"] <= max_runs) & (counts["hits"] <= max_hits)]
    matrix[valid["runs"].to_numpy(), valid["hits"].to_numpy()] = valid["count"].to_numpy()
    return matrix


def _power_of_ten_ticks(vmax: float) -> list[int]:
    ticks = [1]
    p = 1
    while p * 10 <= vmax:
        p *= 10
        ticks.append(p)
    return ticks


def _axis_ticks(max_val: int) -> np.ndarray:
    return np.arange(0, max_val + 1, 1)


def _draw_heatmap_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    norm: LogNorm,
    cmap: plt.Colormap,
    max_runs: int,
    max_hits: int,
) -> plt.AxesImage:
    masked = np.ma.masked_where(matrix < 1, matrix)
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Hits", fontsize=15)
    ax.set_ylabel("Runs", fontsize=15)
    ax.set_xlim(-0.5, max_hits + 0.5)
    ax.set_ylim(-0.5, max_runs + 0.5)

    major_ticks_x = _axis_ticks(max_hits)
    major_ticks_y = _axis_ticks(max_runs)
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticklabels([str(t) if t % 5 == 0 else "" for t in major_ticks_x], fontsize=10)
    ax.set_yticklabels([str(t) if t % 5 == 0 else "" for t in major_ticks_y], fontsize=10)

    # Major ticks: full size for multiples of 5, half size for others
    major_tick_len  = 6
    minor_tick_len  = 3
    for tick, loc in zip(ax.xaxis.get_major_ticks(), major_ticks_x):
        size = major_tick_len if loc % 5 == 0 else minor_tick_len
        tick.tick1line.set_markersize(size)
        tick.tick2line.set_markersize(size)
    for tick, loc in zip(ax.yaxis.get_major_ticks(), major_ticks_y):
        size = major_tick_len if loc % 5 == 0 else minor_tick_len
        tick.tick1line.set_markersize(size)
        tick.tick2line.set_markersize(size)

    # Faint cell grid lines (true minor ticks, invisible)
    ax.set_xticks(np.arange(-0.5, max_hits + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, max_runs + 1, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.1, alpha=0.6)
    ax.grid(which="major", visible=False)
    ax.set_aspect("equal")

    text_threshold = np.sqrt(norm.vmax)
    for r in range(max_runs + 1):
        for h in range(max_hits + 1):
            v = int(matrix[r, h])
            if v <= 0:
                continue
            text_color = "black" if v >= text_threshold else "white"
            ax.text(
                h, r, str(v),
                ha="center", va="center",
                fontsize=3, color=text_color, alpha=0.9,
            )
    return im

def _plot_error_panels(
    df: pd.DataFrame,
    title_prefix: str,
    output_path: Path,
    max_runs: int,
    max_hits: int,
    error_values: list[int],
    n_rows: int,
    n_cols: int,
) -> None:
    n_panels = len(error_values)
    needed_rows = int(np.ceil(n_panels / n_cols))
    n_rows = max(n_rows, needed_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.1*15, 1.1*5 * n_rows), dpi=350)
    axes = np.atleast_1d(axes).ravel()
    matrices: list[np.ndarray] = []
    for err in error_values:
        subset = df.loc[df["errors"] == err]
        matrices.append(_make_count_matrix(subset, max_runs=max_runs, max_hits=max_hits))
    global_vmax = max(1.0, float(max(np.max(m) for m in matrices)))
    norm = LogNorm(vmin=1, vmax=global_vmax)
    cmap = plt.cm.copper.copy()
    cmap.set_bad("white")
    cmap.set_under("white")
    for idx, err in enumerate(error_values):
        
        subset_n = int((df["errors"] == err).sum())
        im = _draw_heatmap_panel(
            axes[idx],
            matrices[idx],
            "",  # title cleared — label will be drawn inside the panel
            norm=norm,
            cmap=cmap,
            max_runs=max_runs,
            max_hits=max_hits,
        )
        # Colorbar along the top edge
        divider = make_axes_locatable(axes[idx])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Count", fontsize=11)
        cax.xaxis.set_label_position("top")
        cax.xaxis.tick_top()
        ticks = _power_of_ten_ticks(global_vmax)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])

        # "Errors = " annotation inside the panel, top-left
        axes[idx].text(
            0.03, 0.97,
            #f"Errors = {err} | n={subset_n:,}",
            f"Errors = {err}",
            transform=axes[idx].transAxes,
            fontsize=20,
            verticalalignment="top",
            horizontalalignment="left",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.4),
        )

    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")
    
    fig.suptitle(title_prefix, fontsize=25, y=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.subplots_adjust(hspace=-0.15, wspace=0.2)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

def plot_master_scorigami(df: pd.DataFrame) -> None:
    max_runs = GRID_MAX
    max_hits = GRID_MAX
    error_values = list(range(0, int(df["errors"].max()) + 1))
    unique_triples = df[["runs", "hits", "errors"]].drop_duplicates().shape[0]
    title = (
        f"MLB Team-Game Scorigami (Regular Season {START_YEAR}-{END_YEAR})\n"
        #f"Total team-games={len(df):,}, unique R/H/E combos={unique_triples:,}"
    )
    _plot_error_panels(
        df=df,
        title_prefix=title,
        output_path=PLOTS_DIR / f"master_scorigami_{START_YEAR}_{END_YEAR}.png",
        max_runs=max_runs,
        max_hits=max_hits,
        error_values=error_values,
        n_rows=2,
        n_cols=4,
    )


def plot_franchise_error_panels(df: pd.DataFrame) -> None:
    max_runs = GRID_MAX
    max_hits = GRID_MAX
    error_values = list(range(0, int(df["errors"].max()) + 1))
    extra = sorted(set(df["franchise"].unique()) - set(FRANCHISE_ORDER))
    franchise_order = FRANCHISE_ORDER + extra
    for franchise in franchise_order:
        subset = df.loc[df["franchise"] == franchise]
        if subset.empty:
            continue
        unique_rhe = subset[["runs", "hits", "errors"]].drop_duplicates().shape[0]
        title = (
            f"{franchise} Team-Game Scorigami by Errors ({START_YEAR}-{END_YEAR})\n"
            f"Total team-games={len(subset):,}, unique R/H/E combos={unique_rhe:,}"
        )
        _plot_error_panels(
            df=subset,
            title_prefix=title,
            output_path=FRANCHISE_PLOTS_DIR / f"{franchise}.png",
            max_runs=max_runs,
            max_hits=max_hits,
            error_values=error_values,
            n_rows=2,
            n_cols=4,
        )


def plot_franchise_combo_tracking(df: pd.DataFrame) -> None:
    yearly = (
        df.groupby(["franchise", "year", "runs", "hits", "errors"])
        .size()
        .reset_index(name="count")
        .sort_values(["franchise", "year"])
    )
    yearly_unique = (
        yearly.groupby(["franchise", "year"])
        .size()
        .rename("unique_rhe_combos")
        .reset_index()
    )

    full_years = pd.DataFrame({"year": range(START_YEAR, END_YEAR + 1)})
    lines = []
    for franchise in sorted(df["franchise"].unique()):
        base = full_years.merge(
            yearly_unique[yearly_unique["franchise"] == franchise][["year", "unique_rhe_combos"]],
            on="year",
            how="left",
        ).fillna(0)
        base["franchise"] = franchise
        base["cumulative_unique_rhe_combos"] = base["unique_rhe_combos"].cumsum().astype(int)
        lines.append(base)

    tracking = pd.concat(lines, ignore_index=True)
    tracking.to_csv(VALIDATION_DIR / "franchise_combo_tracking.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 8))
    for franchise in FRANCHISE_ORDER:
        sub = tracking.loc[tracking["franchise"] == franchise]
        if sub.empty:
            continue
        ax.plot(
            sub["year"],
            sub["cumulative_unique_rhe_combos"],
            linewidth=1.5,
            alpha=0.85,
            label=franchise,
        )
    ax.set_title(f"Cumulative Unique R/H/E Combos by Franchise ({START_YEAR}-{END_YEAR})")
    ax.set_xlabel("Season")
    ax.set_ylabel("Cumulative unique R/H/E combos")
    ax.grid(alpha=0.2)
    ax.legend(ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"franchise_combo_tracking_{START_YEAR}_{END_YEAR}.png", dpi=180)
    plt.close(fig)


def write_rhe_combo_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_df = df.copy()
    summary_df["game_date"] = pd.to_datetime(summary_df["game_date"])
    key_cols = ["hits", "runs", "errors"]

    first_dates = (
        summary_df.groupby(key_cols, as_index=False)
        .agg(
            game_date=("game_date", "min"),
            total_team_games=("game_id", "size"),
        )
    )
    first_rows = (
        summary_df.merge(first_dates[key_cols + ["game_date"]], on=key_cols + ["game_date"], how="inner")
        .sort_values(key_cols + ["game_date", "game_id", "side"])
        .drop_duplicates(subset=key_cols, keep="first")
        [key_cols + ["game_id", "team", "franchise"]]
        .rename(
            columns={
                "game_id": "first_appearance_game_id",
                "team": "first_appearance_team",
                "franchise": "first_appearance_franchise",
            }
        )
    )

    last_dates = (
        summary_df.groupby(key_cols, as_index=False)
        .agg(game_date=("game_date", "max"))
    )
    last_rows = (
        summary_df.merge(last_dates[key_cols + ["game_date"]], on=key_cols + ["game_date"], how="inner")
        .sort_values(
            key_cols + ["game_date", "game_id", "side"],
            ascending=[True, True, True, False, False, False],
        )
        .drop_duplicates(subset=key_cols, keep="first")
        [key_cols + ["game_id", "team", "franchise"]]
        .rename(
            columns={
                "game_id": "most_recent_game_id",
                "team": "most_recent_team",
                "franchise": "most_recent_franchise",
            }
        )
    )

    summary = (
        first_dates.rename(columns={"game_date": "first_appearance_date"})
        .merge(
            first_rows[
                key_cols + ["first_appearance_game_id", "first_appearance_team", "first_appearance_franchise"]
            ],
            on=key_cols,
            how="left",
        )
        .merge(last_dates.rename(columns={"game_date": "most_recent_date"}), on=key_cols, how="left")
        .merge(
            last_rows[
                key_cols + ["most_recent_game_id", "most_recent_team", "most_recent_franchise"]
            ],
            on=key_cols,
            how="left",
        )
        .sort_values(["most_recent_date", "hits", "runs", "errors"], ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    out = summary.copy()
    out["first_appearance_date"] = out["first_appearance_date"].dt.strftime("%Y-%m-%d")
    out["most_recent_date"] = out["most_recent_date"].dt.strftime("%Y-%m-%d")
    out.to_csv(RHE_SUMMARY_PATH, index=False)
    return summary


def print_recent_box_scorigamis(summary: pd.DataFrame, n: int = 10) -> None:
    recent = summary.sort_values("first_appearance_date", ascending=False).head(n).copy()
    recent["first_appearance_date"] = recent["first_appearance_date"].dt.strftime("%Y-%m-%d")
    cols = [
        "hits",
        "runs",
        "errors",
        "first_appearance_date",
        "first_appearance_team",
        "total_team_games",
    ]
    print("\n10 most recent first-time box-scorigamis:")
    print(recent[cols].to_string(index=False))


def update_readme(summary: pd.DataFrame) -> None:
    readme_path = Path("README.md")
    readme_content = readme_path.read_text()
    
    master_plot_path = PLOTS_DIR / f"master_scorigami_{START_YEAR}_{END_YEAR}.png"
    
    recent = summary.sort_values("first_appearance_date", ascending=False).head(10).copy()
    recent["first_appearance_date"] = recent["first_appearance_date"].dt.strftime("%Y-%m-%d")
    
    table_lines = [
        "| Hits | Runs | Errors | First Appearance | Team | Total Games |",
        "|------|------|--------|------------------|------|-------------|",
    ]
    for _, row in recent.iterrows():
        table_lines.append(
            f"| {row['hits']} | {row['runs']} | {row['errors']} | {row['first_appearance_date']} | {row['first_appearance_team']} | {row['total_team_games']} |"
        )
    table_str = "\n".join(table_lines)
    
    new_section = f"""## Most Recent Scorigamis

![Master Scorigami]({master_plot_path})

{table_str}

"""
    
    if "## Most Recent Scorigamis" in readme_content:
        readme_content = readme_content.split("## Most Recent Scorigamis")[0]
    
    if "## Run" in readme_content:
        run_idx = readme_content.index("## Run")
        readme_content = readme_content[:run_idx].rstrip() + "\n\n" + new_section + readme_content[run_idx:]
    else:
        readme_content = readme_content.rstrip() + "\n\n" + new_section
    
    readme_path.write_text(readme_content)


def main() -> None:
    ensure_output_dirs()
    years = range(START_YEAR, END_YEAR + 1)
    all_team_games, _ = build_all_years(years)

    master_path = DATA_DIR / f"box_scores_{START_YEAR}_{END_YEAR}_master.parquet"
    all_team_games.to_parquet(master_path, index=False)
    run_validations(all_team_games)
    summary = write_rhe_combo_summary(all_team_games)
    print_recent_box_scorigamis(summary, n=10)
    plot_master_scorigami(all_team_games)
    print('plotted master')
    plot_franchise_error_panels(all_team_games)
    plot_franchise_combo_tracking(all_team_games)

    update_readme(summary)

    print(f"Master parquet: {master_path}")
    print(f"RHE combo summary CSV: {RHE_SUMMARY_PATH}")
    print(f"Plots written to: {PLOTS_DIR}")
    print(f"Validation written to: {VALIDATION_DIR}")


if __name__ == "__main__":
    main()

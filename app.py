from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

START_YEAR = 1925
END_YEAR = 2026
GRID_MAX = 31

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

DATA_PATH = Path(__file__).parent / "data" / "scorigami_for_streamlit.parquet"

VIRIDIS_LOG_SCALE = [
    [0.1, "#440154"],
    [0.2, "#482878"],
    [0.3, "#3e4a89"],
    [0.4, "#31688e"],
    [0.5, "#26828e"],
    [0.6, "#1f9e89"],
    [0.7, "#35b779"],
    [0.8, "#6ece58"],
    [0.9, "#b5de2b"],
    [1, "#fde725"],
]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def get_last_updated() -> str:
    df = pd.read_parquet(DATA_PATH)
    recent_date = df["overall_recent_date"].max()
    return recent_date


def create_heatmap_matrix(df: pd.DataFrame, errors: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((GRID_MAX + 1, GRID_MAX + 1), np.nan, dtype=float)
    log_matrix = np.full((GRID_MAX + 1, GRID_MAX + 1), np.nan, dtype=float)
    
    subset = df[df["errors"] == errors]
    
    for _, row in subset.iterrows():
        r = int(row["runs"])
        h = int(row["hits"])
        if r <= GRID_MAX and h <= GRID_MAX:
            count = row["count"]
            matrix[r, h] = count
            log_matrix[r, h] = np.log10(count) if count > 0 else np.nan
    
    return matrix, log_matrix


def create_heatmaps(df: pd.DataFrame, selected_team: str) -> go.Figure:
    franchise = None if selected_team == "All Teams" else selected_team
    
    if franchise is None:
        subset = df[df["franchise"] == "All"]
    else:
        subset = df[df["franchise"] == franchise]
    
    n_rows = 2
    n_cols = 4
    error_values = list(range(8))
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[""] * 8,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )
    
    matrices = []
    log_matrices = []
    for err in error_values:
        matrix, log_matrix = create_heatmap_matrix(subset, err)
        matrices.append(matrix)
        log_matrices.append(log_matrix)
    
    global_log_vmax = float(max(np.nanmax(m) for m in log_matrices))
    
    for idx, err in enumerate(error_values):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        matrix, log_matrix = create_heatmap_matrix(subset, err)
        
        fig.add_trace(
            go.Heatmap(
                z=log_matrix,
                x=list(range(GRID_MAX + 1)),
                y=list(range(GRID_MAX + 1)),
                colorscale=VIRIDIS_LOG_SCALE,
                zmin=0,
                zmax=global_log_vmax,
                showscale=True,
                customdata=matrix,
                hovertemplate=(
                    "Runs: %{y}<br>"
                    "Hits: %{x}<br>"
                    f"Errors: {err}<br>"
                    "Count: %{{customdata:.0f}}<extra></extra>"
                ),
                xgap=1,
                ygap=1,
                colorbar=dict(
                    len=1.05,
                    x=1.02,
                    y=0.5,
                    xanchor="left",
                    yanchor="middle",
                    orientation="v",
                    thickness=35,
                    lenmode="fraction",
                    title=dict(text="log₁₀(Count)", font=dict(size=12, color="black")),
                    tickfont=dict(size=10, color="black"),
                ),
            ),
            row=row,
            col=col,
        )
        
        fig.add_annotation(
            x=1,
            y=GRID_MAX - 1,
            xref=f"x{idx + 1}",
            yref=f"y{idx + 1}",
            text=f"Errors = {err}",
            showarrow=False,
            font=dict(size=18, color="white"),
            xanchor="left",
            yanchor="top",
            bgcolor="black",
            opacity=0.7,
            xshift=-2,
            yshift=-5,
        )
        
        fig.update_xaxes(
            range=[-0.5, GRID_MAX + 0.5],
            dtick=5,
            tickmode="array",
            tickvals=list(range(0, GRID_MAX + 1, 5)),
            ticktext=[str(t) for t in range(0, GRID_MAX + 1, 5)],
            tickfont=dict(size=10, color="black"),
            tickcolor="black",
            title_text="Hits",
            title_font=dict(size=13, color="black"),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            row=row,
            col=col,
        )
        
        fig.update_yaxes(
            range=[-0.5, GRID_MAX + 0.5],
            dtick=5,
            tickmode="array",
            tickvals=list(range(0, GRID_MAX + 1, 5)),
            ticktext=[str(t) for t in range(0, GRID_MAX + 1, 5)],
            tickfont=dict(size=10, color="black"),
            tickcolor="black",
            title_text="Runs",
            title_font=dict(size=13, color="black"),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            row=row,
            col=col,
            scaleanchor=f"x{idx + 1}",
            scaleratio=1,
        )
    
    title_prefix = "MLB" if franchise is None else franchise
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} Team-Game Scorigami ({START_YEAR}-{END_YEAR})",
            font=dict(size=24),
            x=0.5,
            y=0.98,
        ),
        width=1650,
        height=950,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        autosize=True,
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="MLB Box-Scorigami",
        page_icon="⚾",
        layout="wide",
    )
    
    last_updated = get_last_updated()
    
    st.title("MLB Box-Scorigami")
    st.markdown(f"**{START_YEAR}-{END_YEAR}** Regular Season Team-Game R/H/E Combinations")
    st.markdown(f"*Last updated: {last_updated}*")
    
    df = load_data()
    
    team_options = ["All Teams"] + FRANCHISE_ORDER
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        selected_team = st.selectbox("Select Team", team_options, index=0)
    
    fig = create_heatmaps(df, selected_team)
    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    st.caption("*There are a handful of games with more than 7 errors, but I left them off as an aesthetic choice.*")
    
    st.markdown("---")
    st.subheader("Recent Box-Scorigamis")
    
    if selected_team == "All Teams":
        history_df = df[df["franchise"] == "All"].copy()
        recent = history_df.dropna(subset=["overall_first_date"]).copy()
        recent = recent.sort_values("overall_first_date", ascending=False)
        
        display_df = recent[["runs", "hits", "errors", "overall_first_date", "overall_first_team", "overall_first_opponent"]].drop_duplicates(subset=["runs", "hits", "errors"]).reset_index(drop=True)
        display_df.columns = ["Runs", "Hits", "Errors", "First Appearance Date", "Team", "Opponent"]
    else:
        history_df = df[df["franchise"] == selected_team].copy()
        recent = history_df.dropna(subset=["team_first_appearance_date"]).copy()
        recent = recent.sort_values("team_first_appearance_date", ascending=False)
        
        display_df = recent[["runs", "hits", "errors", "team_first_appearance_date", "team_first_appearance_team", "team_first_appearance_opponent"]].drop_duplicates(subset=["runs", "hits", "errors"]).reset_index(drop=True)
        display_df.columns = ["Runs", "Hits", "Errors", "First Appearance Date", "Team", "Opponent"]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown("---")
    st.caption(f"Data updated daily. Hover over cells to see details.")


if __name__ == "__main__":
    main()

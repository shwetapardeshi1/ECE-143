import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


DATA_PATH = "planecrashinfo_clean.csv"
PLOTS_DIR = "plots"


def ensure_output_dir(path=PLOTS_DIR):
    os.makedirs(path, exist_ok=True)
    return path


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path} in current directory.")
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date_parsed" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
    elif "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date_parsed"] = pd.NaT

    df["year"] = df["date_parsed"].dt.year
    df["decade"] = (df["year"] // 10) * 10

    for col in [
        "fatalities_total",
        "fatalities_passengers",
        "fatalities_crew",
        "ground_fatalities",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    df["is_fatal"] = df["fatalities_total"] > 0

    if "aboard_total" in df.columns:
        df["aboard_total"] = pd.to_numeric(df["aboard_total"], errors="coerce")
    else:
        if "aboard" in df.columns:
            df["aboard_total"] = (
                df["aboard"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )
        else:
            df["aboard_total"] = pd.NA


    df["fatality_ratio"] = df["fatalities_total"] / df["aboard_total"]

    if "time_hhmm" in df.columns:
        df["hour"] = (
            df["time_hhmm"]
            .astype(str)
            .str.extract(r"^(\d{1,2})", expand=False)
            .astype(float)
        )
    else:
        df["hour"] = pd.NA

    return df


def plot_yearly_trends(df, outdir):

    yearly = (
        df.groupby("year", dropna=True)
        .agg(crashes=("year", "count"), fatalities=("fatalities_total", "sum"))
        .reset_index()
    )


    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(yearly["year"], yearly["crashes"], label="Crashes")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of crashes")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(yearly["year"], yearly["fatalities"], color="tab:red", label="Fatalities")
    ax2.set_ylabel("Total fatalities")
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Crashes and fatalities per year")
    plt.tight_layout()
    fname = os.path.join(outdir, "yearly_crashes_fatalities.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_countries(df, outdir, top_n=20):

    counts = df["location_country"].dropna().value_counts().head(top_n).sort_values()

    plt.figure(figsize=(8, 6))
    counts.plot(kind="barh")
    plt.xlabel("Number of accidents")
    plt.ylabel("Country")
    plt.title(f"Top {top_n} countries by number of accidents")
    plt.tight_layout()
    fname = os.path.join(outdir, "top_countries_accidents.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_operators(df, outdir, top_n=15):
    counts = df["operator"].dropna().value_counts().head(top_n).sort_values()

    plt.figure(figsize=(8, 6))
    counts.plot(kind="barh")
    plt.xlabel("Number of accidents")
    plt.ylabel("Operator")
    plt.title(f"Top {top_n} operators by number of accidents")
    plt.tight_layout()
    fname = os.path.join(outdir, "top_operators_accidents.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

def plot_aircraft_severity(df, outdir, top_n=15):

    subset = df.dropna(subset=["aircraft_type", "fatality_ratio"])
    if subset.empty:
        print("Skipping aircraft severity plot (no valid rows).")
        return

    stats = (
        subset.groupby("aircraft_type")["fatality_ratio"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
    )

    if stats.empty:
        print("Skipping aircraft severity plot (no grouped stats).")
        return

    plt.figure(figsize=(8, 6))
    stats.plot(kind="barh")
    plt.xlabel("Median fatality ratio")
    plt.ylabel("Aircraft type")
    plt.title(f"Aircraft types by median fatality ratio (top {top_n})")
    plt.tight_layout()
    fname = os.path.join(outdir, "aircraft_type_median_fatality_ratio.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

def plot_aboard_vs_fatalities(df, outdir):
    subset = df.dropna(subset=["aboard_total", "fatalities_total"])

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=subset,
        x="aboard_total",
        y="fatalities_total",
        hue="decade",
        alpha=0.6,
        palette="viridis",
    )
    max_aboard = subset["aboard_total"].max()
    plt.plot([0, max_aboard], [0, max_aboard], linestyle="--", color="gray")
    plt.xlabel("Number aboard")
    plt.ylabel("Fatalities")
    plt.title("Fatalities vs number aboard (color = decade)")
    plt.tight_layout()
    fname = os.path.join(outdir, "aboard_vs_fatalities_by_decade.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fatality_ratio_by_decade(df, outdir):
    subset = df[
        (df["fatality_ratio"].notna())
        & (df["fatality_ratio"] >= 0)
        & (df["fatality_ratio"] <= 1)
        & df["decade"].notna()
    ].copy()


    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=subset, x="fatality_ratio", hue="decade", common_norm=False)
    plt.xlabel("Fatalities / aboard")
    plt.title("Distribution of fatality ratios by decade")
    plt.tight_layout()
    fname = os.path.join(outdir, "fatality_ratio_density_by_decade.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hour_hist(df, outdir):
    subset = df["hour"].dropna()

    plt.figure(figsize=(8, 4))
    sns.histplot(subset, bins=24, discrete=True)
    plt.xlabel("Hour of day")
    plt.ylabel("Number of accidents")
    plt.title("Accidents by time of day")
    plt.tight_layout()
    fname = os.path.join(outdir, "accidents_by_hour_of_day.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fatalities_by_group_decade(df, outdir):
    needed = ["decade", "fatalities_passengers", "fatalities_crew", "ground_fatalities"]
    if any(col not in df.columns for col in needed):
        print("Skipping fatalities by group/decade (missing columns).")
        return

    agg = (
        df.groupby("decade", dropna=True)
        .agg(
            pax=("fatalities_passengers", "sum"),
            crew=("fatalities_crew", "sum"),
            ground=("ground_fatalities", "sum"),
        )
        .reset_index()
    )

    if agg.empty:
        print("Skipping fatalities by group/decade (no grouped data).")
        return

    melted = agg.melt(
        id_vars="decade",
        value_vars=["pax", "crew", "ground"],
        var_name="group",
        value_name="fatalities",
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="decade", y="fatalities", hue="group")
    plt.title("Passenger, crew, and ground fatalities by decade")
    plt.xlabel("Decade")
    plt.ylabel("Fatalities")
    plt.tight_layout()
    fname = os.path.join(outdir, "fatalities_by_group_decade.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

def plot_hourly_severity(df, outdir):
    """Existing: accidents & mean fatality ratio by hour."""
    if "hour" not in df.columns or "fatality_ratio" not in df.columns:
        print("Skipping hourly severity plot (missing 'hour' or 'fatality_ratio').")
        return

    subset = df[
        df["hour"].notna()
        & df["fatality_ratio"].notna()
        & (df["fatality_ratio"] >= 0)
        & (df["fatality_ratio"] <= 1)
    ].copy()
    if subset.empty:
        print("Skipping hourly severity plot (no valid rows).")
        return

    agg = (
        subset.groupby("hour")
        .agg(
            crashes=("hour", "size"),
            mean_fatality_ratio=("fatality_ratio", "mean"),
        )
        .reset_index()
        .sort_values("hour")
    )

    fig, ax1 = plt.subplots(figsize=(9, 4))
    sns.barplot(data=agg, x="hour", y="crashes", ax=ax1)
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Number of accidents")

    ax2 = ax1.twinx()
    ax2.plot(
        agg["hour"],
        agg["mean_fatality_ratio"],
        marker="o",
        linestyle="--",
        color="tab:red",
        label="Mean fatality ratio",
    )
    ax2.set_ylabel("Mean fatality ratio")
    ax2.legend(loc="upper right")

    plt.title("Accidents and fatality severity by hour of day")
    plt.tight_layout()
    fname = os.path.join(outdir, "hourly_severity.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_hourly_severity_stacked(df, outdir):
    """Stacked bar chart of non-fatal / partial-fatal / total-loss crashes by hour."""
    if "hour" not in df.columns or "fatality_ratio" not in df.columns:
        print("Skipping hourly severity stacked plot (missing 'hour' or 'fatality_ratio').")
        return

    subset = df[
        df["hour"].notna()
        & df["fatality_ratio"].notna()
        & (df["fatality_ratio"] >= 0)
        & (df["fatality_ratio"] <= 1)
    ].copy()
    if subset.empty:
        print("Skipping hourly severity stacked plot (no valid rows).")
        return

    subset["severity_cat"] = pd.cut(
        subset["fatality_ratio"],
        bins=[-0.01, 0, 0.99, 1.01],
        labels=["Non-fatal", "Partial fatal", "Total-loss"],
    )

    hour_sev = (
        subset.groupby(["hour", "severity_cat"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(12, 6))
    hour_sev.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.xlabel("Hour of day")
    plt.ylabel("Number of crashes")
    plt.title("Crash severity distribution by hour of day")
    plt.tight_layout()
    fname = os.path.join(outdir, "hourly_severity_stacked.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")



def plot_aircraft_category_trends(df, outdir):
    needed = ["aircraft_category", "decade"]
    if any(col not in df.columns for col in needed):
        print("Skipping aircraft category trends (missing columns).")
        return

    sub = df.dropna(subset=needed)
    if sub.empty:
        print("Skipping aircraft category trends (no valid rows).")
        return

    agg = (
        sub.groupby(["decade", "aircraft_category"])
        .size()
        .reset_index(name="crashes")
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=agg,
        x="decade",
        y="crashes",
        hue="aircraft_category",
        marker="o"
    )
    plt.xlabel("Decade")
    plt.ylabel("Crashes")
    plt.title("Crashes per decade by aircraft category")
    plt.tight_layout()

    fname = os.path.join(outdir, "aircraft_category_trends.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", fname)



def plot_weather_condition_counts(df, outdir):

    sub = df["weather_condition"].dropna()

    plt.figure(figsize=(9, 5))
    sns.countplot(
        y="weather_condition",
        data=df,
        order=sub.value_counts().index
    )
    plt.xlabel("Number of crashes")
    plt.ylabel("Weather condition")
    plt.title("Crash count by weather condition")
    plt.tight_layout()

    fname = os.path.join(outdir, "weather_condition_counts.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_aircraft_decade_proportion(df, outdir):
    """
    For each decade, show what fraction of crashes are in each aircraft category.
    This complements the 'crashes per decade' plot by focusing on proportions.
    """
    needed = ["aircraft_category", "decade"]

    sub = df.dropna(subset=needed)

    counts = (
        sub.groupby(["decade", "aircraft_category"])
        .size()
        .reset_index(name="crashes")
    )
    totals = counts.groupby("decade")["crashes"].transform("sum")
    counts["proportion"] = counts["crashes"] / totals

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=counts,
        x="decade",
        y="proportion",
        hue="aircraft_category",
        marker="o"
    )
    plt.xlabel("Decade")
    plt.ylabel("Proportion of crashes")
    plt.title("Proportion of crashes by aircraft category over time")
    plt.tight_layout()

    fname = os.path.join(outdir, "aircraft_decade_proportion.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_aircraft_median_fatalities(df, outdir):
    """
    For each aircraft category, show the median number of fatalities
    in crashes (including zeros). This is a severity measure.
    """
    needed = ["aircraft_category", "fatalities_total"]

    sub = df.dropna(subset=needed)

    agg = (
        sub.groupby("aircraft_category")["fatalities_total"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=agg,
        x="fatalities_total",
        y="aircraft_category",
        palette="magma"
    )
    plt.xlabel("Median fatalities per crash")
    plt.ylabel("Aircraft category")
    plt.title("Median fatalities per crash by aircraft category")
    plt.tight_layout()

    fname = os.path.join(outdir, "aircraft_median_fatalities.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_survival_rate_by_decade(df, outdir):
    """
    Show survival rate (1 - fatality_ratio) trends across decades.
    """
    subset = df[
        df["fatality_ratio"].notna()
        & (df["fatality_ratio"] >= 0)
        & (df["fatality_ratio"] <= 1)
        & df["decade"].notna()
    ].copy()

    subset["survival_rate"] = 1 - subset["fatality_ratio"]

    agg = (
        subset.groupby("decade")
        .agg(
            mean_survival=("survival_rate", "mean"),
            median_survival=("survival_rate", "median"),
            total_accidents=("survival_rate", "count"),
        )
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "#2ecc71"
    color2 = "#3498db"
    ax1.bar(agg["decade"], agg["total_accidents"], alpha=0.3, color="gray", label="Total accidents")
    ax1.set_xlabel("Decade")
    ax1.set_ylabel("Number of accidents", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")

    ax2 = ax1.twinx()
    ax2.plot(agg["decade"], agg["mean_survival"], marker="o", color=color1, linewidth=2, label="Mean survival rate")
    ax2.plot(agg["decade"], agg["median_survival"], marker="s", color=color2, linewidth=2, linestyle="--", label="Median survival rate")
    ax2.set_ylabel("Survival rate")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")

    plt.title("Survival Rate Trends Across Decades")
    plt.tight_layout()
    fname = os.path.join(outdir, "survival_rate_by_decade.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_flight_phase_analysis(df, outdir):
    """
    Analyze accidents by flight phase (extracted from summary).
    """
    subset = df["phase_clean"].dropna()
    phase_counts = subset.value_counts()
    colors = sns.color_palette("rocket", len(phase_counts))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(phase_counts.index, phase_counts.values, color=colors)
    plt.xlabel("Number of Accidents")
    plt.ylabel("Flight Phase")
    plt.title("Accidents by Flight Phase")

    for bar, val in zip(bars, phase_counts.values):
        plt.text(val + 10, bar.get_y() + bar.get_height()/2, str(val),
                 va="center", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(outdir, "accidents_by_flight_phase.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()



def plot_monthly_seasonal_pattern(df, outdir):
    """
    Show monthly accident patterns to identify seasonal trends.
    """
    df_copy = df.copy()
    df_copy["date_parsed"] = pd.to_datetime(df_copy["date_parsed"], errors="coerce")
    df_copy["month"] = df_copy["date_parsed"].dt.month
    df_copy["month_name"] = df_copy["date_parsed"].dt.month_name()

    subset = df_copy.dropna(subset=["month"])

    monthly = subset.groupby(["month", "month_name"]).size().reset_index(name="accidents")
    monthly = monthly.sort_values("month")

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(data=monthly, x="month_name", y="accidents", palette="coolwarm")
    plt.xlabel("Month")
    plt.ylabel("Number of Accidents")
    plt.title("Seasonal Pattern: Accidents by Month")
    plt.xticks(rotation=45, ha="right")

    ax.plot(range(len(monthly)), monthly["accidents"].values, color="darkred", linewidth=2, marker="o")

    plt.tight_layout()
    fname = os.path.join(outdir, "monthly_accident_pattern.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fatality_ratio_boxplot_by_category(df, outdir):
    """
    Boxplot showing fatality ratio distribution for each aircraft category.
    """
    needed = ["aircraft_category", "fatality_ratio"]

    subset = df[
        df["fatality_ratio"].notna()
        & (df["fatality_ratio"] >= 0)
        & (df["fatality_ratio"] <= 1)
        & df["aircraft_category"].notna()
    ].copy()

    order = subset.groupby("aircraft_category")["fatality_ratio"].median().sort_values(ascending=False).index

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=subset,
        x="aircraft_category",
        y="fatality_ratio",
        order=order,
        palette="RdYlGn_r"
    )
    plt.xlabel("Aircraft Category")
    plt.ylabel("Fatality Ratio")
    plt.title("Fatality Ratio Distribution by Aircraft Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fname = os.path.join(outdir, "fatality_ratio_boxplot_by_category.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()



def plot_decade_heatmap(df, outdir):
    """
    Heatmap showing accidents across decades and aircraft categories.
    """
    needed = ["decade", "aircraft_category"]


    subset = df.dropna(subset=needed)

    pivot = subset.pivot_table(
        index="aircraft_category",
        columns="decade",
        aggfunc="size",
        fill_value=0
    )

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Number of Accidents"}
    )
    plt.xlabel("Decade")
    plt.ylabel("Aircraft Category")
    plt.title("Accident Frequency: Aircraft Category vs Decade")
    plt.tight_layout()

    fname = os.path.join(outdir, "decade_category_heatmap.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cumulative_fatalities(df, outdir):
    """
    Show cumulative fatalities over time.
    """
    df_copy = df.copy()
    df_copy["date_parsed"] = pd.to_datetime(df_copy["date_parsed"], errors="coerce")

    subset = df_copy.dropna(subset=["date_parsed", "fatalities_total"])

    subset = subset.sort_values("date_parsed")
    subset["cumulative_fatalities"] = subset["fatalities_total"].cumsum()
    subset["cumulative_accidents"] = range(1, len(subset) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.fill_between(subset["date_parsed"], subset["cumulative_fatalities"], alpha=0.4, color="#e74c3c")
    ax1.plot(subset["date_parsed"], subset["cumulative_fatalities"], color="#c0392b", linewidth=1.5)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cumulative Fatalities", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")

    ax2 = ax1.twinx()
    ax2.plot(subset["date_parsed"], subset["cumulative_accidents"], color="#3498db", linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Cumulative Accidents", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    plt.title("Cumulative Aviation Fatalities & Accidents Over Time")
    plt.tight_layout()
    fname = os.path.join(outdir, "cumulative_fatalities.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_crew_vs_passenger_fatalities(df, outdir):
    """
    Scatter plot comparing crew vs passenger fatalities.
    """
    needed = ["fatalities_passengers", "fatalities_crew"]
    subset = df.dropna(subset=needed)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        subset["fatalities_passengers"],
        subset["fatalities_crew"],
        c=subset["decade"] if "decade" in subset.columns else None,
        cmap="viridis",
        alpha=0.5,
        s=30
    )
    if "decade" in subset.columns:
        plt.colorbar(scatter, label="Decade")

    max_val = max(subset["fatalities_passengers"].max(), subset["fatalities_crew"].max())
    plt.plot([0, max_val], [0, max_val], linestyle="--", color="gray", alpha=0.5)

    plt.xlabel("Passenger Fatalities")
    plt.ylabel("Crew Fatalities")
    plt.title("Crew vs Passenger Fatalities per Accident")
    plt.tight_layout()

    fname = os.path.join(outdir, "crew_vs_passenger_fatalities.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_weather_vs_fatality_ratio(df, outdir):
    """
    Show mean fatality ratio by weather condition.
    """
    needed = ["weather_condition", "fatality_ratio"]


    subset = df[
        df["weather_condition"].notna()
        & df["fatality_ratio"].notna()
        & (df["fatality_ratio"] >= 0)
        & (df["fatality_ratio"] <= 1)
    ].copy()


    agg = (
        subset.groupby("weather_condition")
        .agg(
            mean_ratio=("fatality_ratio", "mean"),
            count=("fatality_ratio", "count")
        )
        .reset_index()
        .sort_values("mean_ratio", ascending=False)
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.barh(agg["weather_condition"], agg["mean_ratio"], color=sns.color_palette("Reds_r", len(agg)))
    ax1.set_xlabel("Mean Fatality Ratio")
    ax1.set_ylabel("Weather Condition")
    ax1.set_xlim(0, 1)

    for i, (ratio, count) in enumerate(zip(agg["mean_ratio"], agg["count"])):
        ax1.text(ratio + 0.02, i, f"n={count}", va="center", fontsize=8)

    plt.title("Fatality Severity by Weather Condition")
    plt.tight_layout()
    fname = os.path.join(outdir, "weather_vs_fatality_ratio.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()



def plot_phase_fatality_heatmap(df, outdir):
    """
    Heatmap showing fatality ratio by flight phase and decade.
    """
    needed = ["phase_clean", "decade", "fatality_ratio"]

    subset = df[
        df["phase_clean"].notna()
        & df["decade"].notna()
        & df["fatality_ratio"].notna()
        & (df["fatality_ratio"] >= 0)
        & (df["fatality_ratio"] <= 1)
    ].copy()


    pivot = subset.pivot_table(
        values="fatality_ratio",
        index="phase_clean",
        columns="decade",
        aggfunc="mean"
    )

    plt.figure(figsize=(14, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Mean Fatality Ratio"}
    )
    plt.xlabel("Decade")
    plt.ylabel("Flight Phase")
    plt.title("Mean Fatality Ratio by Flight Phase and Decade")
    plt.tight_layout()

    fname = os.path.join(outdir, "phase_fatality_heatmap.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()



def plot_top_deadly_years(df, outdir, top_n=15):
    """
    Bar chart showing years with the highest total fatalities.
    """
    subset = df.dropna(subset=["year", "fatalities_total"])

    agg = (
        subset.groupby("year")
        .agg(
            total_fatalities=("fatalities_total", "sum"),
            num_accidents=("fatalities_total", "count")
        )
        .reset_index()
        .nlargest(top_n, "total_fatalities")
        .sort_values("total_fatalities", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("Reds", len(agg))
    bars = ax.barh(agg["year"].astype(int).astype(str), agg["total_fatalities"], color=colors)

    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(row["total_fatalities"] + 20, bar.get_y() + bar.get_height()/2,
                f"{int(row['num_accidents'])} crashes", va="center", fontsize=8)

    ax.set_xlabel("Total Fatalities")
    ax.set_ylabel("Year")
    ax.set_title(f"Top {top_n} Deadliest Years in Aviation History")
    plt.tight_layout()

    fname = os.path.join(outdir, "top_deadly_years.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ground_fatalities_analysis(df, outdir):
    """
    Analyze ground fatalities distribution.
    """
    if "ground_fatalities" not in df.columns:
        print("Skipping ground fatalities plot (missing column).")
        return

    subset = df[df["ground_fatalities"].notna() & (df["ground_fatalities"] > 0)].copy()
    if subset.empty:
        print("Skipping ground fatalities plot (no ground fatalities data).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(subset["ground_fatalities"], bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Ground Fatalities")
    axes[0].set_ylabel("Number of Accidents")
    axes[0].set_title("Distribution of Ground Fatalities")
    axes[0].set_yscale("log")

    top_ground = subset.nlargest(10, "ground_fatalities")[["date_parsed", "location", "ground_fatalities", "fatalities_total"]]
    if not top_ground.empty:
        y_labels = [f"{row['date_parsed']}" if pd.notna(row['date_parsed']) else "Unknown" for _, row in top_ground.iterrows()]
        axes[1].barh(range(len(top_ground)), top_ground["ground_fatalities"], color="#8e44ad")
        axes[1].set_yticks(range(len(top_ground)))
        axes[1].set_yticklabels(y_labels, fontsize=8)
        axes[1].set_xlabel("Ground Fatalities")
        axes[1].set_title("Top 10 Accidents by Ground Fatalities")
        axes[1].invert_yaxis()

    plt.tight_layout()
    fname = os.path.join(outdir, "ground_fatalities_analysis.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_operator_safety_comparison(df, outdir, min_accidents=20):

    needed = ["operator", "fatality_ratio"]
    if any(col not in df.columns for col in needed):
        print("Skipping operator safety plot (missing columns).")
        return

    subset = df[
        df["operator"].notna()
        & df["fatality_ratio"].notna()
        & (df["fatality_ratio"] >= 0)
        & (df["fatality_ratio"] <= 1)
    ].copy()


    agg = (
        subset.groupby("operator")
        .agg(
            mean_ratio=("fatality_ratio", "mean"),
            num_accidents=("fatality_ratio", "count"),
            total_fatalities=("fatalities_total", "sum") if "fatalities_total" in subset.columns else ("fatality_ratio", "count")
        )
        .reset_index()
    )

    agg = agg[agg["num_accidents"] >= min_accidents]


    agg = agg.sort_values("mean_ratio", ascending=True).tail(20)

    plt.figure(figsize=(12, 8))
    colors = ["#2ecc71" if r < 0.5 else "#e74c3c" if r > 0.8 else "#f39c12" for r in agg["mean_ratio"]]
    bars = plt.barh(agg["operator"], agg["mean_ratio"], color=colors)

    for bar, count in zip(bars, agg["num_accidents"]):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f"n={int(count)}", va="center", fontsize=8)

    plt.xlabel("Mean Fatality Ratio")
    plt.ylabel("Operator")
    plt.title(f"Operator Fatality Comparison (min {min_accidents} accidents)")
    plt.xlim(0, 1.1)
    plt.tight_layout()

    fname = os.path.join(outdir, "operator_safety_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    outdir = ensure_output_dir()
    df = load_data()
    df = preprocess(df)

   
    plot_yearly_trends(df, outdir)
    plot_top_countries(df, outdir)

    plot_top_operators(df, outdir)

    plot_aircraft_severity(df, outdir)
    

    plot_aboard_vs_fatalities(df, outdir)
    
    plot_fatality_ratio_by_decade(df, outdir)
    
    plot_hour_hist(df, outdir)
    
    plot_fatalities_by_group_decade(df, outdir)

    plot_hourly_severity(df, outdir)
    
    plot_hourly_severity_stacked(df, outdir)
    

    plot_aircraft_category_trends(df, outdir)
    
    plot_weather_condition_counts(df, outdir)
    
    plot_aircraft_decade_proportion(df, outdir)
    
    plot_aircraft_median_fatalities(df, outdir)

    # === NEW Analysis Plots ===
    plot_survival_rate_by_decade(df, outdir)
    
    plot_flight_phase_analysis(df, outdir)
    
    plot_monthly_seasonal_pattern(df, outdir)
    
    plot_fatality_ratio_boxplot_by_category(df, outdir)
    
    plot_decade_heatmap(df, outdir)
    
    plot_cumulative_fatalities(df, outdir)
    
    plot_crew_vs_passenger_fatalities(df, outdir)
    
    plot_weather_vs_fatality_ratio(df, outdir)
    
    plot_phase_fatality_heatmap(df, outdir)
    
    plot_top_deadly_years(df, outdir)
    
    plot_ground_fatalities_analysis(df, outdir)
    
    plot_operator_safety_comparison(df, outdir)


if __name__ == "__main__":
    main()

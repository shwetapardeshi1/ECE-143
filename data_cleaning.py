import re

import pandas as pd

RAW_CSV = "./planecrashinfo_accidents.csv"
CLEAN_CSV = "./planecrashinfo_clean.csv"

US_STATES = {
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
}

US_ABBREVS = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}

KNOWN_COUNTRIES = {
    "United States",
    "USA",
    "U.S.A.",
    "U.S.",
    "United States of America",
    "Canada",
    "Mexico",
    "England",
    "United Kingdom",
    "UK",
    "Scotland",
    "Wales",
    "Northern Ireland",
    "France",
    "Germany",
    "Belgium",
    "Italy",
    "Spain",
    "Portugal",
    "Netherlands",
    "Switzerland",
    "Austria",
    "Sweden",
    "Norway",
    "Finland",
    "Denmark",
    "Russia",
    "Soviet Union",
    "Japan",
    "China",
    "India",
    "Australia",
    "New Zealand",
    "Brazil",
    "Argentina",
    "Chile",
    "South Africa",
}


def read_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        on_bad_lines="skip",
    )
    print("Read data with shape:", df.shape)
    print("Original columns:", list(df.columns))
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    col_map = {}
    for col in df.columns:
        col_clean = col.strip().lower()

        if col_clean.startswith("aboard"):
            col_map[col] = "aboard"
        elif "type" in col_clean:
            col_map[col] = "aircraft_type"
        elif col_clean.startswith("cn"):
            col_map[col] = "cn_ln"
        elif col_clean == "date":
            col_map[col] = "date"
        elif col_clean == "detail_url":
            col_map[col] = "detail_url"
        elif "fatalit" in col_clean:
            col_map[col] = "fatalities"
        elif "flight" in col_clean:
            col_map[col] = "flight_no"
        elif col_clean in ("ground", "ground_fatalities"):
            col_map[col] = "ground_fatalities"
        elif col_clean == "location":
            col_map[col] = "location"
        elif "operator" in col_clean:
            col_map[col] = "operator"
        elif "registr" in col_clean:
            col_map[col] = "registration"
        elif col_clean == "route":
            col_map[col] = "route"
        elif col_clean == "summary":
            col_map[col] = "summary"
        elif col_clean == "time":
            col_map[col] = "time"
        elif "year_page_url" in col_clean:
            col_map[col] = "year_page_url"
        else:
            tmp = re.sub(r"[^0-9a-zA-Z]+", "_", col_clean).strip("_")
            col_map[col] = tmp or col_clean

    df = df.rename(columns=col_map)
    print("Normalized columns:", list(df.columns))
    return df


def parse_fatalities(text: str):
    if pd.isna(text):
        return None, None, None

    s = str(text)

    m_total = re.search(r"(\d+)", s)
    total = int(m_total.group(1)) if m_total else None

    m_pax = re.search(r"passengers:\s*([0-9?]+)", s, re.IGNORECASE)
    pax = None
    if m_pax and m_pax.group(1) != "?":
        pax = int(m_pax.group(1))

    m_crew = re.search(r"crew:\s*([0-9?]+)", s, re.IGNORECASE)
    crew = None
    if m_crew and m_crew.group(1) != "?":
        crew = int(m_crew.group(1))

    return total, pax, crew


def split_location(loc: str):
    if pd.isna(loc):
        return None, None, None

    s = str(loc).strip()
    if not s:
        return None, None, None

    if "," not in s:
        if s in KNOWN_COUNTRIES or any(ctry in s for ctry in KNOWN_COUNTRIES):
            return None, None, s
        else:
            return s, None, None

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1:
        return parts[0], None, None
    elif len(parts) == 2:
        city_region = parts[0]
        last = parts[1]

        if last in KNOWN_COUNTRIES:
            return city_region, None, last

        if last in US_STATES or last in US_ABBREVS:
            return city_region, last, "United States"

        if any(ctry in last for ctry in KNOWN_COUNTRIES):
            return city_region, None, last

        return city_region, last, None
    else:
        city_region = parts[0]
        state = parts[1]
        country = parts[-1]
        return city_region, state, country


def parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def parse_time_col(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        return df

    df["time_raw"] = df["time"]

    def _parse_time(t):
        if pd.isna(t):
            return None
        s = str(t).strip()
        if s == "?" or s == "":
            return None

        s = re.sub(r"\D", "", s)
        if not s:
            return None

        if len(s) <= 2:
            return None
        if len(s) == 3:
            s = "0" + s
        elif len(s) > 4:
            s = s[-4:]

        hh = int(s[:2])
        mm = int(s[2:4])
        if hh > 23 or mm > 59:
            return None
        return f"{hh:02d}:{mm:02d}"

    df["time_hhmm"] = df["time"].apply(_parse_time)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    df = parse_date_col(df)

    df = parse_time_col(df)

    if "fatalities" in df.columns:
        totals = df["fatalities"].apply(lambda x: parse_fatalities(x))
        df["fatalities_total"] = totals.apply(lambda x: x[0])
        df["fatalities_passengers"] = totals.apply(lambda x: x[1])
        df["fatalities_crew"] = totals.apply(lambda x: x[2])

        df["fatalities_total"] = pd.to_numeric(df["fatalities_total"], errors="coerce")
        df["fatalities_passengers"] = pd.to_numeric(
            df["fatalities_passengers"], errors="coerce"
        )
        df["fatalities_crew"] = pd.to_numeric(df["fatalities_crew"], errors="coerce")

    if "location" in df.columns:
        loc_split = df["location"].apply(lambda x: split_location(x))
        df["location_city"] = loc_split.apply(lambda x: x[0])
        df["location_state"] = loc_split.apply(lambda x: x[1])
        df["location_country"] = loc_split.apply(lambda x: x[2])

    if "ground_fatalities" in df.columns:
        df["ground_fatalities"] = pd.to_numeric(
            df["ground_fatalities"], errors="coerce"
        )

    if "aircraft_type" in df.columns:
        atype = df["aircraft_type"].astype(str)

        def categorize_aircraft(s: str) -> str:
            if pd.isna(s) or s.strip() == "" or s.strip() == "?":
                return "Unknown"
            s = s.lower()

            if any(k in s for k in [
                "helicopter", " heli ", "bell ", "uh-", "ch-", "mh-", "ah-",
                "s-61", "s-76", "mi-8", "mi-17", "mi-24"
            ]):
                return "Helicopter"

            if "glider" in s or "sailplane" in s:
                return "Glider"

            if any(k in s for k in [
                "amphibian", "seaplane", "floatplane", "flying boat",
                "catalina", "pby", "goose", "otter", "beaver", "sunderland"
            ]):
                return "Amphibian/Seaplane"

            if any(k in s for k in [
                " c-1", " c-2", " c-3", " c-4", "kc-", "ec-", "rc-",
                " f-", " b-17", " b-24", " b-29", " b-52",
                "mig", "mig-", "su-", "tu-", "an-12", "an-22", "il-76",
            ]):
                return "Military"

            if any(k in s for k in [
                "boeing", "airbus", "embraer", "erj", "e-jet",
                "bombardier", "crj", "md-", "dc-9", "dc-10", "l-1011",
                "fokker 70", "fokker 100", "yak-40", "yak-42", "tu-134", "tu-154"
            ]):
                return "Jet"

            if any(k in s for k in [
                "turboprop", " turbo prop", "dhc-", "dash 8", "atr-",
                "saab 340", "saab 2000", "fokker 27", "fokker 50", "hs-748",
                "l-188", "herald", "shorts 3", "shorts-3", "metro ii", "metro iii",
                "an-24", "an-26", "an-32", "an-72", "casa 212", "jetstream 31", "jetstream 32"
            ]):
                return "Turboprop"

            if any(k in s for k in [
                "cessna", "piper", "beech", "king air", "baron",
                "bonanza", "mooney", "seneca", "aztec", "navajo",
                "dc-3", "dakota", "convair", "cv-", "dc-4", "dc-6", "dc-7",
                "an-2", "il-14", "lockheed 10", "lockheed 12", "lockheed 18",
                "do-", "dornier", "y-7"
            ]):
                return "Piston/Prop"

            if any(k in s for k in [
                "trimotor", "tri-motor", "waco", "curtiss", "junker", "junkers",
                "tiger moth", "biplane", "stearman"
            ]):
                return "Vintage/Early"

            return "Other/Unmapped"

        df["aircraft_category"] = atype.apply(categorize_aircraft)
    else:
        df["aircraft_category"] = pd.NA

    if "summary" in df.columns:
        summ = df["summary"].astype(str).str.lower()

        def extract_phase(s: str) -> str:
            if any(x in s for x in ["taxi", "ground", "parked"]):
                return "Ground/Taxi"
            if any(x in s for x in ["takeoff", "shortly after takeoff", "rotation"]):
                return "Takeoff"
            if "initial climb" in s:
                return "Initial climb"
            if " climb" in s:
                return "Climb"
            if any(x in s for x in ["cruise", "en route", "enroute"]):
                return "Cruise"
            if "descent" in s:
                return "Descent"
            if any(x in s for x in ["approach", "final approach", "ils"]):
                return "Approach"
            if any(x in s for x in ["landing", "touchdown", "flare"]):
                return "Landing"
            if any(x in s for x in ["go-around", "missed approach"]):
                return "Go-around"
            return "Unknown"

        df["phase_clean"] = summ.apply(extract_phase)
    else:
        df["phase_clean"] = pd.NA

    if "summary" in df.columns:
        summ = df["summary"].astype(str).str.lower()

        def extract_weather(s: str) -> str:
            if not s or s.strip() == "" or s.strip() == "?":
                return "None/Not mentioned"

            if any(x in s for x in [
                "thunderstorm", "thunder storm", "t-storm", "tstorm",
                "storm", "squall", "microburst", "downburst", "heavy storm"
            ]):
                return "Storm/Thunderstorm"

            if any(x in s for x in [
                "fog", "mist", "low visibility", "reduced visibility",
                "poor visibility", "haze", "smog", "whiteout"
            ]):
                return "Fog/Low visibility"

            if any(x in s for x in [
                "snow", "blizzard", "sleet", "snowstorm", "snow storm",
                "icy runway", "ice on runway", "runway ice"
            ]):
                return "Snow/Icy surface"

            if any(x in s for x in [
                "icing", "ice accretion", "wing ice", "airframe ice",
                "freezing rain", "freezing drizzle"
            ]):
                return "Icing (in-flight)"

            if any(x in s for x in [
                "rain", "heavy rain", "rainstorm", "rain storm", "showers", "downpour"
            ]):
                return "Rain"

            if any(x in s for x in [
                "wind shear", "windshear", "crosswind", "cross wind", "gust",
                "strong winds", "gusty", "tailwind", "headwind"
            ]):
                return "Wind/Wind shear"

            if "turbulence" in s:
                return "Turbulence"

            if any(x in s for x in ["clear weather", "good weather", "vfr conditions", "clear skies"]):
                return "Good/Visual conditions"

            return "None/Not mentioned"

        df["weather_condition"] = summ.apply(extract_weather)

        def has_adverse(w: str) -> bool:
            if w in (
                "Storm/Thunderstorm",
                "Fog/Low visibility",
                "Snow/Icy surface",
                "Icing (in-flight)",
                "Rain",
                "Wind/Wind shear",
                "Turbulence",
            ):
                return True
            return False

        df["weather_adverse"] = df["weather_condition"].apply(has_adverse)
    else:
        df["weather_condition"] = pd.NA
        df["weather_adverse"] = pd.NA

    return df


def main():
    df = read_raw_data(RAW_CSV)
    df_clean = clean_dataset(df)

    df_clean.to_csv(CLEAN_CSV, index=False)
    print(f"Saved cleaned data to {CLEAN_CSV}")
    print(df_clean.head())


if __name__ == "__main__":
    main()

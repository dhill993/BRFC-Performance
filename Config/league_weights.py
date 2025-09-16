# Config/league_weights.py
LEAGUE_WEIGHTS = {
        "Premier League": 0.95,
        "Spain 1": 0.9,
        "Italy 1": 0.89,
        "Germany 1": 0.89,
        "France 1": 0.87,
        "Netherlands 1": 0.82,
        "Portugal 1": 0.82,
        "Belgium 1": 0.82,
        "Championship": 0.81,
        "MLS": 0.79,
        "Japan 1": 0.78,
        "Turkey 1": 0.78,
        "Norway 1": 0.77,
        "Denmark 1": 0.77,
        "Austria 1": 0.77,
        "Italy 2": 0.76,
        "Croatia 1": 0.76,
        "Premiership": 0.76,
        "Poland 1": 0.76,
        "Allsvenskan": 0.76,
        "NB I": 0.75,
        "Czech 1": 0.75,
        "Germany 2": 0.75,
        "Greece 1": 0.75,
        "Romania 1": 0.75,
        "Spain 2": 0.75,
        "K League 1": 0.74,
        "Swiss 1": 0.74,
        "Serbia 1": 0.73,
        "France 2": 0.73,
        "League One": 0.72,
        "Slovakia 1": 0.72,
        "Bulgaria 1": 0.71,
        "1. SNL": 0.7,
        "Belgium 2": 0.7,
        "Portugal 2": 0.7,
        "Netherlands 2": 0.7,
        "A-League": 0.69,
        "Germany 3": 0.68,
        "France 3": 0.68,
        "Japan 2": 0.68,
        "Iceland 1": 0.67,
        "Finland 1": 0.67,
        "Latvia 1": 0.66,
        "Norway 2": 0.66,
        "Sweden 2": 0.66,
        "Swizz 2": 0.66,
        "Moldova 1": 0.64,
        "K League 2": 0.64,
        "League Two": 0.64,
        "Austria 2": 0.64,
        "Denmark 2": 0.63,
        "Scotland 2": 0.62,
        "Canada 1": 0.62,
        "Hungary 2": 0.62,
        "Ireland 1": 0.61,
        "Germany 4": 0.6,
        "Turkey 2": 0.6,
        "National League": 0.6,
        "Wales 1": 0.58,
        "NIreland1": 0.58,
        "Ireland 2": 0.52,
        "Scotland 3": 0.52,
        "VNL 2": 0.48
    }

DEFAULT_LEAGUE_WEIGHT = 0.70

def normalize_weight_key(country_name, competition_name) -> str:
    """
    Convert raw API (country_name, competition_name) -> a key in LEAGUE_WEIGHTS.
    Disambiguates Germany vs Austria "Bundesliga" and maps common aliases.
    Accepts any type and coerces to safe lowercase strings.
    """
    # Coerce to strings safely
    c = "" if country_name is None else str(country_name)
    n = "" if competition_name is None else str(competition_name)
    # Some feeds send floats like nan; guard them
    if c.lower() in {"nan", "none"}:
        c = ""
    if n.lower() in {"nan", "none"}:
        n = ""

    c = c.strip()
    n = n.strip()
    cl = c.casefold()
    nl = n.casefold()

    # If either is missing, return raw competition so caller can still try LEAGUE_WEIGHTS.get(...)
    if not c or not n:
        return n

    # England
    if cl == "england":
        if nl == "premier league": return "Premier League"
        if nl == "championship": return "Championship"
        if nl in {"league one", "english league one"}: return "League One"
        if nl in {"league two", "english league two"}: return "League Two"
        if nl == "national league": return "National League"

    # Germany
    if cl == "germany":
        if nl in {"1. bundesliga", "bundesliga 1", "bundesliga i", "bundesliga"}: return "Germany 1"
        if nl in {"2. bundesliga", "bundesliga 2", "2. bundesliga - liga 2"}: return "Germany 2"
        if nl in {"3. liga", "bundesliga 3"}: return "Germany 3"
        if nl in {"regionalliga", "bundesliga 4"}: return "Germany 4"

    # Austria
    if cl == "austria":
        if "bundesliga" in nl: return "Austria 1"
        if nl in {"2. liga", "austria 2"}: return "Austria 2"

    # Spain
    if cl == "spain":
        if nl in {"la liga", "primera division"}: return "Spain 1"
        if nl in {"la liga 2", "segunda division", "laliga 2"}: return "Spain 2"

    # France
    if cl == "france":
        if nl == "ligue 1": return "France 1"
        if nl == "ligue 2": return "France 2"
        if nl == "championnat national": return "France 3"

    # Italy
    if cl == "italy":
        if nl == "serie a": return "Italy 1"
        if nl in {"serie b", "italy serie b"}: return "Italy 2"

    # Portugal
    if cl == "portugal":
        if nl in {"primeira liga", "liga nos"}: return "Portugal 1"
        if nl == "liga portugal 2": return "Portugal 2"

    # Netherlands
    if cl == "netherlands":
        if nl == "eredivisie": return "Netherlands 1"
        if nl == "eerste divisie": return "Netherlands 2"

    # Switzerland
    if cl == "switzerland":
        if nl in {"swiss super league", "super league"}: return "Swiss 1"
        if nl == "challenge league": return "Swiss 2"

    # Belgium
    if cl == "belgium":
        if nl == "jupiler pro league": return "Belgium 1"
        if nl in {"challenger pro league", "first division b"}: return "Belgium 2"

    # Denmark
    if cl == "denmark":
        if nl == "superliga": return "Denmark 1"
        if nl in {"1st division", "1. division"}: return "Denmark 2"

    # Norway
    if cl == "norway":
        if nl == "eliteserien": return "Norway 1"
        if nl in {"obos-ligaen", "1. divisjon"}: return "Norway 2"

    # Poland
    if cl == "poland" and nl == "ekstraklasa":
        return "Poland 1"

    # Scotland
    if cl == "scotland":
        if nl == "premiership": return "Premiership"
        if nl == "championship": return "Scotland 2"
        if nl in {"league one", "scottish league one"}: return "Scotland 3"

    # Japan
    if cl == "japan":
        if nl == "j1 league": return "Japan 1"
        if nl == "j2 league": return "Japan 2"

    # USA
    if cl in {"united states of america", "usa", "united states"} and nl == "major league soccer":
        return "MLS"

    # Sweden
    if cl == "sweden" and nl == "allsvenskan":
        return "Allsvenskan"

    # Already a weight key
    if n in LEAGUE_WEIGHTS:
        return n

    # Try "<Country> X" pattern if competition ends with a digit
    if n and n[-1].isdigit():
        candidate = f"{c} {n[-1]}"
        if candidate in LEAGUE_WEIGHTS:
            return candidate

    # Fallback to raw competition name so caller can still look it up
    return n


def league_weight(country_name: str, competition_name: str) -> float:
    """
    Public helper: get the numeric weight for a given country + competition.
    """
    key = normalize_weight_key(country_name, competition_name)
    return LEAGUE_WEIGHTS.get(key, DEFAULT_LEAGUE_WEIGHT)
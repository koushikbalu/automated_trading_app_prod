"""Stock universe, sector map, and NSE freeze quantities for the momentum strategy."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)


def load_universe(
    csv_path: str | Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Load ticker universe and sector map from a CSV file.

    Falls back to the hardcoded BROAD_UNIVERSE / SECTOR_MAP if the file
    doesn't exist or is unreadable.

    Returns (tickers, sector_map).
    """
    path = Path(csv_path) if csv_path else Path(__file__).parent / "universe.csv"
    if not path.exists():
        _logger.info("No universe.csv found, using hardcoded universe")
        return BROAD_UNIVERSE, SECTOR_MAP

    try:
        tickers: list[str] = []
        sectors: dict[str, str] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row["ticker"].strip()
                sector = row.get("sector", "Other").strip()
                tickers.append(ticker)
                sectors[ticker] = sector
        if tickers:
            _logger.info("Loaded %d tickers from %s", len(tickers), path)
            return tickers, sectors
        _logger.warning("universe.csv was empty, using hardcoded universe")
        return BROAD_UNIVERSE, SECTOR_MAP
    except Exception as exc:
        _logger.warning("Failed to load universe.csv (%s), using hardcoded", exc)
        return BROAD_UNIVERSE, SECTOR_MAP

BROAD_UNIVERSE = [
    # Banking
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "INDUSINDBK", "PNB", "BANKBARODA", "CANBK", "FEDERALBNK",
    "IDFCFIRSTB", "BANDHANBNK", "AUBANK", "IDBI", "INDIANB",
    "UNIONBANK", "CENTRALBK", "IOB", "MAHABANK",
    # NBFC / Insurance / Capital Markets
    "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN",
    "MUTHOOTFIN", "MANAPPURAM", "HDFCLIFE", "SBILIFE",
    "ICICIPRULI", "SBICARD", "PFC", "RECLTD", "LICHSGFIN",
    "MFSL", "ABCAPITAL", "LICI", "JIOFIN", "IREDA",
    # IT
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "NAUKRI",
    "LTTS", "KPITTECH", "OFSS", "TATAELXSI",
    # Energy / Oil & Gas / Power
    "RELIANCE", "ONGC", "BPCL", "IOC", "GAIL",
    "HINDPETRO", "ADANIGREEN", "ADANIPOWER", "NTPC",
    "POWERGRID", "TATAPOWER", "JSWENERGY", "NHPC", "SJVN",
    "IRFC", "ATGL", "TORNTPOWER", "SUZLON",
    # Metals & Mining
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL",
    "COALINDIA", "NMDC", "SAIL", "NATIONALUM",
    # Cement
    "ULTRACEMCO", "AMBUJACEM", "ACC", "SHREECEM",
    # Auto
    "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO",
    "EICHERMOT", "HEROMOTOCO", "TVSMOTOR", "ASHOKLEY",
    "ESCORTS", "MOTHERSON",
    # Pharma & Healthcare
    "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB",
    "TORNTPHARM", "LUPIN", "AUROPHARMA", "APOLLOHOSP",
    "ALKEM", "ZYDUSLIFE", "IPCALAB", "GLAND",
    "MAXHEALTH", "MANKIND", "BIOCON", "LALPATHLAB",
    # FMCG
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA",
    "DABUR", "COLPAL", "MARICO", "TATACONSUM",
    "UNITDSPR", "GODREJCP", "EMAMILTD",
    # Capital Goods / Defence / Infra
    "LT", "SIEMENS", "ABB", "BHEL", "HAL", "BEL",
    "CGPOWER", "CUMMINSIND", "BOSCHLTD", "HAVELLS",
    "VOLTAS", "POLYCAB", "CROMPTON", "THERMAX",
    "RVNL", "IRCTC", "CONCOR",
    # Conglomerate / Diversified
    "ADANIENT", "ADANIPORTS", "TITAN", "GRASIM",
    "PIDILITIND", "ASIANPAINT", "BERGEPAINT",
    # Telecom
    "BHARTIARTL",
    # Real Estate
    "DLF", "LODHA", "GODREJPROP", "OBEROIRLTY",
    "PRESTIGE", "BRIGADE",
    # Consumer Discretionary / Retail
    "TRENT", "DMART", "ETERNAL", "PAGEIND",
    "JUBLFOOD", "INDIGO", "INDHOTEL", "BATAINDIA",
    # Chemicals / Specialty
    "SRF", "PIIND", "DEEPAKNTR", "ATUL", "SOLARINDS",
    "ASTRAL", "SUPREMEIND", "AIAENG",
    # Mid-cap additions
    "BALKRISIND", "GMRINFRA", "FACT", "CESC",
    "ABFRL", "DELHIVERY", "NYKAA",
    "PHOENIXLTD", "KAYNES", "TIINDIA",
    "POONAWALLA", "JKCEMENT", "RAMCOCEM",
    "LAURUSLABS", "METROPOLIS",
    "HINDCOPPER", "KFINTECH", "SONACOMS",
    # Historical Nifty 200 stocks (delisted/dropped/collapsed) -- survivorship bias fix
    "YESBANK", "RBLBANK", "IBULHSGFIN",
    "DHFL", "JPASSOCIAT", "RCOM", "RPOWER",
    "IDEA",
    "JETAIRWAYS",
    "JINDALSAW", "WELCORP",
    "PCJEWELLER", "INFIBEAM", "VAKRANGEE",
    "GRAPHITE", "RAIN", "HEG",
    "NBCC", "NCC", "ENGINERSIN", "PEL",
    "UNITECH", "JPPOWER",
]

SECTOR_MAP: dict[str, str] = {
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "PNB": "Banking", "BANKBARODA": "Banking", "CANBK": "Banking",
    "FEDERALBNK": "Banking", "IDFCFIRSTB": "Banking", "BANDHANBNK": "Banking",
    "AUBANK": "Banking", "IDBI": "Banking", "INDIANB": "Banking",
    "UNIONBANK": "Banking", "CENTRALBK": "Banking", "IOB": "Banking",
    "MAHABANK": "Banking",
    # NBFC / Insurance
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
    "SHRIRAMFIN": "NBFC", "MUTHOOTFIN": "NBFC", "MANAPPURAM": "NBFC",
    "HDFCLIFE": "NBFC", "SBILIFE": "NBFC", "ICICIPRULI": "NBFC",
    "SBICARD": "NBFC", "PFC": "NBFC", "RECLTD": "NBFC",
    "LICHSGFIN": "NBFC", "MFSL": "NBFC", "ABCAPITAL": "NBFC",
    "LICI": "NBFC", "JIOFIN": "NBFC", "IREDA": "NBFC",
    "POONAWALLA": "NBFC", "KFINTECH": "NBFC",
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT", "PERSISTENT": "IT", "COFORGE": "IT",
    "MPHASIS": "IT", "NAUKRI": "IT", "LTTS": "IT", "KPITTECH": "IT",
    "OFSS": "IT", "TATAELXSI": "IT",
    # Energy / Oil & Gas / Power
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "GAIL": "Energy", "HINDPETRO": "Energy",
    "ADANIGREEN": "Energy", "ADANIPOWER": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "TATAPOWER": "Energy", "JSWENERGY": "Energy",
    "NHPC": "Energy", "SJVN": "Energy", "IRFC": "Energy",
    "ATGL": "Energy", "TORNTPOWER": "Energy", "SUZLON": "Energy",
    "CESC": "Energy",
    # Metals & Mining
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "COALINDIA": "Metals", "NMDC": "Metals",
    "SAIL": "Metals", "NATIONALUM": "Metals", "HINDCOPPER": "Metals",
    # Cement
    "ULTRACEMCO": "Cement", "AMBUJACEM": "Cement", "ACC": "Cement",
    "SHREECEM": "Cement", "JKCEMENT": "Cement", "RAMCOCEM": "Cement",
    # Auto
    "MARUTI": "Auto", "M&M": "Auto", "TATAMOTORS": "Auto",
    "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto", "HEROMOTOCO": "Auto",
    "TVSMOTOR": "Auto", "ASHOKLEY": "Auto", "ESCORTS": "Auto",
    "MOTHERSON": "Auto", "SONACOMS": "Auto",
    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma",
    "DIVISLAB": "Pharma", "TORNTPHARM": "Pharma", "LUPIN": "Pharma",
    "AUROPHARMA": "Pharma", "APOLLOHOSP": "Pharma", "ALKEM": "Pharma",
    "ZYDUSLIFE": "Pharma", "IPCALAB": "Pharma", "GLAND": "Pharma",
    "MAXHEALTH": "Pharma", "MANKIND": "Pharma", "BIOCON": "Pharma",
    "LALPATHLAB": "Pharma", "LAURUSLABS": "Pharma", "METROPOLIS": "Pharma",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "COLPAL": "FMCG",
    "MARICO": "FMCG", "TATACONSUM": "FMCG", "UNITDSPR": "FMCG",
    "GODREJCP": "FMCG", "EMAMILTD": "FMCG",
    # Capital Goods / Defence / Infra
    "LT": "CapGoods", "SIEMENS": "CapGoods", "ABB": "CapGoods",
    "BHEL": "CapGoods", "HAL": "CapGoods", "BEL": "CapGoods",
    "CGPOWER": "CapGoods", "CUMMINSIND": "CapGoods", "BOSCHLTD": "CapGoods",
    "HAVELLS": "CapGoods", "VOLTAS": "CapGoods", "POLYCAB": "CapGoods",
    "CROMPTON": "CapGoods", "THERMAX": "CapGoods",
    "RVNL": "CapGoods", "IRCTC": "CapGoods", "CONCOR": "CapGoods",
    "KAYNES": "CapGoods", "TIINDIA": "CapGoods",
    # Conglomerate / Diversified
    "ADANIENT": "Diversified", "ADANIPORTS": "Diversified",
    "TITAN": "Diversified", "GRASIM": "Diversified",
    "PIDILITIND": "Diversified", "ASIANPAINT": "Diversified",
    "BERGEPAINT": "Diversified", "GMRINFRA": "Diversified",
    # Telecom
    "BHARTIARTL": "Telecom",
    # Real Estate
    "DLF": "RealEstate", "LODHA": "RealEstate", "GODREJPROP": "RealEstate",
    "OBEROIRLTY": "RealEstate", "PRESTIGE": "RealEstate",
    "BRIGADE": "RealEstate", "PHOENIXLTD": "RealEstate",
    # Consumer Discretionary / Retail
    "TRENT": "Consumer", "DMART": "Consumer", "ETERNAL": "Consumer",
    "PAGEIND": "Consumer", "JUBLFOOD": "Consumer", "INDIGO": "Consumer",
    "INDHOTEL": "Consumer", "BATAINDIA": "Consumer",
    "ABFRL": "Consumer", "DELHIVERY": "Consumer", "NYKAA": "Consumer",
    # Chemicals / Specialty
    "SRF": "Chemicals", "PIIND": "Chemicals", "DEEPAKNTR": "Chemicals",
    "ATUL": "Chemicals", "SOLARINDS": "Chemicals", "ASTRAL": "Chemicals",
    "SUPREMEIND": "Chemicals", "AIAENG": "Chemicals",
    "BALKRISIND": "Chemicals", "FACT": "Chemicals",
    # Historical stocks -- survivorship bias fix
    "YESBANK": "Banking", "RBLBANK": "Banking",
    "IBULHSGFIN": "NBFC", "DHFL": "NBFC", "PEL": "NBFC",
    "JPASSOCIAT": "CapGoods", "JPPOWER": "Energy",
    "RCOM": "Telecom", "IDEA": "Telecom",
    "RPOWER": "Energy",
    "JETAIRWAYS": "Consumer",
    "JINDALSAW": "Metals", "WELCORP": "Metals",
    "PCJEWELLER": "Consumer", "INFIBEAM": "IT", "VAKRANGEE": "IT",
    "GRAPHITE": "Metals", "RAIN": "Chemicals", "HEG": "Metals",
    "NBCC": "CapGoods", "NCC": "CapGoods", "ENGINERSIN": "CapGoods",
    "UNITECH": "RealEstate",
}

# NSE freeze quantities for major stocks (conservative subset).
# Source: NSE circulars. Stocks not listed here default to 50,000.
FREEZE_QTY: dict[str, int] = {
    "RELIANCE": 1900, "TCS": 1500, "HDFCBANK": 1800, "INFY": 2400,
    "ICICIBANK": 2816, "HINDUNILVR": 1500, "SBIN": 4500, "BHARTIARTL": 2700,
    "ITC": 6400, "KOTAKBANK": 1800, "LT": 1500, "HCLTECH": 2800,
    "AXISBANK": 3600, "MARUTI": 900, "SUNPHARMA": 2500, "BAJFINANCE": 500,
    "TITAN": 1500, "TATAMOTORS": 5631, "ADANIENT": 4000, "NTPC": 7000,
    "WIPRO": 6400, "NESTLEIND": 150, "ULTRACEMCO": 500, "POWERGRID": 9000,
    "M&M": 2100, "TATASTEEL": 6081, "COALINDIA": 7600, "BAJAJFINSV": 500,
    "ONGC": 13200, "JSWSTEEL": 4000, "TECHM": 3600, "DRREDDY": 550,
    "CIPLA": 2600, "DIVISLAB": 850, "EICHERMOT": 700, "ASIANPAINT": 1000,
    "BRITANNIA": 500, "HINDALCO": 5750, "INDUSINDBK": 2400, "HAL": 700,
    "APOLLOHOSP": 500, "SHREECEM": 100, "BAJAJ-AUTO": 500,
    "TRENT": 1500, "ADANIPORTS": 5000, "VEDL": 10000, "GAIL": 12500,
    "BPCL": 5400, "IOC": 9200, "DLF": 4800, "PNB": 13000,
    "HEROMOTOCO": 700, "TVSMOTOR": 1500, "BEL": 8800,
}

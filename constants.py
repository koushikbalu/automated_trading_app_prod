"""Stock universe and sector map for the momentum strategy."""

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

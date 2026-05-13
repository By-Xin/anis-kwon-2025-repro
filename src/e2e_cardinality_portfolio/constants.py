from __future__ import annotations

PAPER_TICKERS = [
    "CVX", "HES", "OXY", "SO",
    "BALL", "ECL", "VMC",
    "FDX", "LMT", "MMM", "RHI", "UPS",
    "AMZN", "AZO", "BBWI", "F", "HAS", "YUM",
    "CPB", "EL", "MKC", "PEP", "PM",
    "A", "ABC", "BIIB", "CVS", "DGX", "JNJ", "SYK",
    "AIG", "BAC", "PGR", "SCHW", "WFC",
    "AAPL", "AKAM", "CRM", "CTSH", "MA", "ORCL",
    "DIS", "EA", "T",
    "AES", "CMS", "DUK",
    "EQR", "PLD", "SPG",
]

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

METHODS = ["nominal", "linreg", "e2e_m", "e2e_socp", "e2e_sdp"]

"""
Price caching layer to avoid repeated yfinance downloads.

Stores all adjusted close data in a single CSV file at `data/yfinance-cache/prices.csv`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yfinance as yf


class PriceCache:
    def __init__(
        self,
        cache_dir: Path | str = Path("data") / "yfinance-cache",
        filename: str = "prices.csv",
    ) -> None:
        self.cache_path = Path(cache_dir) / filename
        self.names_path = Path(cache_dir) / "names.json"
        self.metadata_path = Path(cache_dir) / "metadata.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()
        self._names: Dict[str, str] = self._load_names()
        self._metadata: Dict[str, Dict[str, str]] = self._load_metadata()

    def _load_cache(self) -> pd.DataFrame:
        if self.cache_path.exists():
            df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            if df.columns.duplicated().any():
                df = df.groupby(level=0, axis=1).first()
            df.sort_index(inplace=True)
            return df
        return pd.DataFrame()

    def _save_cache(self) -> None:
        self._cache.sort_index(inplace=True)
        self._cache.to_csv(self.cache_path)

    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        if self.metadata_path.exists():
            try:
                data = json.loads(self.metadata_path.read_text())
                cleaned: Dict[str, Dict[str, str]] = {}
                for ticker, meta in data.items():
                    key = ticker.upper()
                    name = self._clean_name(str(meta.get("name", key)), key)
                    sector = str(meta.get("sector", "Unknown")) or "Unknown"
                    cleaned[key] = {"name": name, "sector": sector}
                self.metadata_path.write_text(json.dumps(cleaned, indent=2))
                return cleaned
            except json.JSONDecodeError:
                return {}
        return {}

    def _load_names(self) -> Dict[str, str]:
        if self.names_path.exists():
            try:
                data = json.loads(self.names_path.read_text())
                cleaned = {}
                for k, v in data.items():
                    key = k.upper()
                    cleaned_val = self._clean_name(str(v), key)
                    cleaned[key] = cleaned_val
                # persist cleaned names back to disk
                self.names_path.write_text(json.dumps(cleaned, indent=2))
                return cleaned
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_names(self) -> None:
        self.names_path.write_text(json.dumps(self._names, indent=2))

    def _save_metadata(self) -> None:
        self.metadata_path.write_text(json.dumps(self._metadata, indent=2))

    @staticmethod
    def _clean_name(name: str, ticker: str) -> str:
        """Simplify company names for readability."""
        raw = name or ticker
        # Remove punctuation that commonly clutters names
        raw = raw.replace(",", " ").replace(".", " ")
        # Remove common suffixes (case-insensitive) at word boundaries
        suffixes = ["incorporated", "inc", "corp", "corporation", "company", "co", "ltd", "llc"]
        pattern = r"\b(" + "|".join(suffixes) + r")\b"
        raw = re.sub(pattern, " ", raw, flags=re.IGNORECASE)
        cleaned = " ".join(raw.split())
        return cleaned.strip() or ticker

    def _needs_fetch(self, ticker: str, start: str, end: str) -> bool:
        if ticker not in self._cache.columns:
            return True
        subset = self._cache.loc[start:end, [ticker]]
        return subset.empty or subset[ticker].isna().any()

    def _fetch_name(self, ticker: str) -> str:
        try:
            info = yf.Ticker(ticker).get_info()
            name = info.get("longName") or info.get("shortName") or ticker
            return self._clean_name(str(name), ticker)
        except Exception:
            return ticker

    def _fetch_metadata(self, ticker: str) -> Dict[str, str]:
        try:
            info = yf.Ticker(ticker).get_info()
        except Exception:
            info = {}
        name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector") or "Unknown"
        return {"name": self._clean_name(str(name), ticker), "sector": str(sector)}

    def _ensure_names(self, tickers: List[str]) -> None:
        updated = False
        for t in tickers:
            key = t.upper()
            if key in self._names:
                continue
            self._names[key] = self._clean_name(self._fetch_name(key), key)
            updated = True
        if updated:
            self._save_names()

    def _ensure_metadata(self, tickers: List[str]) -> None:
        updated = False
        for t in tickers:
            key = t.upper()
            if key in self._metadata:
                continue
            self._metadata[key] = self._fetch_metadata(key)
            updated = True
        if updated:
            self._save_metadata()

    def get_company_names(self, tickers: Iterable[str]) -> Dict[str, str]:
        tickers_list: List[str] = [t.strip().upper() for t in tickers if t.strip()]
        self._ensure_names(tickers_list)
        return {
            t: self._clean_name(self._names.get(t.upper(), t.upper()), t)
            for t in tickers_list
        }

    def get_company_metadata(self, tickers: Iterable[str]) -> Dict[str, Dict[str, str]]:
        tickers_list: List[str] = [t.strip().upper() for t in tickers if t.strip()]
        self._ensure_metadata(tickers_list)
        return {
            t: {
                "name": self._clean_name(
                    self._metadata.get(t.upper(), {}).get("name", t),
                    t,
                ),
                "sector": self._metadata.get(t.upper(), {}).get("sector", "Unknown"),
            }
            for t in tickers_list
        }

    def get_prices(self, tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
        tickers_list: List[str] = [t.strip().upper() for t in tickers if t.strip()]
        if not tickers_list:
            raise ValueError("No tickers provided to PriceCache.get_prices")

        to_fetch = [t for t in tickers_list if self._needs_fetch(t, start, end)]
        if to_fetch:
            raw = yf.download(
                tickers=to_fetch,
                start=start,
                end=end,
                progress=False,
                group_by="column",
                auto_adjust=False,
            )
            if raw.empty:
                raise ValueError("yfinance returned no data; check tickers or date range.")

            # Normalize to DataFrame with tickers as columns, using Adj Close if available, else Close.
            if isinstance(raw, pd.Series):
                data = raw.to_frame(name=to_fetch[0])
            elif isinstance(raw.columns, pd.MultiIndex):
                data = _extract_from_multiindex(raw)
            else:
                field = "Adj Close" if "Adj Close" in raw.columns else "Close" if "Close" in raw.columns else None
                if field is None:
                    raise ValueError("Could not find adjusted or close prices in downloaded data.")
                data = raw[[field]].rename(columns={field: to_fetch[0]})

            # Align indexes to avoid combine_first dtype confusion and ensure monotonic index
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            self._cache.index = pd.to_datetime(self._cache.index)
            self._cache = self._cache.sort_index()

            # Merge: prefer existing cache values, fill gaps with newly fetched data.
            merged_index = self._cache.index.union(data.index)
            data = data.reindex(merged_index)
            self._cache = self._cache.reindex(merged_index)
            for col in data.columns:
                if col in self._cache.columns:
                    self._cache[col] = self._cache[col].combine_first(data[col])
                else:
                    self._cache[col] = data[col]
            self._save_cache()

        self._ensure_names(tickers_list)
        return self._cache.loc[start:end, tickers_list].dropna(how="all")


def _extract_from_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract price columns from a MultiIndex DataFrame returned by yfinance."""
    series_by_ticker: Dict[str, pd.Series] = {}
    for col in raw.columns:
        field = None
        ticker = None
        if isinstance(col, tuple):
            for part in col:
                if isinstance(part, str) and part.lower() in ("adj close", "adjclose", "close"):
                    field = part
                elif isinstance(part, str):
                    ticker = ticker or part
        else:
            continue
        if field and ticker:
            s = raw[col]
            if ticker in series_by_ticker:
                series_by_ticker[ticker] = series_by_ticker[ticker].combine_first(s)
            else:
                series_by_ticker[ticker] = s

    if not series_by_ticker:
        raise ValueError("Could not find adjusted/close price columns in downloaded data (multi-index).")

    data = pd.DataFrame(series_by_ticker)
    return data

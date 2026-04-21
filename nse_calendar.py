"""NSE trading calendar: holiday list and trading-day helpers.

Provides ``is_trading_day``, ``last_trading_day_of_month``, and
``is_last_trading_day`` for holiday-aware scheduling.

The holiday set covers 2024–2027.  Extend ``_RAW_HOLIDAYS`` as NSE
publishes future calendars (typically in December for the next year).
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from functools import lru_cache

_RAW_HOLIDAYS: set[date] = {
    # ── 2024 ──
    date(2024, 1, 26),   # Republic Day
    date(2024, 3, 8),    # Maha Shivaratri
    date(2024, 3, 25),   # Holi
    date(2024, 3, 29),   # Good Friday
    date(2024, 4, 11),   # Id-ul-Fitr (Eid)
    date(2024, 4, 14),   # Dr. Ambedkar Jayanti
    date(2024, 4, 17),   # Ram Navami
    date(2024, 4, 21),   # Mahavir Jayanti
    date(2024, 5, 1),    # Maharashtra Day
    date(2024, 5, 23),   # Buddha Purnima
    date(2024, 6, 17),   # Eid-ul-Adha (Bakri Eid)
    date(2024, 7, 17),   # Muharram
    date(2024, 8, 15),   # Independence Day
    date(2024, 9, 16),   # Milad-un-Nabi
    date(2024, 10, 2),   # Mahatma Gandhi Jayanti
    date(2024, 10, 12),  # Dussehra
    date(2024, 11, 1),   # Diwali (Laxmi Pujan)
    date(2024, 11, 15),  # Guru Nanak Jayanti
    date(2024, 12, 25),  # Christmas

    # ── 2025 ──
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Maha Shivaratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-ul-Fitr (Eid)
    date(2025, 4, 10),   # Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 5, 12),   # Buddha Purnima
    date(2025, 6, 7),    # Eid-ul-Adha (Bakri Eid)
    date(2025, 7, 6),    # Muharram
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 16),   # Janmashtami
    date(2025, 9, 5),    # Milad-un-Nabi
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti / Dussehra
    date(2025, 10, 20),  # Diwali (Laxmi Pujan)
    date(2025, 10, 21),  # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas

    # ── 2026 ──
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 17),   # Maha Shivaratri
    date(2026, 3, 4),    # Holi
    date(2026, 3, 20),   # Id-ul-Fitr (Eid)
    date(2026, 3, 30),   # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day / Buddha Purnima
    date(2026, 5, 27),   # Eid-ul-Adha (Bakri Eid)
    date(2026, 6, 25),   # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 25),   # Milad-un-Nabi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 19),  # Dussehra
    date(2026, 11, 9),   # Diwali (Laxmi Pujan)
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas

    # ── 2027 ──
    date(2027, 1, 26),   # Republic Day
    date(2027, 3, 8),    # Maha Shivaratri
    date(2027, 3, 11),   # Id-ul-Fitr (Eid)
    date(2027, 3, 22),   # Holi
    date(2027, 3, 26),   # Good Friday
    date(2027, 4, 14),   # Dr. Ambedkar Jayanti
    date(2027, 4, 18),   # Ram Navami
    date(2027, 5, 1),    # Maharashtra Day
    date(2027, 5, 17),   # Eid-ul-Adha (Bakri Eid)
    date(2027, 5, 20),   # Buddha Purnima
    date(2027, 8, 15),   # Independence Day
    date(2027, 8, 16),   # Milad-un-Nabi
    date(2027, 10, 2),   # Mahatma Gandhi Jayanti
    date(2027, 10, 8),   # Dussehra
    date(2027, 10, 29),  # Diwali (Laxmi Pujan)
    date(2027, 11, 14),  # Guru Nanak Jayanti
    date(2027, 12, 25),  # Christmas
}


def is_trading_day(d: date) -> bool:
    """True if *d* is a weekday and not an NSE holiday."""
    if d.weekday() >= 5:
        return False
    return d not in _RAW_HOLIDAYS


@lru_cache(maxsize=128)
def last_trading_day_of_month(year: int, month: int) -> date:
    """Return the last NSE trading day of the given month."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def is_last_trading_day(d: date | None = None) -> bool:
    """True if *d* (default: today) is the last trading day of its month."""
    if d is None:
        d = date.today()
    return d == last_trading_day_of_month(d.year, d.month)


def next_trading_day(d: date) -> date:
    """Return the next trading day after *d*."""
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def prev_trading_day(d: date) -> date:
    """Return the trading day immediately before *d*."""
    prev = d - timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    return prev

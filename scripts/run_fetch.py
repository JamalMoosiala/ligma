"""
run_fetch.py

Entry point. One run = one fetch cycle across all three symbols.

Design principles baked in here (per everything discussed building this):
- Skip entirely, cleanly, on weekends/holidays -- no files touched, exit 0.
- Every symbol is fetched and processed independently: one symbol's
  failure (NSE hiccup, parsing error, whatever) is logged and skipped,
  the other symbols still get processed and written.
- fetch_ts is the script's OWN clock, logged on every row -- never trust
  the nominal cron time, cron drift is expected and handled by this.
- IV/Greeks prefer mid_price (bid+ask)/2 over LTP; LTP can be a stale
  trade from hours ago on an illiquid strike, mid_price reflects live
  market-maker quotes even without a trade.
- Bad-quote guardrail: price below intrinsic value -> no IV/Greeks (not
  forced), flagged no_price_available or similar in data_quality_flag.
- Wide-quote flag: a usable price with a very wide bid-ask spread gets
  IV/Greeks computed, but is flagged wide_quote_low_liquidity so you can
  choose whether to trust it downstream.
- Freshness check runs BEFORE writing anything for a symbol.
- Cost-of-carry (and therefore dividend yield) is derived from the
  futures price for the same symbol/expiry, not assumed -- see
  get_cost_of_carry(). Falls back to a static assumption, flagged, when
  futures aren't available.
- Raw JSON (untouched NSE responses) is archived per symbol per day
  regardless of whether the MAIN row-building succeeds, since the raw
  archive is the future-proofing layer and shouldn't be gated on today's
  parsing logic being perfect.
"""

import math
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import greeks
import holidays
import nse_fetch
import vault_io

IST = ZoneInfo("Asia/Kolkata")

SYMBOLS = ["NIFTY", "BANKNIFTY", "NIFTYNXT50"]

RISK_FREE_RATE = 0.065          # static assumption; only affects discounting
                                  # once cost-of-carry is futures-implied
FALLBACK_DIVIDEND_YIELD = 0.0    # used only when futures price unavailable
WIDE_QUOTE_SPREAD_PCT = 0.15     # bid-ask spread / mid_price threshold


def gha_warning(msg: str):
    print(f"::warning::{msg}")


def gha_error(msg: str):
    print(f"::error::{msg}")


def years_to_expiry(expiry_str: str, as_of: datetime) -> float:
    """expiry_str like '25-Sep-2026'. Options stop trading at 15:30 IST."""
    expiry_dt = datetime.strptime(expiry_str, "%d-%b-%Y").replace(
        hour=15, minute=30, tzinfo=IST
    )
    delta = (expiry_dt - as_of).total_seconds()
    return max(delta, 0.0) / (365.0 * 24 * 3600)


def get_dividend_yield_and_carry(futures_by_expiry: dict, expiry_date: str,
                                  S: float, T: float, index_dy_pct):
    """
    Returns (q_used, dividend_yield_source, implied_cost_of_carry, futures_price).

    q_used is sourced from the underlying INDEX'S OWN published dividend
    yield (index_dy_pct, from allIndices' "dy" field) whenever available --
    a slow-moving, NSE-published number. This replaced an earlier design
    that derived q from the futures-basis formula (b = ln(F/S)/T): that
    approach blows up for near-expiry contracts, since dividing by a tiny
    T annualizes even normal basis noise into an extreme rate (confirmed
    live: a 4-day-to-expiry NIFTY strike showed an implied cost-of-carry
    of 20%+ and a dividend yield of -14%, which then distorted every
    Greek). The index's own published yield has no such T-dependency.

    futures_price / implied_cost_of_carry are still computed and returned
    when a futures price is available for this expiry -- purely as
    informational/diagnostic columns now, not used to derive q.
    """
    F = futures_by_expiry.get(expiry_date)
    b = None
    if F and S and T and T > 0:
        b = math.log(F / S) / T

    if index_dy_pct is not None:
        try:
            q_used = float(index_dy_pct) / 100.0
            source = "index_dividend_yield"
        except (TypeError, ValueError):
            q_used = FALLBACK_DIVIDEND_YIELD
            source = "static_fallback"
    else:
        q_used = FALLBACK_DIVIDEND_YIELD
        source = "static_fallback"

    return q_used, source, b, F


def classify_quote(bid, ask, mid) -> str:
    """ok vs wide_quote_low_liquidity, based on relative bid-ask spread."""
    if not bid or not ask or not mid:
        return "ok"  # spread check doesn't apply if we don't have both sides
    spread_pct = (ask - bid) / mid if mid else None
    if spread_pct is not None and spread_pct > WIDE_QUOTE_SPREAD_PCT:
        return "wide_quote_low_liquidity"
    return "ok"


def _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, v3_raw,
                     index_snapshot_raw, status):
    return {
        "fetch_ts_utc": fetch_ts_utc.isoformat(),
        "fetch_ts_ist": fetch_ts_ist.isoformat(),
        "symbol": symbol,
        "responses": {
            "option_chain_bootstrap": bootstrap_raw,
            "option_chain_v3": v3_raw,
            "index_snapshot": index_snapshot_raw,
        },
        "fetch_status": status,
    }


def process_symbol(symbol: str, fetch_ts_utc: datetime, fetch_ts_ist: datetime,
                    index_snapshot_raw: dict):
    """
    Fetches and processes one symbol end to end. Returns (rows, snapshot).
    Raises only on truly unexpected errors the caller should log and move
    past -- most expected failure modes are handled internally and
    reflected in per-row flags / status, not exceptions.

    Two-step fetch (see nse_fetch.py's option-chain-v3 section for why):
      1. Bootstrap via fetch_option_chain() (the flat endpoint) -- gives a
         guaranteed-valid expiry string and futures/cost-of-carry data.
      2. Call option-chain-v3 with that expiry to discover the FULL expiry
         list, pick the nearest MONTHLY expiry from it (per project
         requirements), and fetch that expiry's real chain -- which,
         unlike the flat endpoint, includes actual bid/ask and NSE's own
         published IV.
    """
    status = {
        "option_chain": "ok", "futures": "ok", "lot_size": "static_fallback_only",
        "option_chain_v3": "ok",
    }

    # --- Step 1: bootstrap (flat endpoint) -- futures + a valid expiry ---
    bootstrap_raw = nse_fetch.fetch_option_chain(symbol)
    bootstrap_entries = (bootstrap_raw or {}).get("data", [])

    if not bootstrap_entries:
        status["option_chain"] = "empty_response"
        gha_warning(f"[{symbol}] bootstrap option chain had no entries -- "
                    f"cannot discover a valid expiry, skipping this symbol this cycle.")
        return [], _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, None, index_snapshot_raw, status)

    futures_by_expiry = nse_fetch.parse_futures_from_entries(bootstrap_entries)
    if not futures_by_expiry:
        status["futures"] = "no_futures_entries_in_response"
        gha_warning(f"[{symbol}] no futures entries found in the bootstrap response; "
                    f"dividend yield will use static_fallback for all rows this cycle.")

    bootstrap_expiry = next(
        (e.get("expiryDate") for e in bootstrap_entries if e.get("instrumentType", "").startswith("OPT")),
        None,
    )
    if not bootstrap_expiry:
        status["option_chain_v3"] = "no_bootstrap_expiry_found"
        gha_warning(f"[{symbol}] could not find any option expiry in the bootstrap "
                    f"response to seed option-chain-v3 -- skipping this symbol this cycle.")
        return [], _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, None, index_snapshot_raw, status)

    # --- Step 2: option-chain-v3 -- discover all expiries, pick nearest monthly ---
    v3_bootstrap = nse_fetch.fetch_option_chain_v3(symbol, bootstrap_expiry)
    all_expiries = v3_bootstrap.get("records", {}).get("expiryDates", [])

    if not all_expiries:
        status["option_chain_v3"] = "empty_response"
        gha_warning(f"[{symbol}] option-chain-v3 returned no expiry list even with a "
                    f"known-valid bootstrap expiry ({bootstrap_expiry}) -- skipping this "
                    f"symbol this cycle. Check whether NSE changed this endpoint again.")
        return [], _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, v3_bootstrap, index_snapshot_raw, status)

    target_expiry = nse_fetch.pick_nearest_monthly_expiry(all_expiries, fetch_ts_ist.date())

    if target_expiry == bootstrap_expiry:
        v3_target = v3_bootstrap  # already have it, no extra call needed
    else:
        v3_target = nse_fetch.fetch_option_chain_v3(symbol, target_expiry)

    flat_legs = nse_fetch.flatten_v3_entries(v3_target)
    if not flat_legs:
        status["option_chain_v3"] = "empty_target_expiry_response"
        gha_warning(f"[{symbol}] option-chain-v3 returned no CE/PE legs for target "
                    f"expiry {target_expiry} -- this symbol will fail its freshness "
                    f"check this cycle.")

    lot_size = nse_fetch.LOT_SIZE_FALLBACK.get(symbol)  # no live source exists anymore

    idx_ohlc = nse_fetch.parse_index_snapshot(index_snapshot_raw, symbol) if index_snapshot_raw else {}
    india_vix_row = nse_fetch.parse_india_vix(index_snapshot_raw) if index_snapshot_raw else {}
    india_vix = india_vix_row.get("last")

    rows = []
    for leg in flat_legs:
        opt_type = leg.get("option_type")
        expiry_date = leg.get("expiryDates")  # consistent DD-Mon-YYYY format
        raw_strike = leg.get("strikePrice")
        try:
            strike = float(raw_strike)
        except (TypeError, ValueError):
            continue

        T = years_to_expiry(expiry_date, fetch_ts_ist) if expiry_date else None
        entry_underlying = leg.get("underlyingValue")

        bid = leg.get("buyPrice1") or 0.0
        bid_qty = leg.get("buyQuantity1")
        ask = leg.get("sellPrice1") or 0.0
        ask_qty = leg.get("sellQuantity1")
        mid = (bid + ask) / 2 if (bid and ask) else None
        ltp = leg.get("lastPrice") or 0.0
        nse_iv = leg.get("impliedVolatility")
        pchange = leg.get("pChange")

        row = {c: None for c in vault_io.CSV_COLUMNS}
        row.update({
            "fetch_ts_utc": fetch_ts_utc.isoformat(),
            "fetch_ts_ist": fetch_ts_ist.isoformat(),
            "symbol": symbol,
            "expiry_date": expiry_date,
            "strike": strike,
            "option_type": opt_type,
            "underlying_value": entry_underlying,
            "bid_price": bid or None,
            "bid_qty": bid_qty,
            "ask_price": ask or None,
            "ask_qty": ask_qty,
            "ltp": ltp,
            "mid_price": mid,
            "open_interest": leg.get("openInterest"),
            "change_in_oi": leg.get("changeinOpenInterest"),
            "total_traded_volume": leg.get("totalTradedVolume"),
            "pchange_vs_prev_close": pchange,
            "nse_iv": nse_iv,
            "time_to_expiry_years": T,
            "india_vix": india_vix,
            "lot_size": lot_size,
            "underlying_day_open": idx_ohlc.get("open"),
            "underlying_day_high": idx_ohlc.get("high"),
            "underlying_day_low": idx_ohlc.get("low"),
            "underlying_prev_close": idx_ohlc.get("prev_close"),
        })

        # Real bid/ask are back -- prefer mid-price over LTP again, exactly
        # like the original design intended. Used here purely for the
        # below-intrinsic-value guardrail now (see below), since we no
        # longer solve our own IV from it -- Greeks use nse_iv directly.
        price_for_iv = mid if mid else (ltp if ltp else None)
        row["price_source_for_iv"] = "mid_price" if mid else ("ltp" if ltp else "none")

        if not price_for_iv or not entry_underlying or not strike or not T or T <= 0:
            row["data_quality_flag"] = "no_price_available"
            rows.append(row)
            continue

        # Guardrail, preserved even without our own solver: a price below
        # intrinsic value is a strong sign of a stale/bad quote, regardless
        # of what IV NSE has published for it.
        intrinsic = max(entry_underlying - strike, 0.0) if opt_type == "CE" else max(strike - entry_underlying, 0.0)
        if price_for_iv < intrinsic - 1e-6:
            row["data_quality_flag"] = "no_price_available"
            rows.append(row)
            continue

        q_used, carry_source, b, F = get_dividend_yield_and_carry(
            futures_by_expiry, expiry_date, entry_underlying, T, idx_ohlc.get("dy"),
        )
        row["futures_price"] = F
        row["implied_cost_of_carry"] = b
        row["dividend_yield_used"] = q_used
        row["dividend_yield_source"] = carry_source
        row["risk_free_rate_used"] = RISK_FREE_RATE

        # Greeks now use NSE's own published IV directly -- no in-house
        # solver in the loop. nse_iv arrives as a percentage (e.g. 8.8
        # means 8.8%), so it's converted to decimal here.
        sigma = None
        if nse_iv is not None:
            try:
                candidate = float(nse_iv) / 100.0
                if candidate > 0:
                    sigma = candidate
            except (TypeError, ValueError):
                sigma = None

        if sigma is None:
            row["data_quality_flag"] = "no_nse_iv"
            rows.append(row)
            continue

        g = greeks.compute_all_greeks(
            entry_underlying, strike, T, RISK_FREE_RATE, q_used, sigma, opt_type,
        )
        row.update(g)
        row["data_quality_flag"] = classify_quote(bid or None, ask or None, mid)

        rows.append(row)

    snapshot = _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, v3_target, index_snapshot_raw, status)
    return rows, snapshot


def main():
    run_start = time.monotonic()
    fetch_ts_utc = datetime.now(timezone.utc)
    fetch_ts_ist = fetch_ts_utc.astimezone(IST)
    today = fetch_ts_ist.date()

    if not holidays.is_trading_day(today):
        print(f"{today.isoformat()} is not a trading day (weekend or holiday). "
              f"Skipping run cleanly -- no files touched.")
        sys.exit(0)

    print(f"Fetch cycle started at {fetch_ts_ist.isoformat()} (IST)")

    # One call, reused across all three symbols (day OHLC + India VIX).
    # If this fails, we don't abort the whole run -- OHLC/VIX columns will
    # just be null for this cycle, everything else still gets fetched.
    index_snapshot_raw = None
    try:
        index_snapshot_raw = nse_fetch.fetch_index_snapshot_raw()
    except Exception as exc:  # noqa: BLE001
        gha_warning(f"Index snapshot (OHLC + India VIX) fetch failed after retries "
                    f"({exc}); those columns will be null for this cycle.")

    any_symbol_succeeded = False

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            rows, snapshot = process_symbol(symbol, fetch_ts_utc, fetch_ts_ist, index_snapshot_raw)
        except Exception as exc:  # noqa: BLE001
            gha_error(f"[{symbol}] unexpected failure, skipping this symbol this cycle: {exc}")
            continue

        underlying_value = rows[0]["underlying_value"] if rows else None
        if not vault_io.freshness_ok(symbol, underlying_value, rows):
            gha_warning(f"[{symbol}] failed freshness check -- not writing MAIN rows "
                        f"for this symbol this cycle (raw archive still saved for audit).")
            # Still archive the raw response even on freshness failure --
            # useful for debugging why NSE returned something unusable.
            try:
                vault_io.append_raw_snapshot(symbol, today, snapshot)
            except Exception as exc:  # noqa: BLE001
                gha_warning(f"[{symbol}] could not write raw archive either: {exc}")
            continue

        try:
            vault_io.append_raw_snapshot(symbol, today, snapshot)
        except Exception as exc:  # noqa: BLE001
            gha_warning(f"[{symbol}] raw archive write failed: {exc} (MAIN rows still kept)")

        # Each symbol gets its OWN MAIN file (vault/tables/<SYMBOL>/MAIN-*.csv)
        # -- written immediately per symbol, not accumulated and merged
        # across symbols into one shared file.
        vault_io.append_main_rows(symbol, today, rows)
        any_symbol_succeeded = True
        print(f"  [{symbol}] {len(rows)} rows written to its own MAIN table.")

    if not any_symbol_succeeded:
        gha_error("No symbol produced usable data this cycle. Check NSE endpoint "
                  "health and the warnings/errors above.")
        # Exit 0 anyway -- a single bad cycle shouldn't fail the whole
        # scheduled workflow (retries + the next cycle will likely recover).
        # The ::error:: annotation above still makes this visible in the
        # Actions run summary.

    total_elapsed = time.monotonic() - run_start
    print(f"Fetch cycle finished in {total_elapsed:.1f}s total.")
    sys.exit(0)


if __name__ == "__main__":
    main()

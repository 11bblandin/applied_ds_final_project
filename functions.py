import pandas as pd


def resample_ohlc(ohlc, freq="D"):
    resample_df = (
        pd.DataFrame()
        .assign(open=ohlc["open"].resample(freq).first())
        .assign(high=ohlc["high"].resample(freq).max())
        .assign(low=ohlc["low"].resample(freq).min())
        .assign(close=ohlc["close"].resample(freq).last())
        .assign(volume=ohlc["volume"].resample(freq).sum())
    )
    if "funding" in ohlc.columns:
        resample_df["funding"] = ohlc["funding"].add(1).resample(freq).prod().sub(1)

    if "buy_volume" in ohlc.columns:
        resample_df["buy_volume"] = ohlc["buy_volume"].resample(freq).sum()

    return resample_df

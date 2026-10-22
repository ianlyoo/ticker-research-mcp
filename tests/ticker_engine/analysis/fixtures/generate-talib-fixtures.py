"""Test-time only. Run in a separate virtual environment with TA-Lib and numpy."""
import hashlib, json, pathlib
import numpy as np
import talib

root = pathlib.Path(__file__).resolve().parents[1]
def clean(values):
    return [float(v) if np.isfinite(v) else None for v in values]
result = {'library':'TA-Lib','python_version':talib.__version__,'c_version':talib.__ta_version__.decode(),'symbols':{}}
for symbol in ['AAPL','NVDA','MSFT','TSLA','005930']:
    path = root / 'fixtures' / (symbol+'.json')
    rows = json.loads(path.read_text(encoding='utf8'))['candles']
    close, high, low = [np.array([float(r[key]) for r in rows], dtype=np.float64) for key in ['closePrice','highPrice','lowPrice']]
    # Our MACD contract uses independently seeded EMA legs. TA-Lib MACD itself
    # starts the fast EMA later; record both and label the seed difference.
    fast, slow = talib.EMA(close,12),talib.EMA(close,26)
    line = fast-slow
    signal = np.full(len(close),np.nan)
    signal[25:] = talib.EMA(line[25:],9)
    upper, mid, lower = talib.BBANDS(close,20,2,2)
    raw_line, raw_signal, raw_hist = talib.MACD(close,12,26,9)
    values = {'sma20':talib.SMA(close,20),'ema20':talib.EMA(close,20),'rsi14':talib.RSI(close,14),'atr14':talib.ATR(high,low,close,14),'stdev20':talib.STDDEV(close,20),'highest20':talib.MAX(close,20),'lowest20':talib.MIN(close,20),'roc20':talib.ROC(close,20),'macd_line':line,'macd_signal':signal,'macd_hist':line-signal,'bb_mid':mid,'bb_upper':upper,'bb_lower':lower,'raw_macd_line':raw_line,'raw_macd_signal':raw_signal,'raw_macd_hist':raw_hist}
    result['symbols'][symbol] = {'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'series':{k:clean(v) for k,v in values.items()}}
out = root / 'fixtures' / 'talib-reference.json'
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({'fixture':out.name,'symbols':len(result['symbols']),'series_per_symbol':len(values),'TA-Lib':talib.__version__}))
# Kept unformatted, so excluded from ruff:
# ruff: noqa

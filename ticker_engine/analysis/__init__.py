"""Pure calculation package for ticker-research analysis tools (stdlib only, no IO).

series: multi-symbol alignment, normalised/return/relative-strength/excess
series, drawdown, volatility and currency conversion. indicators: the
built-in daily indicators. discontinuity: possible-split warnings.
eps: historical P/E and EPS scenarios.
"""
from .discontinuity import possible_splits
from .eps import (
    EpsActual,
    EpsEstimate,
    fiscal_elapsed,
    historical_pe,
    ntm_eps,
    scenarios,
)
from .indicators import REGISTRY, IndicatorDef, ParamDef, evaluate
from .series import (
    Aligned,
    Closes,
    align,
    annualized_volatility_pct,
    convert_currency,
    excess_return_pp,
    max_drawdown_pct,
    normalized,
    price_return_pct,
    relative_strength,
)

__all__ = [
    "Aligned", "Closes", "align", "normalized", "price_return_pct", "relative_strength", "excess_return_pp",
    "max_drawdown_pct", "annualized_volatility_pct", "convert_currency", "possible_splits",
    "REGISTRY", "IndicatorDef", "ParamDef", "evaluate", "EpsActual", "EpsEstimate", "historical_pe", "scenarios",
    "fiscal_elapsed", "ntm_eps",
]

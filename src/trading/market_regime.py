import numpy as np

def detect_market_regime(

    prices,

    volatility_threshold=0.1
):

    volatility = np.std(prices)

    if volatility > volatility_threshold:

        return "HIGH VOLATILITY"

    else:

        return "NORMAL VOLATILITY"
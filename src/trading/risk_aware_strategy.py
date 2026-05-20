def risk_adjusted_signal(

    trading_signal,

    uncertainty,

    max_uncertainty=0.15
):

    if uncertainty > max_uncertainty:

        return "AVOID TRADE"

    return trading_signal
import numpy as np

from src.trading.signal_generator import (
    generate_trading_signal
)

from src.trading.risk_aware_strategy import (
    risk_adjusted_signal
)

from src.trading.market_regime import (
    detect_market_regime
)

def run_trading_inference():

    current_price = 5.2

    predicted_price = 5.6

    uncertainty = 0.08

    signal = generate_trading_signal(

        predicted_price,

        current_price
    )

    adjusted_signal = (
        risk_adjusted_signal(

            signal,

            uncertainty
        )
    )

    regime = detect_market_regime(

        np.random.normal(

            5,

            uncertainty,

            100
        )
    )

    print("\nTRADING INTELLIGENCE\n")

    print(
        "Current Price:",
        current_price
    )

    print(
        "Predicted Price:",
        predicted_price
    )

    print(
        "Uncertainty:",
        uncertainty
    )

    print(
        "Market Regime:",
        regime
    )

    print(
        "Trading Signal:",
        adjusted_signal
    )

if __name__ == "__main__":

    run_trading_inference()
import numpy as np

def simulate_strategy(

    actual_prices,

    predicted_prices
):

    capital = 10000

    trade_log = []

    for i in range(

        len(actual_prices) - 1
    ):

        current_price = actual_prices[i]

        predicted_price = (
            predicted_prices[i]
        )

        if predicted_price > current_price:

            profit = (
                actual_prices[i + 1]
                -
                current_price
            )

            capital += profit

            trade_log.append(profit)

    return {

        "final_capital": capital,

        "total_return":
            capital - 10000,

        "number_of_trades":
            len(trade_log),

        "average_profit":
            np.mean(trade_log)
            if trade_log else 0
    }
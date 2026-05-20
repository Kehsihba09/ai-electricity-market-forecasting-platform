import numpy as np

def generate_trading_signal(

    predicted_price,

    current_price,

    threshold=0.02
):

    percentage_change = (

        predicted_price
        -
        current_price

    ) / current_price

    if percentage_change > threshold:

        return "BUY"

    elif percentage_change < -threshold:

        return "SELL"

    else:

        return "HOLD"
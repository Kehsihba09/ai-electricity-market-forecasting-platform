def ensemble_decision(

    predictions,

    threshold=0.02
):

    avg_prediction = (
        sum(predictions)
        / len(predictions)
    )

    bullish_votes = sum(

        pred > avg_prediction

        for pred in predictions
    )

    bearish_votes = sum(

        pred < avg_prediction

        for pred in predictions
    )

    if bullish_votes > bearish_votes:

        return "BUY"

    elif bearish_votes > bullish_votes:

        return "SELL"

    else:

        return "HOLD"
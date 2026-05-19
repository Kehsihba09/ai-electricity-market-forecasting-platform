import pandas as pd

def prepare_features(request):

    df = pd.DataFrame([{

        "Purchase Bid (MW)":
            request.purchase_bid,

        "Sell Bid (MW)":
            request.sell_bid,

        "Final Scheduled Volume (MW)":
            request.final_scheduled_volume,

        "hour":
            request.hour
    }])

    return df
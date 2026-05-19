import pandas as pd
from datetime import datetime

def prepare_features(request):

    current_time = datetime.now()

    df = pd.DataFrame([{

        "Purchase Bid (MW)":
            request.purchase_bid,

        "Sell Bid (MW)":
            request.sell_bid,

        "MCV (MW)":
            (
                request.purchase_bid
                - request.sell_bid
            ),

        "Final Scheduled Volume (MW)":
            request.final_scheduled_volume,

        "hour":
            request.hour,

        "day_of_week":
            current_time.weekday(),

        "month":
            current_time.month,

        "is_weekend":
            int(current_time.weekday() >= 5)
    }])

    return df

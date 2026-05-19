import pandas as pd
from datetime import datetime

def prepare_features(request):

    current_time = datetime.now()

    df = pd.DataFrame([{

        "purchase_bid":
            request.purchase_bid,

        "sell_bid":
            request.sell_bid,

        "mcv":
            (
                request.purchase_bid
                - request.sell_bid
            ),

        "final_scheduled_volume":
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

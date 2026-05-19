from pydantic import BaseModel

class ForecastRequest(BaseModel):

    purchase_bid: float

    sell_bid: float

    final_scheduled_volume: float

    hour: int

class ForecastResponse(BaseModel):

    forecast_price: float

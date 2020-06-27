library(forecast)
class(AirPassengers)
AirPassengers
arima = auto.arima(AirPassengers)
arima
previsao = forecast(arima, h=12)
previsao
plot(previsao)

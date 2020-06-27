library(forecast)

dados=read.csv("sem10-20.csv")
head(dados)
class(dados)
dados

dadosTS=ts(dados)
head(dadosTS)
class(dadosTS)
arima = auto.arima(dadosTS)
arima
previsao = forecast(arima, h=12)
previsao
plot(previsao)


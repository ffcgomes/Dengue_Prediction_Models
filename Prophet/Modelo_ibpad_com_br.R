# Prophet vs forecast vs mafs: Qual o melhor pacote para séries temporais?
# Neste post, falaremos sobre: Os pacotes prophet, forecast e mafs, três pacotes para previsão de Séries temporais e o pacote cranlogs é para baixar os dados de downloads do CRAN.

library(mafs)
library(prophet)
library(cranlogs)
library(tidyverse)
library(lubridate)


# Coleta dos dados
#Vamos definir os parâmetros de data de nossa query:
  
data_inicio <- as.Date("2015-09-30")
data_fim <- as.Date("2017-09-30")
df_dls <- cran_downloads(packages = "forecast", from = data_inicio, to = data_fim)

knitr::kable(head(df_dls))

#Vemos que o dataframe df_dls possui três colunas: a primeira indica a data, a segunda a quantidade de downloads do pacote naquele dia e a terceira a qual pacote os dados se referem.

#Primeiramente, será que tem algum buraco nos dados? Vamos fazer uma verificação:

vetor_datas <- seq.Date(from = min(df_dls$date), to = max(df_dls$date), by = "1 day")
length(vetor_datas) == nrow(df_dls)

#O TRUE acima indica que não temos nenhum buraco nos dados. Isto é, caso haja algum dia onde ninguém baixou o forecast, o dado informado será 0 ao invés de NA.

#A melhor maneira de visualizar os dados que temos é por meio de um gráfico de linha do ggplot2:

ggplot(df_dls, aes(x = date, y = count)) +
  geom_line() +
  theme_minimal() + 
  labs(x = NULL, y = NULL,
       title = "Quantidade de downloads diários do pacote forecast") +
  scale_x_date(date_labels = "%m/%Y", date_breaks = "3 months")

#Existem alguns outliers na série. Como além de ser difícil prever esses picos é improvável que eles aconteçam novamente, vamos os retirar da série:
df_dls <- df_dls %>% filter(date >= as.Date("2017-02-01"))

#Obtendo previsões para a série
#Para este post, vamos simular que o objetivo é prever o mês de Setembro da série, usando o restante como conjunto de treino.

# definir conjuntos de treino e teste

data_treino <- as.Date("2017-09-01")

#Prophet
#A função de ajuste de modelo prophet::prophet() exige que o data frame de input possua duas colunas: uma chamada ds, com o vetor de datas, e uma chamada y, com o vetor numérico da variável que se deseja prever. Aliás, uma crítica pessoal minha ao prophet é a de eles usarem dataframes como objetos de input, e não objetos do tipo ts, que é o normal no R para séries temporais.

df_dls <- df_dls %>% select(ds = date, y = count)
df_treino <- df_dls %>% filter(ds < data_treino)
df_teste <- df_dls %>% filter(ds >= data_treino)
nn <- nrow(df_teste)

#As principais funções do prophet são mostradas abaixo:

# fitar modelo prophet
mod_prophet <- prophet(df_treino)
## Initial log joint probability = -2.90115
## Optimization terminated normally: 
##   Convergence detected: absolute parameter change was below tolerance
fcast_prophet <- predict(mod_prophet,
                         make_future_dataframe(mod_prophet, periods = nn))

#É possível visualizar as previsões fornecidas pelo prophet:
plot(mod_prophet, fcast_prophet)

#A tabela abaixo mostra uma pequena parte do dataframe de output:
knitr::kable(head(fcast_prophet))

#Vemos que o dataframe resultante é bem verboso, possuindo 16 colunas. Para este post, precisamos apenas da coluna yhat, que se refere à previsão obtida pelo prophet, além da coluna de data.
# retornar previsoes
fcast_prophet <- fcast_prophet %>% 
  filter(ds >= data_treino) %>% 
  select(ds, yhat) %>% 
  mutate(ds = as.Date(ds), yhat = round(yhat))

# transformar em objeto ts
ts_dls <- ts(df_treino$y, start = lubridate::decimal_date(data_inicio),
             frequency = 365)

#Assim, já podemos obter os modelos com o mafs. Nos testes que eu fiz, os modelos StructTS (modelo estrutural) e tslm (modelo de regressão que usa a tendência e a sazonalidade como regressores) não funcionam nuito bem para séries diárias (o StructTS demora uma eternidade para rodar para séries diárias).

modelo_mafs <- select_forecast(ts_dls, test_size = nn, horizon = nn,
                               error = "MAPE", verbose = TRUE,
                               dont_apply = c("StructTS", "tslm"))
prev_mafs <- round(modelo_mafs$best_forecast$mean)

#Vemos que alguns dos modelos aplicados pelo mafs produziram alguma mensagem de aviso ou não puderam ser obtidos. De fato, o dataframe modelo_mafs$df_models retorna apenas 13 modelos:
knitr::kable(modelo_mafs$df_models)

#Vamos então obter a previsão futura produzida pelo mafs e a juntar com a previsão do prophet no dataframe de teste:
prev_mafs <- round(modelo_mafs$best_forecast$mean)
fcast_prophet$yhat_mafs <- as.numeric(prev_mafs)
# mudar nome das colunas
names(fcast_prophet) <- c("ds", "previsao_prophet", "previsao_mafs")
# juntar dataframe de resultado com o de previsao
df_teste <- df_teste %>%  left_join(fcast_prophet, by = "ds")


#Vamos então obter a previsão futura produzida pelo mafs e a juntar com a previsão do prophet no dataframe de teste:
prev_mafs <- round(modelo_mafs$best_forecast$mean)
fcast_prophet$yhat_mafs <- as.numeric(prev_mafs)
# mudar nome das colunas
names(fcast_prophet) <- c("ds", "previsao_prophet", "previsao_mafs")
# juntar dataframe de resultado com o de previsao
df_teste <- df_teste %>%  left_join(fcast_prophet, by = "ds")

# plotar previsoes vs resultados reais
df_teste %>% 
  gather(metodo, previsao, -(1:2)) %>% 
  ggplot(aes(x = ds, y = y)) + 
  geom_line() + 
  geom_line(aes(y = previsao, color = metodo))

#O mafs produziu uma previsão de linha reta. Apenas como forma de demonstrar o uso do meu pacote, vamos remover o modelo ets da lista de modelos usados e rever os resultados:

modelo_mafs <- select_forecast(ts_dls, test_size = nn, horizon = nn,
                               error = "MAPE", verbose = FALSE,
                               dont_apply = c("StructTS", "ets", "tslm"))

prev_mafs <- round(modelo_mafs$best_forecast$mean)
fcast_prophet$previsao_mafs <- as.numeric(prev_mafs)
# mudar nome das colunas
names(fcast_prophet) <- c("ds", "previsao_prophet", "previsao_mafs")
# juntar dataframe de resultado com o de previsao
df_teste <- df_dls %>% filter(ds >= data_treino)
df_teste <- df_teste %>%  left_join(fcast_prophet, by = "ds")
df_teste %>% 
  gather(metodo, previsao, -(1:2)) %>% 
  ggplot(aes(x = ds, y = y)) + 
  geom_line() + 
  geom_line(aes(y = previsao, color = metodo))

#Por mais incrível que pareça, mais uma vez uma linha reta foi fornecida como previsão pelo mafs, enquanto o prophet conseguiu prever com muita eficácia a sazonalidade da série.

#Numericamente, o erro médio absoluto de downloads é de:

real <- df_teste$y
prev_prophet <- df_teste$previsao_prophet
prev_mafs <- df_teste$previsao_mafs
mean(abs(real - prev_mafs))
## [1] 1010.867
mean(abs(real - prev_prophet))
## [1] 837.0333

#Considerações finais
#Sobre o título (meio sensacionalista) do post: Em meus estudos sobre séries temporais, é comum encontrar livros e papers afirmando que é impossível determinar que o modelo X sempre será melhor que Y. Cada série temporal possui suas próprias características: sazonalidade, outliers, ciclos de negócios, tendência, frequência, etc. O recomendável é estudar a teoria de cada modelo que se deseja usar, variar seus parâmetros e pesquisar em artigos benchmarks para séries temporais de um determinado contexto (por exemplo, para vendas de produtos de demanda intermitente costuma-se usar Croston).

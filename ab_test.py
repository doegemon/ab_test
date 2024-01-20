# Databricks notebook source
# MAGIC %md
# MAGIC # Teste A/B

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Bibliotecas e Funções Auxiliares

# COMMAND ----------

import math
import numpy as np
import pandas as pd
import statsmodels.stats.power as smpow
import statsmodels.stats.proportion as smpro

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregando os Dados

# COMMAND ----------

df_raw = pd.read_csv( '/Workspace/Repos/twitch.k3a99@passinbox.com/ab_test/data/ab_testing.csv' )
df_raw.sample(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Design do Experimento

# COMMAND ----------

# MAGIC %md
# MAGIC ### Formulação das Hipóteses

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros do Experimento

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Nível de Confiança e Significância

# COMMAND ----------

# DBTITLE 0,Nível de Confiança e Significância
confidence_lvl = 0.95

significance_lvl = 1 - confidence_lvl

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tamanho do Efeito

# COMMAND ----------

# Conversão da página atual 
m1 = 0.13

# Conversão esperada da nova página
m2 = 0.15

# COMMAND ----------

effect_size = smpro.proportion_effectsize( m1, m2 )
print( effect_size )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Poder Estatístico

# COMMAND ----------

stat_power = 0.80

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tamanho da Amostra

# COMMAND ----------

sample_n = math.ceil ( smpow.NormalIndPower().solve_power(
    effect_size=effect_size,
    power = stat_power, 
    alpha = significance_lvl
) )

# COMMAND ----------

print( f'Tamanho da amostra de cada grupo - Controle e Tratamento: { sample_n }' )
print( f'Tamanho total da amostra: { sample_n * 2 }' )

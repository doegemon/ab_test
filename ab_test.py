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
from scipy.stats import chi2_contingency

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregando os Dados

# COMMAND ----------

df_raw = pd.read_csv( '/Workspace/Repos/twitch.k3a99@passinbox.com/ab_test/data/ab_data.csv' )
df_raw.sample(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Design do Experimento

# COMMAND ----------

# MAGIC %md
# MAGIC ### Formulação das Hipóteses

# COMMAND ----------

# MAGIC %md
# MAGIC - H0 = A conversão da nova página é de 13%
# MAGIC - H1 = A conversão da nova página é diferente de 13%

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Preparação dos Dados

# COMMAND ----------

df1 = df_raw.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise Descritiva dos Dados e Valores Faltantes

# COMMAND ----------

df1.head()

# COMMAND ----------

print( f'Número de linhas: {df1.shape[0]}' )
print( f'Número de colunas: {df1.shape[1]}' ) 

# COMMAND ----------

df1.isna().sum()

# COMMAND ----------

df1 = df1.dropna()

# COMMAND ----------

df1['timestamp'] = pd.to_datetime(df1['timestamp']).apply(lambda x: x.strftime('%Y-%m-%d'))
df1 = df1.rename(columns={'timestamp': 'session_dt'})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conferindo as _flags_ dos Grupos

# COMMAND ----------

df1.head()

# COMMAND ----------

df1[['user_id', 'group', 'landing_page']].groupby( ['group', 'landing_page'] ).count().reset_index()

# COMMAND ----------

df1 = df1.drop_duplicates( subset='user_id' )

df1[['user_id', 'group', 'landing_page']].groupby( ['group', 'landing_page'] ).count().reset_index()

# COMMAND ----------

df1['check'] = df1.apply( lambda x: 'right' if ( x['group'] == 'control' and x['landing_page'] == 'old_page' ) 
                                    or ( x['group'] == 'treatment' and x['landing_page'] == 'new_page' ) 
                                    else 'wrong', axis=1 )

df1.head()

# COMMAND ----------

df_aux = df1.loc[df1['check'] == 'right']

df_aux[['user_id', 'group', 'landing_page']].groupby( ['group', 'landing_page'] ).count().reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Amostragem Aleatória dos Grupos

# COMMAND ----------

df2 = df_aux.drop( 'check', axis=1 )
df2.head()

# COMMAND ----------

df_control = df2[df2['group'] == 'control' ].sample( n = sample_n, random_state=42 )
print( f'Tamanho da amostra do Grupo de Controle: {df_control.shape[0]}' )

df_treatment = df2[df2['group'] == 'treatment' ].sample( n = sample_n, random_state=42 )
print( f'Tamanho da amostra do Grupo de Tratamento: {df_treatment.shape[0]}' )

# COMMAND ----------

df_ab = pd.concat( [df_control, df_treatment] ).reset_index( drop=True )
df_ab.sample( 10 )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conversão de cada Grupo

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grupo de Controle

# COMMAND ----------

control_sales = df_control.loc[df_control['converted'] == 1, 'converted'].sum()
control_visits = len( df_control )

conversion_rate_control = np.divide( control_sales, control_visits )
print( f'Conversão - Grupo de Controle: {conversion_rate_control}' )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grupo de Tratamento

# COMMAND ----------

treatment_sales = df_treatment.loc[df_treatment['converted'] == 1, 'converted'].sum()
treatment_visits = len( df_treatment )

conversion_rate_treatment = np.divide( treatment_sales, treatment_visits )
print( f'Conversão - Grupo de Tratamento: {conversion_rate_treatment}' )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Controle _vs._ Tratamento

# COMMAND ----------

conversion_rate_control
conversion_rate_treatment

df_vs = pd.DataFrame( {'Controle': conversion_rate_control, 'Tratamento': conversion_rate_treatment} , index=range( 1,2 ) )
df_vs

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Teste de Hipótese

# COMMAND ----------

df_ab.sample(10)

# COMMAND ----------

df_test = df_ab[['group', 'converted']].groupby('group').agg({'converted': ['sum', 'count']})
df_test = df_test.rename(columns={'sum': 'converted', 'count': 'total'})
df_test

# COMMAND ----------

chi_val, pval, dof, expected = chi2_contingency(df_test)

print(f'p-value: {pval}')

if pval < significance_lvl: 
    print('\nRejeitamos a hipótese nula')
else: 
    print('\nFalhamos em rejeitar a hipótese nula')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Traduzindo para R$

# COMMAND ----------

df5 = df2.copy()

# COMMAND ----------

df5.head()

# COMMAND ----------

df_day = df5[['user_id', 'session_dt']].groupby('session_dt').count().reset_index()
df_day.rename(columns={'user_id': 'total_users'}, inplace=True)
df_day.head()

# COMMAND ----------

# Current GMV
df_day['current_buyers'] = np.ceil(df_day['total_users'] * 0.13).astype(int)
df_day['current_GMV'] = df_day['current_buyers'] * 4500

df_day.head()

# COMMAND ----------

# Expected GMV
df_day['expected_buyers'] = np.ceil(df_day['total_users'] * 0.15).astype(int)
df_day['expected_GMV'] = df_day['expected_buyers'] * 4500

df_day.head()

# COMMAND ----------

gmv = df_day['current_GMV'].sum()
new_gmv = df_day['expected_GMV'].sum()

lift = ((new_gmv - gmv) / gmv)*100
abs_lift = new_gmv - gmv

print(f'Current GMV: {gmv}')
print(f'Expected GMV: {new_gmv}')

print(f'\nExpected % Lift: %{np.round(lift, 2)}')
print(f'Expected Absolute Lift: {abs_lift}')


DESAFIO DIO - Previsão de Estoque Inteligente na AWS com Sagemaker Canvas

Projeto: Previsão de Estoque Inteligente na AWS com SageMaker Canvas

Introdução:

Neste projeto, usaremos o SageMaker Canvas, um serviço de ML de baixo código da AWS, para criar um modelo de previsão de estoque. Este modelo nos ajudará a prever a demanda futura de produtos com base em dados históricos de vendas e outros fatores relevantes.

Pré-requisitos:

Conta da AWS
Conhecimento básico de ML
Experiência com o SageMaker Canvas (opcional)
Passos:

1. Criar um notebook do SageMaker:

import sagemaker
from sagemaker.canvas import *
2. Importar as bibliotecas necessárias:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
3. Carregar os dados:

Carregue os dados de vendas históricos em um DataFrame do Pandas. Certifique-se de que os dados contenham as seguintes colunas:

* product_id
* sales_date
* sales_quantity
4. Explorar os dados:

df.head()
df.describe()
df.plot(x='sales_date', y='sales_quantity')
5. Criar o modelo de previsão:

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df[['sales_date']], df['sales_quantity'], test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)
6. Avaliar o modelo:

# Fazer previsões nos dados de teste
y_pred = model.predict(X_test)

# Calcular as métricas de avaliação
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('MSE:', mse)
print('MAE:', mae)
7. Implantar o modelo:

# Criar um endpoint do SageMaker
endpoint_name = 'previsao-estoque'
role = sagemaker.get_execution_role()

endpoint = sagemaker.Endpoint(endpoint_name, role, 'linear-learner')
endpoint.deploy(model, initial_instance_count=1)
Conclusão:

Neste projeto, criamos um modelo de previsão de estoque usando o SageMaker Canvas. Este modelo pode ser usado para prever a demanda futura de produtos com base em dados históricos de vendas e outros fatores relevantes. Isso pode ajudar as empresas a otimizar seus níveis de estoque e reduzir custos.

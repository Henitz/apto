import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import requests
import tempfile
import random

# Definindo a semente para garantir reprodutibilidade
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Definição da função para baixar arquivos temporários do GitHub
def baixar_arquivo_temporario(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, 'wb') as f:
            f.write(response.content)
        return temp_file.name
    else:
        st.error(f"Erro ao baixar o arquivo: {response.status_code}")
        return None

# Definição do modelo de rede neural com regularização
class RegularizedRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, l1_lambda=0.0, l2_lambda=0.0):
        super(RegularizedRegressionModel, self).__init__()
        layers = []
        last_size = input_size
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size

        layers.append(nn.Linear(last_size, output_size))  # Camada de saída
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def regularization_loss(self):
        l1_norm = sum(p.abs().sum() for p in self.network.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in self.network.parameters())
        return self.l1_lambda * l1_norm + self.l2_lambda * l2_norm

# Função para carregar o modelo e os scalers
def load_model(model_path, scaler_X_path, scaler_y_path):
    model = torch.load(model_path)
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)
    return model, scaler_X, scaler_y

# Função para fazer a predição
def make_prediction(model, scaler_X, scaler_y, input_data):
    input_data_scaled = scaler_X.transform(input_data)
    input_data_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    prediction = model(input_data_tensor)
    prediction_unscaled = scaler_y.inverse_transform(prediction.detach().numpy())
    return prediction_unscaled

# URL dos arquivos necessários
model_url = 'https://github.com/Henitz/apto/raw/master/best_model.pth'
scaler_X_url = 'https://github.com/Henitz/apto/raw/master/scaler_X.pkl'
scaler_y_url = 'https://github.com/Henitz/apto/raw/master/scaler_y.pkl'

# Caminhos temporários
model_path = baixar_arquivo_temporario(model_url)
scaler_X_path = baixar_arquivo_temporario(scaler_X_url)
scaler_y_path = baixar_arquivo_temporario(scaler_y_url)

# Carregar o modelo e os scalers
if model_path and scaler_X_path and scaler_y_path:
    model, scaler_X, scaler_y = load_model(model_path, scaler_X_path, scaler_y_path)
    if model is not None:
        model.eval()
    else:
        st.error("Erro ao carregar o modelo.")
else:
    st.error("Erro ao baixar os arquivos necessários para o modelo.")

# Interface do usuário usando Streamlit
st.title('Predição de Valores com Rede Neural')

# Entrada dos dados pelo usuário
area_util = st.number_input('Área útil (m²):', value=0.0)
suites = st.number_input('Número de suítes:', value=0)
vagas = st.number_input('Número de vagas:', value=0)

input_data = np.array([[area_util, suites, vagas]])

# Botão de predição
if st.button('Prever'):
    prediction = make_prediction(model, scaler_X, scaler_y, input_data)
    st.write(f"Valor Previsto: R$ {prediction[0][0]:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))



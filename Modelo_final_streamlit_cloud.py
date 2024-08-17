import torch
import torch.nn as nn
import streamlit as st
import requests
import tempfile
import pickle

# Definição da função para baixar arquivos temporários
def baixar_arquivo_temporario(url):
    st.write(f"Baixando arquivo de {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, 'wb') as f:
            f.write(response.content)
        st.write(f"Arquivo baixado e salvo em {temp_file.name}")
        return temp_file.name
    else:
        st.error(f"Erro ao baixar o arquivo: {response.status_code}")
        return None

# Definição da classe do modelo
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
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l2_norm = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_lambda * l1_norm + self.l2_lambda * l2_norm

# Função para carregar o modelo e scalers
def load_model(model_path, scaler_X_path, scaler_y_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model_config = checkpoint['model_config']
    model = RegularizedRegressionModel(
        input_size=4,  # Ajustando para 4 entradas na camada de entrada
        hidden_sizes=model_config['hidden_sizes'],
        output_size=model_config['output_size'],
        l1_lambda=model_config['l1_lambda'],
        l2_lambda=model_config['l2_lambda']
    )

    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=False)

    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y

# URLs dos arquivos no GitHub
model_url = 'https://github.com/Henitz/apto/raw/master/best_model.pth'
scaler_X_url = 'https://github.com/Henitz/apto/raw/master/scaler_X.pkl'
scaler_y_url = 'https://github.com/Henitz/apto/raw/master/scaler_y.pkl'

# Baixar os arquivos necessários
model_path = baixar_arquivo_temporario(model_url)
scaler_X_path = baixar_arquivo_temporario(scaler_X_url)
scaler_y_path = baixar_arquivo_temporario(scaler_y_url)

if model_path and scaler_X_path and scaler_y_path:
    model, scaler_X, scaler_y = load_model(model_path, scaler_X_path, scaler_y_path)
    st.write('Modelo e scalers carregados com sucesso.')
else:
    st.error('Erro ao carregar o modelo ou os scalers.')

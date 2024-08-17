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
        model_config['input_size'],
        model_config['hidden_sizes'],
        model_config['output_size'],
        model_config['l1_lambda'],
        model_config['l2_lambda']
    )

    state_dict = checkpoint['model_state_dict']

    # Debugging: Imprimir chaves e formas no state_dict e no modelo carregado
    for key in state_dict:
        if key in model.state_dict():
            st.write(f"Chave: {key}")
            st.write(f"Forma esperada no modelo: {model.state_dict()[key].shape}")
            st.write(f"Forma encontrada no state_dict: {state_dict[key].shape}")
        else:
            st.write(f"Chave {key} não encontrada no modelo carregado.")

    # Ajustar manualmente o state_dict para corresponder as camadas
    adjusted_state_dict = {}
    for key in state_dict:
        if key in model.state_dict() and state_dict[key].shape == model.state_dict()[key].shape:
            adjusted_state_dict[key] = state_dict[key]

    model.load_state_dict(adjusted_state_dict, strict=False)

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

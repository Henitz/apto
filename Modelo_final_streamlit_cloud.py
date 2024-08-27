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
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l2_norm = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_lambda * l1_norm + self.l2_lambda * l2_norm


# Função para carregar o modelo e scalers
def load_model(model_path, scaler_X_path, scaler_y_path):
    set_seed(42)  # Garantir que a semente seja aplicada antes de carregar o modelo

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model_config = checkpoint['model_config']
    model = RegularizedRegressionModel(
        input_size=3,  # Três features: area_util, suites, andar
        hidden_sizes=model_config['hidden_sizes'],
        output_size=model_config['output_size'],
        l1_lambda=model_config['l1_lambda'],
        l2_lambda=model_config['l2_lambda']
    )

    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)

    model.eval()

    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y


# Função para fazer predição
def make_prediction(model, scaler_X, scaler_y, input_data):
    set_seed(42)  # Garantir que a semente seja aplicada antes de cada previsão

    input_data_scaled = scaler_X.transform(input_data)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor)

    prediction_original = scaler_y.inverse_transform(prediction.numpy().reshape(-1, 1))
    return prediction_original


# Interface Streamlit
def main():
    st.title("Predição de Valor de Apartamento")

    set_seed(42)  # Garantir que a semente seja aplicada no início da aplicação

    model_url = 'https://github.com/Henitz/apto/raw/master/best_model.pth'
    scaler_X_url = 'https://github.com/Henitz/apto/raw/master/scaler_X.pkl'
    scaler_y_url = 'https://github.com/Henitz/apto/raw/master/scaler_y.pkl'

    model_path = baixar_arquivo_temporario(model_url)
    scaler_X_path = baixar_arquivo_temporario(scaler_X_url)
    scaler_y_path = baixar_arquivo_temporario(scaler_y_url)

    if model_path and scaler_X_path and scaler_y_path:
        model, scaler_X, scaler_y = load_model(model_path, scaler_X_path, scaler_y_path)

        area_util = st.text_input("Área Útil (m²)", value="0,0")
        suites = st.number_input("Número de Suítes", min_value=0, value=0, step=1)
        andar = st.number_input("Andar", min_value=0, value=0, step=1)

        area_util = float(area_util.replace(",", "."))

        input_data = pd.DataFrame({
            'area_util': [area_util],
            'suites': [suites],
            'andar': [andar]
        })

        if st.button("Prever Valor"):
            prediction = make_prediction(model, scaler_X, scaler_y, input_data)
            valor_formatado = f"R$ {prediction[0][0]:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")
            st.write(f"Valor Previsto: {valor_formatado}")
    else:
        st.error('Erro ao carregar o modelo ou os scalers.')


if __name__ == "__main__":
    main()

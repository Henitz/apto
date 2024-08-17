import torch
import torch.nn as nn
import streamlit as st
import requests
import tempfile

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

# Defina aqui as outras funções ou classes

# Carregar modelo e scalers
model_url = 'https://github.com/Henitz/apto/raw/master/best_model.pth'
scaler_X_url = 'https://github.com/Henitz/apto/raw/master/scaler_X.pkl'
scaler_y_url = 'https://github.com/Henitz/apto/raw/master/scaler_y.pkl'

model_path = baixar_arquivo_temporario(model_url)
scaler_X_path = baixar_arquivo_temporario(scaler_X_url)
scaler_y_path = baixar_arquivo_temporario(scaler_y_url)

if model_path and scaler_X_path and scaler_y_path:
    model, scaler_X, scaler_y = load_model(model_path, scaler_X_path, scaler_y_path)
    st.write('Modelo e scalers carregados com sucesso.')
else:
    st.error('Erro ao carregar o modelo ou os scalers.')

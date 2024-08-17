import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import streamlit as st


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
    checkpoint = torch.load(model_path)

    model_config = checkpoint['model_config']
    model = RegularizedRegressionModel(
        model_config['input_size'],
        model_config['hidden_sizes'],
        model_config['output_size'],
        model_config['l1_lambda'],
        model_config['l2_lambda']
    )

    # Ajustar as chaves do state_dict e corresponder as dimensões do modelo
    state_dict = checkpoint['model_state_dict']
    adjusted_state_dict = {}

    for key in state_dict:
        if key.startswith('network.') and key in model.state_dict():
            if state_dict[key].shape == model.state_dict()[key].shape:
                adjusted_state_dict[key] = state_dict[key]

    model.load_state_dict(adjusted_state_dict, strict=False)

    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y


# Função para fazer predição
def make_prediction(model, scaler_X, scaler_y, input_data):
    input_data_scaled = scaler_X.transform(input_data)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
    prediction_original = scaler_y.inverse_transform(prediction.numpy().reshape(-1, 1))
    return prediction_original


# Interface Streamlit
def main():
    st.title("Predição de Valor de Apartamento")

    # Carregar o modelo e os scalers
    model_path = "best_model.pth"
    scaler_X_path = "scaler_X.pkl"
    scaler_y_path = "scaler_y.pkl"
    model, scaler_X, scaler_y = load_model(model_path, scaler_X_path, scaler_y_path)

    # Inputs do usuário
    area_util = st.number_input("Área Útil (m²)", min_value=0.0, value=0.0, step=1.0)
    suites = st.number_input("Número de Suítes", min_value=0, value=0, step=1)
    andar = st.number_input("Andar", min_value=0, value=0, step=1)

    input_data = pd.DataFrame({
        'area_util': [area_util],
        'suites': [suites],
        'andar': [andar]
    })

    if st.button("Prever Valor"):
        prediction = make_prediction(model, scaler_X, scaler_y, input_data)
        st.write(f"Valor Previsto: R$ {prediction[0][0]:,.2f}")


if __name__ == "__main__":
    main()

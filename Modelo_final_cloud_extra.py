import torch
import torch.nn as nn
import pickle
import streamlit as st
import numpy as np

# Definindo a semente para garantir reprodutibilidade
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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

# Função para carregar o modelo e o state_dict
def load_model_and_config(model_path, config_path, scaler_X_path, scaler_y_path):
    try:
        # Carregando o state_dict do modelo
        state_dict = torch.load(model_path)

        # Carregando as configurações do modelo
        with open(config_path, 'rb') as f:
            model_config = pickle.load(f)

        # Criando a instância do modelo
        model = RegularizedRegressionModel(
            input_size=model_config['input_size'],
            hidden_sizes=model_config['hidden_sizes'],
            output_size=model_config['output_size'],
            l1_lambda=model_config['l1_lambda'],
            l2_lambda=model_config['l2_lambda']
        )

        # Carregando o state_dict no modelo
        model.load_state_dict(state_dict)

        # Carregando os scalers
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)

        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Erro ao carregar o modelo, as configurações ou os scalers: {e}")
        return None, None, None

# Caminhos para os arquivos necessários
model_path = 'path/to/best_model.pth'
config_path = 'path/to/best_model_config.pkl'
scaler_X_path = 'path/to/scaler_X.pkl'
scaler_y_path = 'path/to/scaler_y.pkl'

# Carregar o modelo, as configurações e os scalers
model, scaler_X, scaler_y = load_model_and_config(model_path, config_path, scaler_X_path, scaler_y_path)

# Verificação para garantir que o modelo foi carregado corretamente
if model is not None:
    model.eval()
    st.write("Modelo carregado e colocado em modo de avaliação.")
else:
    st.error("Modelo não foi carregado corretamente. Verifique os arquivos e tente novamente.")

# Interface do usuário usando Streamlit
st.title('Previsão de Valor com Modelo de Regressão Regularizada')

# Entrada dos dados pelo usuário
area_util = st.number_input('Área útil (m²):', value=0.0)
suites = st.number_input('Número de suítes:', value=0)
vagas = st.number_input('Número de vagas:', value=0)

input_data = np.array([[area_util, suites, vagas]])

# Botão para executar a previsão
if st.button('Prever'):
    if model is not None:
        prediction = make_prediction(model, scaler_X, scaler_y, input_data)
        st.write(f"Valor previsto: R$ {prediction[0][0]:,.2f}")
    else:
        st.error("Modelo não carregado. Não é possível fazer a previsão.")

# Função para fazer a predição
def make_prediction(model, scaler_X, scaler_y, input_data):
    input_data_scaled = scaler_X.transform(input_data)
    input_data_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    prediction = model(input_data_tensor)
    prediction_unscaled = scaler_y.inverse_transform(prediction.detach().numpy())
    return prediction_unscaled

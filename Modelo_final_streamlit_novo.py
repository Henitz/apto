import os
import torch
import torch.nn as nn
import pickle
import numpy as np
import streamlit as st

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
    def __init__(self, input_size=3, hidden_sizes=[128, 64], output_size=1, l1_lambda=0.0, l2_lambda=0.0):
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

# Função para carregar o modelo, state_dict e scalers
def load_model_and_config(model_path, config_path, scaler_X_path, scaler_y_path):
    try:
        # Carregando o state_dict do modelo com weights_only=True para evitar o problema de segurança mencionado
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

        # Carregando as configurações do modelo
        with open(config_path, 'rb') as f:
            model_config = pickle.load(f)

        # Criando a instância do modelo com a arquitetura correta
        model = RegularizedRegressionModel(
            input_size=model_config['input_size'],
            hidden_sizes=model_config['hidden_sizes'],
            output_size=model_config['output_size'],
            l1_lambda=model_config['l1_lambda'],
            l2_lambda=model_config['l2_lambda']
        )

        # Ajustando o state_dict se necessário
        new_state_dict = {}
        for key, value in state_dict.items():
            # Adaptando as chaves para que correspondam ao modelo atual
            if key.startswith('network.'):
                new_key = key
            else:
                new_key = f'network.{key}'
            new_state_dict[new_key] = value

        # Carregando o state_dict no modelo
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Chaves ausentes no state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Chaves inesperadas no state_dict: {unexpected_keys}")

        # Carregando os scalers
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)

        return model, scaler_X, scaler_y
    except Exception as e:
        print(f"Erro ao carregar o modelo, as configurações ou os scalers: {e}")


# Função para fazer a predição
def make_prediction(model, scaler_X, scaler_y, input_data):
    input_data_scaled = scaler_X.transform(input_data)
    input_data_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    prediction = model(input_data_tensor)
    prediction_unscaled = scaler_y.inverse_transform(prediction.detach().numpy())
    return prediction_unscaled

# Definir o diretório da aplicação
app_dir = os.path.dirname(os.path.abspath(__file__))

# Caminhos para os arquivos necessários (relativos ao diretório da aplicação)
model_path = os.path.join(app_dir, 'best_model.pth')
config_path = os.path.join(app_dir, 'best_model_config.pkl')
scaler_X_path = os.path.join(app_dir, 'scaler_X.pkl')
scaler_y_path = os.path.join(app_dir, 'scaler_y.pkl')

# Carregar o modelo, as configurações e os scalers
model, scaler_X, scaler_y = load_model_and_config(model_path, config_path, scaler_X_path, scaler_y_path)

# Verificação para garantir que o modelo foi carregado corretamente
if model is not None:
    model.eval()
    st.write("Modelo carregado e colocado em modo de avaliação.")

    # Interface do usuário usando Streamlit
    st.title('Previsão de Valor com Modelo de Regressão Regularizada')

    # Entrada dos dados pelo usuário
    area_util = st.number_input('Área útil (m²):', value=0.0)
    suites = st.number_input('Número de suítes:', value=0)
    vagas = st.number_input('Número de vagas:', value=0)

    input_data = np.array([[area_util, suites, vagas]])

    # Botão para executar a previsão
    if st.button('Prever'):
        prediction = make_prediction(model, scaler_X, scaler_y, input_data)
        st.write(f"Valor previsto: R$ {prediction[0][0]:,.2f}")
else:
    st.error("Modelo não foi carregado corretamente. Verifique os arquivos e tente novamente.")

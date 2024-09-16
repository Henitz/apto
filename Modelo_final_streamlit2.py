import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle


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
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Função para fixar as sementes aleatórias
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Fixando a semente aleatória
set_seed(42)


# Função para carregar o modelo salvo
def load_model(model_path, scaler_X_path, scaler_y_path, config_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Carregar a configuração do modelo
    with open(config_path, 'rb') as f:
        model_config = pickle.load(f)

    # Criar uma instância do modelo com a configuração carregada
    model = RegularizedRegressionModel(
        model_config['input_size'],
        model_config['hidden_sizes'],
        model_config['output_size'],
        model_config['l1_lambda'],
        model_config['l2_lambda']
    )

    # Carregar o state_dict ajustando as chaves, se necessário
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("model."):
            new_state_dict[k] = v
        else:
            new_state_dict["model." + k] = v

    model.load_state_dict(new_state_dict, strict=False)

    # Carregar os scalers
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y


# Função para formatar valores em moeda brasileira
def format_brazilian_currency(value):
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# Exemplo de uso
model_path = "best_model.pth"  # Caminho para o modelo na raiz da aplicação
scaler_X_path = "scaler_X.pkl"  # Caminho para o scaler X na raiz da aplicação
scaler_y_path = "scaler_y.pkl"  # Caminho para o scaler y na raiz da aplicação
config_path = "best_model_config.pkl"  # Caminho para o arquivo de configuração na raiz da aplicação

# Configuração da interface Streamlit
st.title("Predição com Modelo de Regressão Regularizado")

# Carregar o modelo e os scalers
model, scaler_X, scaler_y = None, None, None
if st.button("Carregar Modelo"):
    model, scaler_X, scaler_y = load_model(model_path, scaler_X_path, scaler_y_path, config_path)
    st.success("Modelo carregado com sucesso!")

# Entrada do usuário para novas previsões
st.subheader("Insira os valores para as predições")

area_util = st.number_input("Área Útil (m²)", min_value=0.0, step=0.1)
suites = st.number_input("Número de Suítes", min_value=0, step=1)
andar = st.number_input("Andar", min_value=0, step=1)

if st.button("Prever") and model:
    # Dados de entrada para previsão
    X_example = np.array([[area_util, suites, andar]])
    X_example_scaled = scaler_X.transform(X_example)

    # Fazer previsões
    with torch.no_grad():
        model.eval()
        X_tensor = torch.tensor(X_example_scaled, dtype=torch.float32)
        predictions = model(X_tensor).numpy()

    # Inverter a escala das previsões para o valor original
    predictions_original = scaler_y.inverse_transform(predictions)
    predicted_value = predictions_original[0]

    # Exibir o valor previsto formatado em moeda brasileira
    st.write("Predição do Valor Total Fator:", format_brazilian_currency(predicted_value))

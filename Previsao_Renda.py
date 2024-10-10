import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(page_title="Análise e Previsão de Renda", layout="wide")

# Carregar o modelo salvo com pickle
model = pickle.load(open('modelo_renda_log.pkl', 'rb'))

# Carregar os dados (substitua 'seu_arquivo.csv' pelo nome real do seu arquivo de dados)
@st.cache_data
def load_data():
    data = pd.read_csv('previsao_de_renda.csv')
    return data

data = load_data()

# Função para criar gráficos
def plot_relation(x, y, data, kind='scatter'):
    plt.figure(figsize=(8, 6))
    if kind == 'scatter':
        sns.scatterplot(x=x, y=y, data=data,palette='Set1')
    elif kind == 'box':
        sns.barplot(x=x, y=y, data=data,palette='Set1')
    plt.title(f'Relação entre {x} e {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    return plt

# Criar navegação na sidebar
pagina = st.sidebar.selectbox("Escolha a página", ["Análises Preliminares", "Preveja sua Renda!"])

# Página de Análises Preliminares
if pagina == "Análises Preliminares":
    st.header("Análises Preliminares")
    st.write(" ##### Nesta página separamos algumas análises entre as variáveis que utilizamos no modelo de previsão")
    

    st.subheader("Relação entre Tempo de Emprego e Renda")
    fig1 = plot_relation('tempo_emprego', 'renda', data)
    st.pyplot(fig1)
    
    st.subheader("Relação entre Idade e Renda")
    fig2 = plot_relation('idade', 'renda', data)
    st.pyplot(fig2)
    
    st.subheader("Renda por Posse de Imóvel")
    fig3 = plot_relation('posse_de_imovel', 'renda', data, kind='box')
    st.pyplot(fig3)
    
    st.subheader("Renda por Tipo de Renda")
    fig4 = plot_relation('tipo_renda', 'renda', data, kind='box')
    st.pyplot(fig4)

# Página de Previsão de Renda
elif pagina == "Preveja sua Renda!":
    st.header("Previsão de Renda")
    st.write("Insira as variáveis abaixo para obter uma previsão de renda.")

    # Inputs do usuário
    tempo_emprego = st.number_input("Tempo de Emprego (em anos):", min_value=0, value=10)
    idade = st.number_input("Idade:", min_value=18, value=30)
    posse_de_imovel = st.selectbox("Possui imóvel próprio?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
    tipo_renda_empresario = st.selectbox("É empresário?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")

    # Converter os valores para um dataframe adequado ao modelo
    entrada = pd.DataFrame([[tempo_emprego, idade, posse_de_imovel, tipo_renda_empresario]], 
                           columns=['tempo_emprego', 'idade', 'posse_de_imovel', 'tipo_renda_empresario'])

    # Realizar a previsão ao clicar no botão
    if st.button("Prever Renda"):
        # Prever renda_log e converter de log para escala original
        predicao_log = model.predict(entrada)
        predicao = np.exp(predicao_log)

        st.write(f"A renda prevista é: R$ {round(predicao[0], 2)}")

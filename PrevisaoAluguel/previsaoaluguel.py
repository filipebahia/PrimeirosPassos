import streamlit as st
import pandas as pd
import numpy as np
from pandas import read_csv
from wordcloud import WordCloud
import base64
from joblib import load
from PIL import Image

def tratamento(dados, cidade):
    # Aplicando o filtro de acordo com a cidade selecionada
    dados = dados[(dados['city'] == cidade)]
    filtro = dados['city'] == cidade
    dados = dados[filtro]

    # Excluindo os atributos categóricos e demais atributos indesejáveis
    dados = dados.drop(columns = ['city','animal', 'furniture', 'hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)'])

    # Renomeando todas as colunas para o idioma Portugês
    colunas = ['Área', 'Quartos', 'Banheiro', 'Vagas de Garagem', 'Andar', 'Valor Total']
    dados.set_axis(colunas, axis='columns', inplace=True)

    # Eliminando linhas que contém o caractere "-"
    filtro = dados['Andar'] != '-'
    dados = dados[filtro]

    # Realizando reset do index
    dados = dados.reset_index(drop=True)

    # Convertendo o atributo "Andar" do tipo object para valor numérico
    dados['Andar'] = pd.to_numeric(dados['Andar'])

    #st.subheader('**Dataframe após o tratamento**')
    #st.dataframe(dados.head(10))
    #dados_tratados = dados.copy()
    return dados

def aplica_modelo(dados_tratados):
    # Importando bibliotecas para a modelagem preditiva
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score

    # Separando o array em componentes de input e output
    array = dados_tratados.values
    X = array[:,0:5]
    Y = array[:,5]

    # Divide os dados em treino e teste
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)

    # Criando o modelo
    modelo = LinearRegression()

    # Treinando o modelo
    modelo.fit(X_train, Y_train)

    # Fazendo previsões
    Y_pred = modelo.predict(X_test)
    
    return modelo

def main():
    # Ocultando mensagens de warnings
    st.set_option('deprecation.showImageFormat', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    image = Image.open('D:/Cursos/UNI7/uni7.png')
    #st.image(image, caption='Sunrise by the mountains', use_column_width=True)
    st.image(image, use_column_width=True)

    #st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)
    #st.title("<h1 style='text-align: center;>Análise Exploratória de Dados<h1>", unsafe_allow_html=True)

    st.markdown('*Disciplina: Programação para Data Science*')
    st.markdown('*Professor: Edson Cavalcanti*')
    st.markdown('*Aluno: Filipe de Macêdo Peixoto / Marcelo Medeiros de Vasconcellos / Antônio de Pádua Dias Costa Júnior*')
    st.markdown("<h1 style='text-align: center;'>Análise Exploratória e Previsão de Dados</h1>", unsafe_allow_html=True)


    # Carregando os dados
    #arquivo = "D:/Cursos/UNI7/07 - Programação para Data Science/Trabalho/streamlit/houses_to_rent_v2.csv"
    #dados = read_csv(arquivo)
    #array = dados.values

    # Exibindo o dataframe
    #dados.head(10)

    st.subheader('**Carregando o dataframe**')
    df = st.file_uploader('Escolha o dataset (.csv)', type = 'csv')
    if df is not None:
        data = pd.read_csv(df)
        #st.subheader('**Dataframe selecionado**')
        qtd = st.slider('Quantos itens?', 0, 12000, 5)
        st.dataframe(data.head(qtd))

    city = st.text_input('Informe a cidade: ')

    st.subheader('**Tratamendo dos dados**')
    if st.checkbox('Tratar os dados'):
    #if st.button('Tratar os dados'):
        dados_tratados = pd.DataFrame(tratamento(data, city))
        st.dataframe(dados_tratados.head(10))

    st.subheader('**Criação do modelo de previsão**')
    if st.checkbox('Gerar modelo'):
    #if st.button('Gerar modelo'):
        dados_tratados = pd.DataFrame(tratamento(data, city))
        model = aplica_modelo(dados_tratados)
        st.write('Modelo de regressão linear criado!\nO último passo é inserir as informações para previsão.')

    ## Selecionando as informações do imóvel
    st.subheader('**Informações de entrada para previsão**')
    if st.checkbox('Inserir informações'):
    #if st.button('Testar modelo'):
        dados_tratados = pd.DataFrame(tratamento(data, city))
        model = aplica_modelo(dados_tratados)

        area = st.number_input('Área: ', max_value=1100.0, format="%f", step=1.0)
        quartos = st.number_input('Quartos: ', max_value=1100.0, format="%f", step=1.0)
        banheiros = st.number_input('Banheiro: ', max_value=1100.0, format="%f", step=1.0)
        vagas = st.number_input('Vagas: ', max_value=1100.0, format="%f", step=1.0)
        andar = st.number_input('Andar: ', max_value=1100.0, format="%f", step=1.0)

    st.subheader('**Resultado da previsão**')
    if st.checkbox('Previsão'):
        # Dados do imóvel para predição do valor total
        # Array com valores de entrada para previsão -> [Área, Quartos, Banheiro, Vagas de Garagem, Andar, Condomínio, Aluguel, IPTU, Seguro]
        valores_entrada = np.array([area, quartos, banheiros, vagas, andar]).reshape((-1,5))
        predicao = model.predict(valores_entrada)
        num = predicao[0]
        #print("\nO valor previsto do imóvel é: ",predicao[0])
        st.write('O valor previsto do imóvel é ', num)

if __name__ == '__main__':
    main()

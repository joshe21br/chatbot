#!-*-coding: utf-8 -*-
#
# josuesouzadasilva@gmail.com
# joshebr18@protonmail.com
# Linkdin: Josué Souza da Silva
# Gitlab: https://gitlab.com/joshe_sotero
# Pequenos projetos pessoais: http://joshe.epizy.com/
# https://myadmin-21.herokuapp.com/
"""
Criação do chatbot com NLTK -->> Nivel Básico
---------------------------
# Teste realizado na propria máquina utilizando o postman
# Resultado: 100 funcionando
"""

#------------------------------------
# Etapa 1: Importação das bibliotecas
#------------------------------------
import os
import requests
from flask import Flask, request, jsonify 
import bs4 as bs
import urllib.request
import re
import nltk
import numpy as np
import random
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')

#----------------------------------------------------------
# Etapa 2: Carregamento e pré-pocessamento da base de dados
#----------------------------------------------------------
print('\nTodo o conteúdo do html\n')
dados = urllib.request.urlopen('adicionar a url para baixar as iformações')
dados = dados.read()
print(dados)

print('\nSomente o html de forma mais organizado\n')
dados_html = bs.BeautifulSoup(dados, 'lxml')
print(dados_html)

print('\nApenas os parágrafos\n')
paragrafos = dados_html.find_all('p')
print(paragrafos)
print('\n')

print('Busca por parágrafos\n')
print(paragrafos[0])
print('\n')

print('Mostrar sem as tags\n')
print(paragrafos[0].text)

print('Mostrar todos os textos sem as tags\n')

conteudo = ''
for p in paragrafos:
    conteudo += p.text

print(conteudo)
print('')

print('Pré-processamento\n')
print('Transformando todas as letras em minúsculas\n')

conteudo = conteudo.lower()
print(conteudo)
print('')

print('Separando por frases\n')
lista_sentencas = nltk.sent_tokenize(conteudo)

print(lista_sentencas)
print('')

print('Quantidade de frases\n')
print(len(lista_sentencas))
print('')

print('Tipo da variável\n')
print(type(lista_sentencas))
    
print('Selecionar paragrafo\n')
print(lista_sentencas[0])
print('')

pln = spacy.load('pt')
print(pln)

print('As stop_words\n')
stop_words = spacy.lang.pt.stop_words.STOP_WORDS
print(stop_words)
print('')

print(type(stop_words))
print(len(stop_words))
print('')

print(string.punctuation)
print('')

print('Remoção de URLs\n')

def preprocessamento(texto):
    # Remoção das URLs
    texto = re.sub("https?://[A-Za-z0-9./]+", ' ', texto)
    # Espaço em branco
    texto = re.sub(r' +', ' ', texto ) 

    documento = pln(texto)
    
    lista = []
    
    for token in documento:
        lista.append(token.lemma_)

    
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
   
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()]) 
    
    return lista


texto_teste = ' Adcionar a url para extração e processamento  ' + lista_sentencas[0]
print(texto_teste)
print('')

resultado = preprocessamento(texto_teste)
print(resultado)

lista_processada = []
for i in range(len(lista_sentencas)):
    lista_processada.append(preprocessamento(lista_sentencas[i]))

for _ in range(5):
    i = random.randint(0, len(lista_sentencas) - 1)
    print(lista_sentencas[i])
    print(lista_sentencas[i])
    print('-------')


#------------------------------------
# Etapa 3:  Mensagens de boas-vindas
#------------------------------------
saudacoes = ('hey', 'ola', 'oi', 'opa', 'eae')
resposta = ('hey', 'ola', 'oi', 'opa', 'Como vai!', 'Oi vc esta bem?')

def responder_saudacao(texto):
    for palavra in texto.split():
        if palavra.lower() in saudacoes:
            return random.choice(resposta)


print('Olá, tudo bem?')

resp = str(input('>>: '))
print(responder_saudacao(resp))
print('')

frases_teste = lista_processada[:3]
print(frases_teste)
print('')

frases_teste.append(frases_teste[0])
print(frases_teste)
print('')

#---------------
# vetores
#--------------

vectores_palavras = TfidfVectorizer()
palavras_vectorizadas = vectores_palavras.fit_transform(frases_teste)
print(type(palavras_vectorizadas))
print('')
print(palavras_vectorizadas)
print('')
print(vectores_palavras.get_feature_names())
print('')
print(len(vectores_palavras.get_feature_names()))
print('')
print(vectores_palavras.vocabulary_)
print('')
print(vectores_palavras.idf_)
print('')
print(palavras_vectorizadas.todense())
print('')
print(palavras_vectorizadas.todense().shape)
print('')

#----------------------
#Cosine similaridades
#---------------------
print(cosine_similarity(palavras_vectorizadas[0], palavras_vectorizadas[1]))
print('')
print(cosine_similarity(palavras_vectorizadas[0], palavras_vectorizadas[3]))
print('')
print(cosine_similarity(palavras_vectorizadas[0], palavras_vectorizadas))
print('')

similaridade = cosine_similarity(palavras_vectorizadas[0], palavras_vectorizadas)
print(similaridade)
print(similaridade.argsort())
print('')

i = similaridade.argsort()[0][-2]
print(i)
print('')
i = i.flatten()
print(i)

#---------------------------------------------
# Etapa 4: Função para as Resposta do chatbot
#---------------------------------------------
def responder(texto_usuario):
    resposta_chatbot = ''
    lista_processada.append(texto_usuario)

    tfidf = TfidfVectorizer()
    palavras_vectorizadas = tfidf.fit_transform(lista_processada)

    similaridade = cosine_similarity(palavras_vectorizadas[-1], palavras_vectorizadas)

    indice_sentenca = similaridade.argsort()[0][-2]
    vetor_similar = similaridade.flatten()
    vetor_similar.sort()
    vetor_encontrado = vetor_similar[-2]

    if(vetor_encontrado == 0):
        resposta_chatbot = resposta_chatbot + 'Desculpa, mas não entendi!'
        return resposta_chatbot
    else:
        resposta_chatbot = resposta_chatbot + lista_sentencas[indice_sentenca]
        return resposta_chatbot

#------------------------------
# Etapa 5 criação da API Flask
#------------------------------
app = Flask(__name__)

#------------------------------------------
# Etapa 6: Função que retorna as respostas
#------------------------------------------
@app.route("/<string:txt>", methods=["POST"])

def conversar(txt):
    resposta = ''
    texto_usuario = txt
    texto_usuario = texto_usuario.lower()
    if (responder_saudacao(texto_usuario) != None):
        resposta = responder_saudacao(texto_usuario)
    else:
        resposta = responder(preprocessamento(texto_usuario))
        lista_processada.remove(preprocessamento(texto_usuario))
    return jsonify({"texto_responder": resposta})

#-------------------------------    
# Etapa 7:  Iniciar a aplicação
#-------------------------------
app.run(port=5000, debug=False)


import streamlit as st
import gensim
import nltk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from langdetect import detect
from transformers import pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import time
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Exemplo de uso de barra de progresso
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('nps_chat', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("portuguese"))

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'last_sentiment' not in st.session_state:
    st.session_state.last_sentiment = ""
if 'sentiment_scores' not in st.session_state:
    st.session_state.sentiment_scores = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = Tokenizer()

st.info("⚠️ O código ainda precisa de melhorias e ajustes. 🧠 Trabalhando para aprimorar todas as funcionalidades.")
st.info("✅ A funcionalidade de análise de sentimento(ùltimo gráfico a esquerda, aparece ao ser eviada uma menssagem) está operando com uma certa precisão. Mas também necessita de melhorias")


with st.expander("Detalhes do Modelo"):
    st.write("""
    Este código implementa um chatbot inteligente para análise de sentimentos e detecção de intenções usando Streamlit. O chatbot utiliza técnicas avançadas de Processamento de Linguagem Natural (PLN), incluindo:
    
    - **Análise de Sentimento**: Utiliza o modelo pré-treinado `nlptown/bert-base-multilingual-uncased-sentiment` para classificar textos como positivos, negativos ou neutros.
    
    - **Treinamento de Modelo LSTM**: Cria e treina um modelo de rede neural recorrente (LSTM) usando embeddings do Word2Vec para detectar intenções em mensagens de usuários.
    
    - **Tokenização e Embeddings**: Implementa técnicas de pré-processamento, incluindo tokenização, remoção de stopwords e criação de vetores de palavras.
    
    - **Pipeline de Classificação**: Utiliza um classificador Naive Bayes para análise complementar baseada em TF-IDF.
    
    - **Tradução Automática**: Detecta e traduz textos automaticamente para garantir compatibilidade com diferentes idiomas.
    
    - **Histórico de Conversa**: Mantém um histórico das interações do usuário para oferecer respostas contextuais.
    """)


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Carregar dataset ampliado de intenções
@st.cache_data
def load_intent_data():
    posts = nltk.corpus.nps_chat.xml_posts()
    data = [(post.text, post.get("class")) for post in posts]
    intent_data = pd.DataFrame(data, columns=["text", "intent"])
    intent_data['text'] = intent_data['text'].apply(preprocess_text)
    return intent_data

intent_data = load_intent_data()

X_train, X_test, y_train, y_test = train_test_split(intent_data['text'], intent_data['intent'], test_size=0.1, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Treinamento do Word2Vec
@st.cache_data
def train_word2vec(X_train):
    sentences = [text.split() for text in X_train]
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return word2vec_model

word2vec_model = train_word2vec(X_train)

# Criar matriz de embeddings a partir do Word2Vec
def create_embedding_matrix(word2vec_model, tokenizer, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
    return embedding_matrix

# Tokenização e preparação dos dados para o LSTM
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer.fit_on_texts(X_train)
vocab_size = len(st.session_state.tokenizer.word_index) + 1
embedding_dim = 100  # Dimensão dos embeddings do Word2Vec

X_train_seq = st.session_state.tokenizer.texts_to_sequences(X_train)
X_test_seq = st.session_state.tokenizer.texts_to_sequences(X_test)

max_length = 100  # Comprimento máximo das sequências
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Criar matriz de embeddings
embedding_matrix = create_embedding_matrix(word2vec_model, st.session_state.tokenizer, vocab_size, embedding_dim)

# Modelo LSTM com embeddings do Word2Vec
@st.cache_resource
def create_sentiment_model(embedding_matrix, vocab_size, embedding_dim, max_length):
    model = Sequential([
        Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), 
            input_shape=(max_length,),  # Define o comprimento da sequência aqui
            trainable=False
        ),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dense(32, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')  # Número de classes
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if 'sentiment_model' not in st.session_state:
    st.session_state.sentiment_model = create_sentiment_model(embedding_matrix, vocab_size, embedding_dim, max_length)
    
# Treinamento do modelo (apenas uma vez)
if not st.session_state.model_trained:
    st.session_state.sentiment_model.fit(
        X_train_pad, 
        y_train_encoded,  # Usando rótulos codificados
        epochs=10, 
        batch_size=32, 
        validation_split=0.1
    )
    st.session_state.model_trained = True

# Gerar previsões
y_pred = st.session_state.sentiment_model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convertendo probabilidades em classes
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# Função de análise de sentimento aprimorada
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    if result['label'] == '1 star' or result['label'] == '2 stars':
        return "Negativo"
    elif result['label'] == '4 stars' or result['label'] == '5 stars':
        return "Positivo"
    return "Neutro"

# Tradução automática para maior compatibilidade
def detect_and_translate(text, target_lang='pt'):
    detected_lang = detect(text)
    translated_text = GoogleTranslator(source=detected_lang, target=target_lang).translate(text) if detected_lang != target_lang else text
    return translated_text

# Respostas personalizadas com contexto aprimorado
def chatbot_response(user_input):
    translated_input = detect_and_translate(user_input)
    sentiment = analyze_sentiment(translated_input)
    
    # Pré-processar o texto do usuário
    processed_input = preprocess_text(translated_input)
    
    # Tokenizar e padronizar o texto
    input_seq = st.session_state.tokenizer.texts_to_sequences([processed_input])
    input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')
    
    # Prever a intenção usando o modelo LSTM
    intent_pred = st.session_state.sentiment_model.predict(input_pad)
    intent_class = np.argmax(intent_pred, axis=1)
    intent_label = label_encoder.inverse_transform(intent_class)[0]
    
    # Respostas contextualizadas com base no histórico de conversas
    response_templates = {
        "saudação": ["Olá! Como posso ajudar?", "Oi! Tudo bem?", "Bom dia!", "Oi! Como vai?"],
        "ajuda": ["Claro! Qual sua dúvida?", "Estou aqui para ajudar!", "Como posso ser útil hoje?"],
        "documento": ["Envie o documento para análise!", "Vou processar seu documento!", "Por favor, envie o documento que deseja analisar."],
        "raiva": ["Sinto muito! Como posso resolver sua insatisfação?", "Entendo sua frustração, vamos resolver isso juntos.", "Peço desculpas pelo ocorrido. Como posso ajudar?"],
        "elogio": ["Muito obrigado! Fico feliz em ajudar!", "Agradeço pelo feedback positivo!", "Fico feliz que esteja satisfeito!"],
        "tristeza": ["Sinto muito que você esteja se sentindo assim. Posso ajudar em algo?", "Lamento ouvir isso. Estou aqui para o que precisar.", "Sinto muito. Conte comigo para o que precisar."],
        "nome": ["Meu nome é Chatbot. Como posso ajudar você hoje?", "Eu sou o Chatbot, seu assistente virtual.", "Pode me chamar de Chatbot. Como posso ajudar?"],
        "felicidade": ["Que bom que você está feliz! Como posso ajudar?", "Fico feliz em saber que você está contente!", "É ótimo ver você feliz! O que posso fazer por você?"],
        "despedida": ["Até logo! Espero ter ajudado.", "Tchau! Volte sempre que precisar.", "Até mais! Estou aqui se precisar."],
        "dúvida": ["Pode me explicar melhor?", "Não entendi completamente. Pode reformular?", "Poderia detalhar um pouco mais?"]
    }
    
    # Lógica para respostas contextualizadas
    if intent_label in response_templates:
        # Verificar o histórico de conversas para contexto
        if st.session_state.chat_history:
            last_user_message = st.session_state.chat_history[-1][1]  # Última mensagem do usuário
            last_bot_message = st.session_state.chat_history[-2][1] if len(st.session_state.chat_history) > 1 else ""  # Última resposta do bot
            
            # Exemplo: Se o usuário perguntar sobre um documento após uma saudação
            if intent_label == "documento" and "saudação" in last_bot_message.lower():
                response = "Claro! Por favor, envie o documento que deseja analisar."
            else:
                response = response_templates[intent_label][0]  # Escolhe a primeira resposta da lista (não aleatória)
        else:
            response = response_templates[intent_label][0]  # Escolhe a primeira resposta da lista (não aleatória)
    else:
        response = "Desculpe, não entendi. Pode reformular?"
    
    return response, sentiment

# Interface de chat
st.subheader("Chat com Chatbot Inteligente")
chat_container = st.container()

with chat_container:
    chat_placeholder = st.empty()
    
    # Renderizando mensagens anteriores
    for sender, message in st.session_state.chat_history:
        if sender == "Você":
            st.chat_message("user").markdown(f"<strong>Você:</strong> {message}", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"<strong>Chatbot:</strong> {message}", unsafe_allow_html=True)

user_input = st.chat_input("Digite sua mensagem")
if user_input:
    with st.spinner("Processando..."):
        response, sentiment = chatbot_response(user_input)
    
    # Adicionando a mensagem do usuário
    st.session_state.chat_history.append(("Você", user_input))
    
    # Adicionando a resposta do chatbot com o ícone
    with st.chat_message("assistant"):
        st.markdown(f"<strong>Chatbot:</strong> {response}", unsafe_allow_html=True)
    
    st.session_state.chat_history.append(("Chatbot", response))
    st.session_state.last_sentiment = sentiment
    
    # Re-renderizando as mensagens
    for sender, message in st.session_state.chat_history:
        if sender == "Você":
            st.chat_message("user").markdown(f"<strong>Você:</strong> {message}", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"<strong>Chatbot:</strong> {message}", unsafe_allow_html=True)
    
    st.rerun()

# Sidebar para métricas
st.sidebar.header("Análises e Métricas")
st.sidebar.subheader("Desempenho do Modelo")
st.sidebar.write("Accuracy:", accuracy_score(y_test, y_pred_labels))
st.sidebar.write("Classification Report:")
st.sidebar.text(classification_report(y_test, y_pred_labels))

st.sidebar.subheader("Sentimento da Última Mensagem")
st.sidebar.write(f"{st.session_state.last_sentiment}")

# Gráfico de evolução do sentimento
st.sidebar.subheader("Evolução do Sentimento")
data = [analyze_sentiment(msg[1]) for msg in st.session_state.chat_history if msg[0] == "Você"]
if data:
    fig, ax = plt.subplots()
    sns.lineplot(x=range(len(data)), y=[1 if d == "Positivo" else -1 if d == "Negativo" else 0 for d in data], ax=ax, marker='o')
    ax.set_title("Evolução do Sentimento")
    ax.set_xlabel("Mensagens")
    ax.set_ylabel("Pontuação de Sentimento")
    st.sidebar.pyplot(fig)

# Botão para limpar o histórico do chat
if st.sidebar.button("Limpar Histórico do Chat"):
    st.session_state.chat_history = []
    st.rerun()
import streamlit as st
import os
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
from textblob import TextBlob
from transformers import pipeline, Conversation
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import Callback
import time
import random

st.set_page_config(
page_title="ChatBot", 
page_icon="	:robot_face:",
)
st.title("Análise de sentimentos - chatbot")

nltk.download('punkt', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('nps_chat', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("portuguese"))

# Inicialização de variáveis na sessão
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
st.info("📊 A funcionalidade de análise de sentimento (último gráfico à esquerda, aparece ao enviar uma mensagem) está operando com certa imprecisão e necessita de melhorias...")

with st.expander("Detalhes do Modelo"):
    st.write("""
    Este código implementa um chatbot inteligente para análise de sentimentos e detecção de intenções usando Streamlit. O chatbot utiliza técnicas de Processamento de Linguagem Natural (PLN), incluindo:
    
    - **Análise de Sentimento**: Utiliza um modelo LSTM treinado com dados de sentimento (arquivo `tw_pt.csv`) para classificar o sentimento das mensagens do usuário. Além disso, integra o `TextBlob` para uma análise complementar de polaridade.
    
    - **Detecção de Intenções**: Um modelo LSTM bidirecional é treinado com embeddings do Word2Vec para detectar intenções em mensagens de usuários. O modelo é capaz de identificar intenções como "saudação", "ajuda", "despedida", entre outras, com base em um conjunto de dados do `nps_chat`.
    
    - **Pré-processamento de Texto**: Implementa técnicas de pré-processamento, incluindo tokenização, remoção de stopwords, e normalização de texto para garantir que as entradas sejam adequadas para os modelos.
    
    - **Tradução Automática**: Detecta automaticamente o idioma da mensagem do usuário e a traduz para o português, garantindo compatibilidade com os modelos treinados.
    
    - **Histórico de Conversa**: Mantém um histórico das interações do usuário, permitindo respostas contextuais e uma análise contínua do sentimento ao longo da conversa.
    
    - **Pipeline de Classificação**: Além do modelo LSTM, utiliza um classificador Naive Bayes baseado em TF-IDF para análise complementar de intenções.

    - **Interface Interativa**: A interface do chatbot é construída com Streamlit, permitindo uma interação fluida e amigável com o usuário. O histórico de conversas é exibido em tempo real, e o usuário pode limpar o histórico a qualquer momento.
    
    - **Modelo de Conversação**: Em casos onde a resposta gerada pelo modelo de intenção é muito genérica, o chatbot utiliza um modelo de conversação baseado no DialoGPT para gerar respostas mais contextualizadas.
    
    - **Melhorias Futuras**: O código está em constante evolução, com planos para melhorar a precisão da análise de sentimentos, adicionar mais intenções e integrar modelos de linguagem mais avançados, como o GPT-4.
    """)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

##############################################
# MODELO DE INTENÇÃO (dados do nps_chat)
##############################################

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

@st.cache_data
def train_word2vec(X_train):
    sentences = [text.split() for text in X_train]
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return word2vec_model

word2vec_model = train_word2vec(X_train)

def create_embedding_matrix(word2vec_model, tokenizer, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
    return embedding_matrix

# Tokenização e preparação dos dados para o modelo de intenção
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer.fit_on_texts(X_train)
vocab_size = len(st.session_state.tokenizer.word_index) + 1
embedding_dim = 100  # Dimensão dos embeddings do Word2Vec

X_train_seq = st.session_state.tokenizer.texts_to_sequences(X_train)
X_test_seq = st.session_state.tokenizer.texts_to_sequences(X_test)

max_length = 100  # Comprimento máximo das sequências
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

embedding_matrix = create_embedding_matrix(word2vec_model, st.session_state.tokenizer, vocab_size, embedding_dim)

@st.cache_resource
def create_intent_model(embedding_matrix, vocab_size, embedding_dim, max_length):
    model = Sequential([
        Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), 
            input_shape=(max_length,),
            trainable=True
        ),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dense(32, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if 'intent_model' not in st.session_state:
    st.session_state.intent_model = create_intent_model(embedding_matrix, vocab_size, embedding_dim, max_length)
    
# Callback para visualização do progresso do treinamento
class ProgressBarCallback(Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Treinamento do modelo: {epoch + 1}/{self.total_epochs} épocas concluídas.")

# Treinamento do modelo de intenção
if not st.session_state.model_trained:
    epochs = 10
    progress_callback = ProgressBarCallback(total_epochs=epochs)
    with st.spinner("Treinando modelo de intenção..."):
        st.session_state.intent_model.fit(
            X_train_pad,
            y_train_encoded,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            callbacks=[progress_callback],
            verbose=0
        )
    if os.path.exists("modelo.weights.h5"):
        os.remove("modelo.weights.h5")
    st.session_state.model_trained = True
    st.session_state.intent_model.save_weights("modelo.weights.h5")

if os.path.exists("modelo.weights.h5"):
    st.session_state.intent_model = create_intent_model(embedding_matrix, vocab_size, embedding_dim, max_length)
    st.session_state.intent_model.load_weights("modelo.weights.h5")

# Gerar previsões para métricas
y_pred = st.session_state.intent_model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

##############################################
# MODELO DE SENTIMENTO (dados do tw_pt.csv)
##############################################

@st.cache_data
def load_tw_data():
    df = pd.read_csv("tw_pt.csv")
    df['Text'] = df['Text'].apply(preprocess_text)
    return df

tw_data = load_tw_data()
X_tw = tw_data['Text']
y_tw = tw_data['Classificacao']

label_encoder_tw = LabelEncoder()
y_tw_encoded = label_encoder_tw.fit_transform(y_tw)

if 'tokenizer_tw' not in st.session_state:
    st.session_state.tokenizer_tw = Tokenizer()
    st.session_state.tokenizer_tw.fit_on_texts(X_tw)
vocab_size_tw = len(st.session_state.tokenizer_tw.word_index) + 1
max_length_tw = 100
X_tw_seq = st.session_state.tokenizer_tw.texts_to_sequences(X_tw)
X_tw_pad = pad_sequences(X_tw_seq, maxlen=max_length_tw, padding='post')

def create_tw_sentiment_model(vocab_size, embedding_dim, max_length, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),  # Removido input_length
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if 'sentiment_model_tw' not in st.session_state:
    num_classes_tw = len(label_encoder_tw.classes_)
    st.session_state.sentiment_model_tw = create_tw_sentiment_model(vocab_size_tw, embedding_dim, max_length_tw, num_classes_tw)
    epochs_tw = 5
    progress_callback_tw = ProgressBarCallback(total_epochs=epochs_tw)
    with st.spinner("Treinando modelo de sentimento..."):
        st.session_state.sentiment_model_tw.fit(
            X_tw_pad, 
            y_tw_encoded, 
            epochs=epochs_tw, 
            batch_size=32, 
            validation_split=0.1, 
            callbacks=[progress_callback_tw],
            verbose=0
        )
    st.session_state.sentiment_model_tw.save_weights("sentiment_model_tw.weights.h5")

if os.path.exists("sentiment_model_tw.weights.h5"):
    st.session_state.sentiment_model_tw = create_tw_sentiment_model(vocab_size_tw, embedding_dim, max_length_tw, len(label_encoder_tw.classes_))
    st.session_state.sentiment_model_tw.build(input_shape=(None, max_length_tw))
    st.session_state.sentiment_model_tw.load_weights("sentiment_model_tw.weights.h5")

def get_sentiment_tw(text):
    """Utiliza o modelo de sentimento treinado com tw_pt.csv para classificar o texto."""
    processed = preprocess_text(text)
    seq = st.session_state.tokenizer_tw.texts_to_sequences([processed])
    pad_seq = pad_sequences(seq, maxlen=max_length_tw, padding='post')
    pred = st.session_state.sentiment_model_tw.predict(pad_seq)
    label = label_encoder_tw.inverse_transform([np.argmax(pred, axis=1)[0]])[0]
    return label

##############################################
# CHATBOT RESPONSE
##############################################

def detect_and_translate(text, target_lang='pt'):
    try:
        detected_lang = detect(text)
    except Exception as e:
        detected_lang = target_lang
    translated_text = GoogleTranslator(source=detected_lang, target=target_lang).translate(text) if detected_lang != target_lang else text
    return translated_text

# Inicializa a pipeline conversacional
if 'conversational_pipeline' not in st.session_state:
    st.session_state.conversational_pipeline = pipeline(
        "conversational", 
        model="microsoft/DialoGPT-medium", 
        framework="pt" 
    )

def chatbot_response(user_input):
    
    translated_input = detect_and_translate(user_input)
    
    
    blob = TextBlob(translated_input)
    polarity = blob.sentiment.polarity  # valor entre -1 (muito negativo) e 1 (muito positivo)
    if polarity < -0.2:
        sentiment_label = "negativo"
    elif polarity > 0.2:
        sentiment_label = "positivo"
    else:
        sentiment_label = "neutro"
    
    # Pré-processamento e preparação para previsão de intenção
    processed_input = preprocess_text(translated_input)
    input_seq = st.session_state.tokenizer.texts_to_sequences([processed_input])
    input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')
    
    # Previsão da intenção com o modelo LSTM treinado
    with st.spinner("Processando intenção..."):
        intent_pred = st.session_state.intent_model.predict(input_pad)
    intent_class = np.argmax(intent_pred, axis=1)
    intent_label = label_encoder.inverse_transform(intent_class)[0]
    
    # Intenção com regras baseadas em palavras-chave
    input_lower = translated_input.lower()
    if any(s in input_lower for s in ["olá", "oi", "bom dia", "boa tarde", "boa noite"]):
        intent_label = "saudação"
    elif "nome" in input_lower:
        intent_label = "nome"
    elif any(s in input_lower for s in ["ajuda", "suporte", "preciso de ajuda"]):
        intent_label = "ajuda"
    elif any(s in input_lower for s in ["tudo bem", "como vai", "como está"]):
        intent_label = "saudação"
    
    # Definição de templates de resposta para cada intenção
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
    
    # Seleção da resposta baseada na intenção, sentimento e contexto
    if intent_label in response_templates:
        if st.session_state.chat_history:
            if sentiment_label in ["negativo", "raiva", "triste"]:
                response = "Percebo que você está passando por um momento difícil. Estou aqui para ajudar. Pode me contar mais?"
            elif sentiment_label in ["positivo", "felicidade", "elogio"]:
                response = random.choice(response_templates.get("felicidade", ["Que bom!"]))
            else:
                response = random.choice(response_templates[intent_label])
        else:
            response = random.choice(response_templates[intent_label])
    else:
        response = "Poderia me explicar melhor? Quero entender para poder ajudar!"
    
    # Fallback com modelo de conversação se a resposta for muito genérica
    if response.startswith("Poderia me explicar melhor"):
        with st.spinner("Gerando resposta..."):
            conversation = Conversation(translated_input)
            result = st.session_state.conversational_pipeline(conversation)
            # Usa a última resposta gerada pelo modelo de conversação
            response = result.generated_responses[-1]
    
    return response, sentiment_label
##############################################
# INTERFACE DO CHAT
##############################################

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
    with st.spinner("Processando sua mensagem..."):
        response, sentiment = chatbot_response(user_input)
        
    # Adicionando a mensagem do usuário e a resposta do chatbot ao histórico
    st.session_state.chat_history.append(("Você", user_input))
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

##############################################
# SIDEBAR COM MÉTRICAS E ANÁLISES
##############################################

st.sidebar.header("Análises e Métricas")
st.sidebar.subheader("Desempenho do Modelo de Intenção")
st.sidebar.write("Accuracy:", accuracy_score(y_test, y_pred_labels))
st.sidebar.write("Classification Report:")
st.sidebar.text(classification_report(y_test, y_pred_labels))

st.sidebar.subheader("Sentimento da Última Mensagem")
st.sidebar.write(f"{st.session_state.last_sentiment}")

st.sidebar.subheader("Evolução do Sentimento")
# Extrai os sentimentos das mensagens do usuário utilizando o modelo tw_pt
sentiments = [get_sentiment_tw(msg[1]) for msg in st.session_state.chat_history if msg[0] == "Você"]
if sentiments:
    # Converte os rótulos para pontuações para plot (ex.: Positivo: 1, Neutro: 0, Negativo: -1)
    sentiment_scores = [1 if s.lower() in ["positivo", "felicidade", "elogio"] else -1 if s.lower() in ["negativo", "raiva", "triste"] else 0 for s in sentiments]
    fig, ax = plt.subplots()
    sns.lineplot(x=range(len(sentiment_scores)), y=sentiment_scores, ax=ax, marker='o')
    ax.set_title("Evolução do Sentimento")
    ax.set_xlabel("Mensagens")
    ax.set_ylabel("Pontuação de Sentimento")
    st.sidebar.pyplot(fig)

if st.sidebar.button("Limpar Histórico do Chat"):
    st.session_state.chat_history = []
    st.rerun()

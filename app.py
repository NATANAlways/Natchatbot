import streamlit as st
from chatbot_main import predict_class, get_response, intents

st.title("💬 Simple ChatBot")

user_input = st.text_input("You: ")

if user_input:
    intent_prediction = predict_class(user_input)
    response = get_response(intent_prediction, intents)
    st.text_area("Bot:", value=response, height=100)

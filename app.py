import streamlit as st
from transformers import pipeline

text = st.text_area('enter some text:')
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

if text:
  out = classifier(text)
  st.json(out)
import my_gradio as gr
import joblib
import logging

# Carregar os modelos a partir dos arquivos
loaded_language_model = joblib.load('language_model.joblib')
loaded_sentiment_model = joblib.load('sentiment_model.joblib')  # Corrigir o nome do modelo, se necessário

logging.info("função Gradio:".upper())
def show_interface_gradio(input):
    language = loaded_language_model.predict([input])[0]
    sentiment = loaded_sentiment_model.predict([input])[0]
    output = f'language: {language}, sentiment: {sentiment}'
    logging.info(f'output = language: {language}, sentiment: {sentiment}'.upper())
    return output

# Crie a interface Gradio
iface = gr.Interface(fn=show_interface_gradio, inputs='textbox', outputs='textbox', live=False)

# Inicie a interface Gradio
iface.launch()
      
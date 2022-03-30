import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from nltk.corpus   import stopwords
import re
from unidecode import unidecode
from nltk.corpus import stopwords
import pandas as pd
import pickle
import nltk
from PIL import Image

#Diseño de la pagina
nltk.download('stopwords')


st.header("Predicción si la duda/queja pertenece al equipo de token o de transferencias")
st.subheader('''**Intrucciones de uso:** ''')
st.markdown('''Tienen que presionar el boton "Decir duda" y decir la duda que se tenga, 
            el programa solo esperara a que dejes de hablar para hacer la prediccion, por lo que, procura no hacer pausas largas 
            y hablar lo mas claro y cerca del microfono.''')

st.markdown('''
Puedes intentar con:
* Tengo problemas con mi token
* No puedo hacer transferencias
* Tengo mas de 2 horas que hice una transferencia y no se refleja. En mis otras apps de otros bancos jámas pasa esto
* Quiero hacer transferencias a otros bancos y nisiquiera aparecen activadas las cuentas para transferir. Ya cuento con el súper token y ni así puede realizar el movimiento.
''')



# Cargamos funciones para limpiar texto
def remove_accents(a):
    return unidecode(a)

def clean_re(txt):
    # Pasamos a minusculas
    txt = txt.lower()
    # Elimina caracteres especiales de tipo \t\n\r\f\v
    txt = re.sub(r'[^\w\s]',' ',txt)
    # Elimina palabras con menos de tres letras
    txt = ' '.join([w for w in txt.split() if len(w)>3])
    # Elimina números
    txt = re.sub(r'\b\d+(?:\.\d+)?\s+', ' ', txt)
    return txt

def remove_stopwords(txt):
    txt = pd.Series(txt).apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
    return txt

# Cargamos el vectorizer y el modelo
vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
modelo = pickle.load(open("logistic_model.pickle", "rb"))

# Cargamos el stop words
stop_words = list(set(stopwords.words('spanish')))
stop_words.append("algo")
stop_words.append('ambos')

# Ejemplo inicial
# text_predict = 'tengo un problema con el token'

# Grabamos el audio cuando den click en el boton
stt_button = Button(label="Decir duda", width=100)

stt_button.js_on_event("button_click", CustomJS(code="""
                        var recognition = new webkitSpeechRecognition();
                        recognition.continuous = false;
                        recognition.interimResults = true;
                    
                        recognition.onresult = function (e) {
                            var value = "";
                            for (var i = e.resultIndex; i < e.results.length; ++i) {
                                if (e.results[i].isFinal) {
                                    value += e.results[i][0].transcript;
                                }
                            }
                            if ( value != "") {
                                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                            }
                        }
                        recognition.start();
                        """))

result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=True,
    override_height=75,
    debounce_time=10)

if result:
    if "GET_TEXT" in result:
        text_predict = result.get("GET_TEXT")
        st.write("**Has dicho:** *{0}*".format(result.get("GET_TEXT")))
        # Limpiamos el texto
        text_predict = clean_re(text_predict)
        text_predict = remove_accents(text_predict)
        text_predict = remove_stopwords(text_predict)
        # Vectorizamos el texto
        x_predict = pd.DataFrame(vectorizer.transform(text_predict).toarray(),columns = vectorizer.get_feature_names())
        # Hacemos la prediccion bruta y la de con probabilidad
        prediccion = modelo.predict(x_predict)
        prediccion_proba = modelo.predict_proba(x_predict)
        proba_token = round(prediccion_proba[0][1]*100,2)
        proba_transacciones = round(prediccion_proba[0][0]*100,2)
        # Tomamos una decision
        st.write("Probabilidad de que su queja sea del equipo encargado del **token: {0}%**".format(proba_token))
        st.write("Probabilidad de que su queja sea del equipo encargado de las **transferencias: {0}%**".format(proba_transacciones))
        if prediccion == 1:
            st.write("**Su llamada sera transferida al equipo encargado del token**")
        else:
            st.write("**Su llamada sera transferida al equipo encargado de transferencias**")
        image = Image.open('gatito_formal.jpg',caption = 'Gerente')

st.write("**Elaborado por Carlos Eduardo Vázquez Chong**")
st.write("Para entrega de examen final del modulo 2 del diplomado de 'Ciencia de Datos'")
# Ayudas
# https://discuss.streamlit.io/t/speech-to-text-on-client-side-using-html5-and-streamlit-bokeh-events/7888/15
# https://stackoverflow.com/questions/32764991/how-do-i-store-a-tfidfvectorizer-for-future-use-in-scikit-learn
# https://stackoverflow.com/questions/68775869/support-for-password-authentication-was-removed-please-use-a-personal-access-to
# https://www.youtube.com/watch?v=kXvmqg8hc70
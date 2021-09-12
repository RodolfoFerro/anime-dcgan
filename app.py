import streamlit as st

from gan import build_dcgan
from gan import generate_faces


dcgan = build_dcgan()
dcgan.generator.load_weights('models/generator_weights.h5')
dcgan.discriminator.load_weights('models/discriminator_weights.h5')


st.markdown(
    """
    # Crea tu waifu
    
    **Modelos generativos para la creación de personajes de anime**

    Con esta sencilla aplicación web podrás crear tu waifu con un modelo
    generativo.

    El modelo generativo se ha entrenado utilizando el conjunto de datos 
    de Kaggle: [https://www.kaggle.com/soumikrakshit/anime-faces]
    (https://www.kaggle.com/soumikrakshit/anime-faces)

    Para generar un conjunto de caras, sólo pulsa el siguiente botón:
    """
)

if st.button("Crear waifu"):
    fig = generate_faces(dcgan)
    st.pyplot(fig)
    st.write('Imagen generada por una DCGAN.')
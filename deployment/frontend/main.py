import streamlit as st
import requests
import numpy as np
import json
from skimage.transform import resize
from PIL import Image


# image = Image.open('gala-apples256.jpg')
# numpy_d = np.asarray(image)
# print(type(image))

# Load page
def run():
    # widget input
    with st.form(key='form_parameters'):
        st.title("Apple or Orange")
        uploaded_file = st.file_uploader("Choose a file", type=['jpg','jpeg'])
        
        st.markdown('---')
        submitted = st.form_submit_button('Predict')
    
    if submitted:
        image = Image.open(uploaded_file)
        np_img = np.asarray(image)
        resized = resize(np_img, (256,256),anti_aliasing=True)
        # st.write(resized.shape)
        x = np.expand_dims(resized, axis=0)
        images = np.vstack([x])
        img_list = images.tolist()
        # model input
        input_data_json = json.dumps({
            'signature_name': 'serving_default',
            'instances': img_list
        })

        # inference
        URL = "https://tf-serving-backend-cv-sam.herokuapp.com/v1/models/cv_model:predict"

        r = requests.post(URL, data=input_data_json)

        if r.status_code == 200:
            res = r.json()
            # st.write(res['predictions'][0][0])
            if res['predictions'][0][0] <= 0.5:
                st.write('Apple')
            else:
                st.write('Orange')
        else:
            st.write('Error, reload page or retry predictions')
        
        st.image(resized)
    

if __name__ == '__main__':
    run()
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import json

app = Flask(__name__)
model = tf.keras.models.load_model('trained_model2.keras')
disease_info = {
    'Apple___Apple_scab': {
        'cause': 'Caused by the fungus *Venturia inaequalis*.',
        'precautions': 'Use resistant varieties, apply fungicide, and remove fallen leaves.'
    },
    'Apple___Black_rot': {
        'cause': 'Caused by the fungus *Botryosphaeria obtusa*.',
        'precautions': 'Prune infected branches and use fungicidal sprays.'
    },
    'Apple___Cedar_apple_rust': {
        'cause': 'Caused by the fungus *Gymnosporangium juniperi-virginianae*.',
        'precautions': 'Remove nearby junipers and use fungicide during early spring.'
    },
    'Apple___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Maintain good care and monitor regularly.'
    },
    'Blueberry___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Ensure proper sunlight and watering.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'cause': 'Caused by the fungus *Podosphaera clandestina*.',
        'precautions': 'Apply sulfur fungicides and improve air circulation.'
    },
    'Cherry_(including_sour)___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Regular pruning and disease checks recommended.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'cause': 'Caused by the fungus *Cercospora zeae-maydis*.',
        'precautions': 'Rotate crops and use resistant hybrids.'
    },
    'Corn_(maize)___Common_rust_': {
        'cause': 'Caused by *Puccinia sorghi* fungus.',
        'precautions': 'Use resistant varieties and apply fungicides if severe.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'cause': 'Caused by *Exserohilum turcicum*.',
        'precautions': 'Practice crop rotation and remove crop debris.'
    },
    'Corn_(maize)___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Continue regular crop care and monitoring.'
    },
    'Grape___Black_rot': {
        'cause': 'Caused by the fungus *Guignardia bidwellii*.',
        'precautions': 'Prune infected vines and apply fungicide early in the season.'
    },
    'Grape___Esca_(Black_Measles)': {
        'cause': 'Complex disease involving several fungi.',
        'precautions': 'Remove affected vines and avoid stress factors.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'cause': 'Caused by *Isariopsis clavispora*.',
        'precautions': 'Use fungicides and maintain canopy airflow.'
    },
    'Grape___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Maintain good air flow and prevent fungal infections.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'cause': 'Caused by *Candidatus Liberibacter* and spread by psyllids.',
        'precautions': 'Control psyllids and remove infected trees.'
    },
    'Peach___Bacterial_spot': {
        'cause': 'Caused by *Xanthomonas campestris* bacteria.',
        'precautions': 'Use copper sprays and plant resistant cultivars.'
    },
    'Peach___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Proper irrigation and pest management.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'cause': 'Caused by *Xanthomonas campestris pv. vesicatoria*.',
        'precautions': 'Use certified seed and copper-based sprays.'
    },
    'Pepper,_bell___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Keep soil well-drained and fertilized.'
    },
    'Potato___Early_blight': {
        'cause': 'Caused by *Alternaria solani* fungus.',
        'precautions': 'Remove infected leaves and apply fungicides.'
    },
    'Potato___Late_blight': {
        'cause': 'Caused by *Phytophthora infestans*.',
        'precautions': 'Avoid overhead watering, remove infected plants, and apply fungicides.'
    },
    'Potato___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Regular inspection and use certified seeds.'
    },
    'Raspberry___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Keep canes pruned and spacing adequate.'
    },
    'Soybean___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Rotate crops and monitor regularly.'
    },
    'Squash___Powdery_mildew': {
        'cause': 'Caused by *Podosphaera xanthii* or *Erysiphe cichoracearum*.',
        'precautions': 'Use resistant varieties and apply fungicide at first sign.'
    },
    'Strawberry___Leaf_scorch': {
        'cause': 'Caused by the fungus *Diplocarpon earlianum*.',
        'precautions': 'Remove infected leaves and apply fungicides.'
    },
    'Strawberry___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Ensure good drainage and mulch use.'
    },
    'Tomato___Bacterial_spot': {
        'cause': 'Caused by *Xanthomonas campestris pv. vesicatoria*.',
        'precautions': 'Use certified seeds and apply copper-based sprays.'
    },
    'Tomato___Early_blight': {
        'cause': 'Caused by *Alternaria solani*.',
        'precautions': 'Use mulch and apply protective fungicides.'
    },
    'Tomato___Late_blight': {
        'cause': 'Caused by *Phytophthora infestans*.',
        'precautions': 'Avoid overhead watering, remove infected plants, and apply fungicides.'
    },
    'Tomato___Leaf_Mold': {
        'cause': 'Caused by *Passalora fulva*.',
        'precautions': 'Improve air circulation and use fungicide sprays.'
    },
    'Tomato___Septoria_leaf_spot': {
        'cause': 'Caused by *Septoria lycopersici* fungus.',
        'precautions': 'Remove infected leaves and apply fungicide.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'cause': 'Caused by *Tetranychus urticae* infestation.',
        'precautions': 'Spray miticides and increase humidity.'
    },
    'Tomato___Target_Spot': {
        'cause': 'Caused by *Corynespora cassiicola* fungus.',
        'precautions': 'Avoid leaf wetness and use fungicides.'
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'cause': 'Caused by *Begomovirus* and spread by whiteflies.',
        'precautions': 'Use resistant varieties and control whiteflies.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'cause': 'Caused by *Tomato mosaic virus (ToMV)*.',
        'precautions': 'Disinfect tools and avoid infected seeds.'
    },
    'Tomato___healthy': {
        'cause': 'No disease detected.',
        'precautions': 'Maintain regular watering and good plant hygiene.'
    }
}

# Class Names
class_names = ['Apple___Apple_scab',
    'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Prediction Function
def model_prediction(image_path):
    image = Image.open(image_path).resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    predicted_disease = class_names[result_index]
    
    # Get cause and precautions
    info = disease_info.get(predicted_disease, {
        'cause': 'Information not available.',
        'precautions': 'Please consult agricultural expert.'
    })
    
    return predicted_disease, info['cause'], info['precautions']


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    cause = None
    precautions = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join('static', filename)
            file.save(image_path)
            prediction_result, cause, precautions = model_prediction(image_path)

    return render_template(
        'predict.html',
        prediction=prediction_result,
        cause=cause,
        precautions=precautions,
        image_path=image_path
    )
  
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        
        experience = request.form.get('experience')

        feedback_data = {
            
            "experience": experience
        }

        feedback_file = 'feedback.json'

        # If file doesn't exist, create an empty list
        if not os.path.exists(feedback_file):
            with open(feedback_file, 'w') as f:
                json.dump([], f)

        # Read, append, and save the new feedback
        with open(feedback_file, 'r+') as f:
            data = json.load(f)
            data.append(feedback_data)
            f.seek(0)
            json.dump(data, f, indent=4)

        return render_template('feedback.html', message="Thank you! Your feedback has been submitted.")

    return render_template('feedback.html')


if __name__ == '__main__':
    app.run(debug=True)

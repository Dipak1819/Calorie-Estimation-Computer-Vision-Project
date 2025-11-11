"""
Flask Backend for Food Calorie Estimation App
Uses trained .h5 model to predict food items and return calorie information
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = 'best_food_model.h5'
model = None

# Food categories (101 classes from Food-101 dataset)
FOOD_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
    'waffles'
]

# Approximate calorie information per serving (in kcal)
CALORIE_DATABASE = {
    'apple_pie': 296, 'baby_back_ribs': 360, 'baklava': 334, 'beef_carpaccio': 175, 'beef_tartare': 220,
    'beet_salad': 150, 'beignets': 280, 'bibimbap': 560, 'bread_pudding': 310, 'breakfast_burrito': 450,
    'bruschetta': 120, 'caesar_salad': 350, 'cannoli': 250, 'caprese_salad': 200, 'carrot_cake': 435,
    'ceviche': 140, 'cheesecake': 321, 'cheese_plate': 250, 'chicken_curry': 400, 'chicken_quesadilla': 480,
    'chicken_wings': 320, 'chocolate_cake': 352, 'chocolate_mousse': 270, 'churros': 237, 'clam_chowder': 180,
    'club_sandwich': 590, 'crab_cakes': 200, 'creme_brulee': 340, 'croque_madame': 560, 'cup_cakes': 305,
    'deviled_eggs': 130, 'donuts': 290, 'dumplings': 220, 'edamame': 120, 'eggs_benedict': 450,
    'escargots': 180, 'falafel': 330, 'filet_mignon': 290, 'fish_and_chips': 585, 'foie_gras': 462,
    'french_fries': 365, 'french_onion_soup': 220, 'french_toast': 340, 'fried_calamari': 330, 'fried_rice': 370,
    'frozen_yogurt': 160, 'garlic_bread': 280, 'gnocchi': 250, 'greek_salad': 180, 'grilled_cheese_sandwich': 380,
    'grilled_salmon': 280, 'guacamole': 160, 'gyoza': 230, 'hamburger': 530, 'hot_and_sour_soup': 100,
    'hot_dog': 290, 'huevos_rancheros': 380, 'hummus': 170, 'ice_cream': 270, 'lasagna': 390,
    'lobster_bisque': 250, 'lobster_roll_sandwich': 436, 'macaroni_and_cheese': 400, 'macarons': 90, 'miso_soup': 84,
    'mussels': 172, 'nachos': 550, 'omelette': 240, 'onion_rings': 410, 'oysters': 85,
    'pad_thai': 430, 'paella': 450, 'pancakes': 340, 'panna_cotta': 310, 'peking_duck': 405,
    'pho': 450, 'pizza': 285, 'pork_chop': 290, 'poutine': 740, 'prime_rib': 420,
    'pulled_pork_sandwich': 480, 'ramen': 436, 'ravioli': 350, 'red_velvet_cake': 430, 'risotto': 376,
    'samosa': 252, 'sashimi': 120, 'scallops': 140, 'seaweed_salad': 70, 'shrimp_and_grits': 450,
    'spaghetti_bolognese': 400, 'spaghetti_carbonara': 480, 'spring_rolls': 140, 'steak': 340, 'strawberry_shortcake': 330,
    'sushi': 200, 'tacos': 210, 'takoyaki': 180, 'tiramisu': 450, 'tuna_tartare': 190,
    'waffles': 320
}

def load_model():
    """Load the trained .h5 model"""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file '{MODEL_PATH}' not found. Please train the model first.")
        return False

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image_bytes):
    """
    Preprocess image for model prediction
    Matches the preprocessing used during training
    """
    # Load image from bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to model input size (224x224)
    img = img.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(img)

    # Expand dimensions to create batch of 1
    img_array = np.expand_dims(img_array, axis=0)

    # Apply MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    return img_array

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict food item and return calorie information
    Expects: multipart/form-data with 'image' file
    Returns: JSON with prediction results
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure best_food_model.h5 exists.'
        }), 500

    # Check if image was provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Read image bytes
        image_bytes = file.read()

        # Preprocess image
        processed_image = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)

        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]

        results = []
        for idx in top_5_indices:
            food_name = FOOD_CLASSES[idx]
            confidence = float(predictions[0][idx])
            calories = CALORIE_DATABASE.get(food_name, 0)

            results.append({
                'food_name': food_name.replace('_', ' ').title(),
                'confidence': round(confidence * 100, 2),
                'calories': calories,
                'serving_info': '100g serving'
            })

        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': results[0]
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("Starting Food Calorie Estimation App...")
    print("=" * 50)

    # Load model on startup
    model_loaded = load_model()

    if not model_loaded:
        print("\n⚠️  WARNING: Model not loaded!")
        print("Please ensure 'best_food_model.h5' exists in the project directory.")
        print("Train the model using main.ipynb first.")
        print("\nThe app will start but predictions won't work until model is available.\n")

    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)

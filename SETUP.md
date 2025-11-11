# Food Calorie Estimator - Setup Guide

This application uses a trained deep learning model to identify food items from images and provide calorie information.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (First Time Only)

Before running the app, you need to train the model using the provided Jupyter notebook:

1. Open `main.ipynb` in Jupyter Notebook or JupyterLab
2. Update the `image_dir` path to point to your Food-101 dataset
3. Run all cells to train the model
4. The model will be saved as `best_food_model.h5` in the project directory

**Note:** Training takes several hours depending on your hardware. The notebook will automatically save the best model during training.

### 3. Verify Model File

Ensure `best_food_model.h5` exists in the project root directory:

```bash
ls -lh best_food_model.h5
```

## Running the Application

### Start the Flask Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## Using the Application

1. **Upload Image**: Click or drag-and-drop a food image
2. **Analyze**: Click "Analyze Food & Get Calories"
3. **View Results**: See the predicted food item, confidence score, and calorie information
4. **Try Another**: Upload another image to analyze

## Features

- **Image Recognition**: Identifies 101 different food items
- **Calorie Information**: Provides approximate calorie count per serving
- **Top 5 Predictions**: Shows alternative possibilities with confidence scores
- **Modern UI**: Clean, responsive interface with drag-and-drop support
- **Real-time Processing**: Fast predictions using MobileNetV2

## Supported Food Categories (101 items)

The model can recognize foods including:
- Desserts: apple pie, cheesecake, chocolate cake, tiramisu, etc.
- Main dishes: pizza, hamburger, sushi, steak, lasagna, etc.
- Asian cuisine: ramen, pho, pad thai, sushi, gyoza, etc.
- Appetizers: bruschetta, spring rolls, nachos, hummus, etc.
- And many more!

## API Endpoints

### POST /api/predict
Upload an image to get food prediction and calorie information.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "food_name": "Pizza",
      "confidence": 95.67,
      "calories": 285,
      "serving_info": "100g serving"
    },
    ...
  ],
  "top_prediction": { ... }
}
```

### GET /api/health
Check if the server and model are running.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Troubleshooting

### Model Not Found Error
If you see "Model not loaded" error:
1. Ensure `best_food_model.h5` exists in the project directory
2. Train the model using `main.ipynb` first
3. Check file permissions

### Low Accuracy
If predictions are inaccurate:
- Use clear, well-lit food images
- Ensure the food item is centered and visible
- Try images from different angles
- The model works best with the 101 trained food categories

### Memory Issues
If you encounter memory errors:
- Close other applications
- Reduce batch size in training
- Use a machine with more RAM (8GB+ recommended)

## Model Information

- **Architecture**: MobileNetV2 (transfer learning)
- **Input Size**: 224x224 pixels
- **Classes**: 101 food categories
- **Dataset**: Food-101
- **Accuracy**: ~42-45% (can be improved with fine-tuning)
- **Top-5 Accuracy**: ~85-90%

## Performance Tips

- Use GPU for faster predictions (automatically detected by TensorFlow)
- Images are automatically resized to 224x224
- Supported formats: JPG, PNG, JPEG

## Credits

- Model: MobileNetV2 pretrained on ImageNet
- Dataset: Food-101
- Framework: TensorFlow/Keras
- Web Framework: Flask

## License

This project is for educational purposes. Please ensure you have the rights to use any images you upload.

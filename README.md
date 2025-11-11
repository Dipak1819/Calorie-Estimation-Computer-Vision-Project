# Calorie-Estimation-Computer-Vision-Project

A deep learning web application that identifies food items from images and provides calorie information using computer vision.

## Features

- **Food Recognition**: Identifies 101 different food categories using a trained MobileNetV2 model
- **Calorie Estimation**: Provides approximate calorie information for detected food items
- **User-Friendly Interface**: Modern web UI with drag-and-drop image upload
- **Real-Time Predictions**: Fast inference using TensorFlow
- **Top-5 Predictions**: Shows alternative food possibilities with confidence scores

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model** (first time only):
   - Open `main.ipynb` and follow the training steps
   - This will create `best_food_model.h5`

3. **Run the web app:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   - Navigate to `http://localhost:5000`
   - Upload a food image and get calorie information!

## Project Structure

```
.
├── app.py              # Flask backend server
├── main.ipynb          # Model training notebook
├── templates/
│   └── index.html      # Frontend web interface
├── requirements.txt    # Python dependencies
├── SETUP.md           # Detailed setup guide
└── best_food_model.h5 # Trained model (generated after training)
```

## How It Works

1. **Image Upload**: User uploads a food image through the web interface
2. **Preprocessing**: Image is resized to 224x224 and preprocessed for MobileNetV2
3. **Prediction**: Model predicts the food category with confidence scores
4. **Calorie Lookup**: Calorie information is retrieved from the database
5. **Results Display**: Top prediction and alternatives are shown to the user

## Model Details

- **Architecture**: MobileNetV2 (transfer learning from ImageNet)
- **Dataset**: Food-101 (101 food categories, 100 images each)
- **Input Size**: 224x224 RGB images
- **Training**: 70/30 train-test split with data augmentation
- **Accuracy**: ~42-45% on test set, ~85-90% top-5 accuracy

## Supported Food Categories (101)

Pizza, Hamburger, Sushi, Steak, Ice Cream, French Fries, Chocolate Cake, Cheesecake, Tacos, Ramen, Pad Thai, and 90+ more!

## Documentation

See [SETUP.md](SETUP.md) for detailed installation and usage instructions.

## API Endpoints

- `POST /api/predict` - Upload image and get prediction
- `GET /api/health` - Check server status

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- Flask 3.0+
- 8GB+ RAM recommended
- GPU optional but recommended for training

## License

This project is for educational purposes.

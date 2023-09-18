
---

# 🌲 Forest Cover Prediction Application 🌲

![Forest Cover](https://www.uncovercolorado.com/wp-content/uploads/2021/02/roosevelt-national-forest-nederland-colorado-1600x800-1-1600x800.jpg)

This repository contains a machine learning prediction application for forest cover types. The application is built using FlaskAPI and is further enhanced with a Swagger API for better documentation and testing. The entire setup is containerized using Docker for easy deployment and scalability.

## 🌟 Features

- **🤖 Machine Learning Model**: Predicts forest cover types based on various environmental features.
- **🌐 FlaskAPI**: A lightweight web server interface for the application.
- **📖 Swagger API**: Provides an interactive UI for testing and documenting the API endpoints.
- **🐳 Docker**: Containerizes the application for consistent deployment.

## 📚 Dataset Description

The study area encompasses four wilderness areas in the Roosevelt National Forest of northern Colorado. Each observation represents a 30m x 30m patch. The goal is to predict an integer classification for the forest cover type.

### 🌳 Forest Cover Types:

1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

### 📊 Data Fields:

- **Elevation**: Elevation in meters
- **Aspect**: Aspect in degrees azimuth
- **Slope**: Slope in degrees
- **Horizontal_Distance_To_Hydrology**: Horizontal distance to the nearest surface water features
- **Vertical_Distance_To_Hydrology**: Vertical distance to the nearest surface water features
- **Horizontal_Distance_To_Roadways**: Horizontal distance to the nearest roadway
- **Hillshade_9am**: Hillshade index at 9am during the summer solstice (0 to 255 index)
- **Hillshade_Noon**: Hillshade index at noon during the summer solstice (0 to 255 index)
- **Hillshade_3pm**: Hillshade index at 3pm during the summer solstice (0 to 255 index)
- **Horizontal_Distance_To_Fire_Points**: Horizontal distance to the nearest wildfire ignition points
- **Wilderness_Area**: Wilderness area designation (4 binary columns, 0 = absence or 1 = presence)
- **Soil_Type**: Soil type designation (40 binary columns, 0 = absence or 1 = presence)
- **Cover_Type**: Forest cover type designation (7 types, integers 1 to 7)

### 🌄 Wilderness Areas:

1. Rawah Wilderness Area
2. Neota Wilderness Area
3. Comanche Peak Wilderness Area
4. Cache la Poudre Wilderness Area






## 🚀 Getting Started

### 🛠 Prerequisites

- Python 3.x 🐍
- Docker 📦 (if you wish to run the application in a container)

### 📥 Installation

1. **🔗 Clone the Repository**:
   ```bash
   git clone https://github.com/ScientificArchisman/iNeuron-internship-ForestCover_Prediction.git
   cd iNeuron-internship-ForestCover_Prediction
   ```

2. **🌐 Set Up a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **📦 Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 🏃 Running the Application

- **Using Flask**:
  ```bash
  python app.py
  ```

- **Using Docker**:
  ```bash
  docker build -t forestcover-prediction .
  docker run -p 5000:5000 forestcover-prediction
  ```

Visit `http://localhost:5000` in your browser to access the application and Swagger UI.

## 🤝 Contributing

Feel free to fork this repository, make changes, and submit pull requests. Any contributions, big or small, are greatly appreciated!

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

The image used is a placeholder from Unsplash. You can replace it with any relevant image or graphic that represents your project. Emojis can help make the README more engaging and visually appealing.
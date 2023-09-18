### Forest Cover Prediction Application 🌲

![Forest Cover](https://www.uncovercolorado.com/wp-content/uploads/2021/02/roosevelt-national-forest-nederland-colorado-1600x800-1-1600x800.jpg)

The Forest Cover Prediction Application is a machine learning-based tool designed to predict forest cover types based on various environmental features. The application is hosted on GitHub and can be accessed [here](https://github.com/ScientificArchisman/iNeuron-internship-ForestCover_Prediction).

#### Overview:
- **Repository Link**: [Forest Cover Prediction Application](https://github.com/ScientificArchisman/iNeuron-internship-ForestCover_Prediction)
- **Application Type**: Machine Learning Prediction
- **Deployment**: FlaskAPI with Docker containerization
- **Documentation**: Swagger API

#### Features:
- **Machine Learning Model**: Predicts forest cover types based on environmental features.
- **FlaskAPI**: Provides a lightweight web server interface.
- **Swagger API**: Offers an interactive UI for testing and documenting the API endpoints.
- **Docker**: Ensures consistent deployment through containerization.

#### Dataset Description:
The dataset represents the Roosevelt National Forest of northern Colorado, covering four wilderness areas. Each observation corresponds to a 30m x 30m patch, and the objective is to predict the forest cover type.

**Forest Cover Types**:
1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

**Data Fields**:
- Elevation, Aspect, Slope
- Distances to hydrology, roadways, and fire points
- Hillshade indices at different times
- Wilderness area and soil type designations
- Forest cover type

**Wilderness Areas**:
1. Rawah Wilderness Area
2. Neota Wilderness Area
3. Comanche Peak Wilderness Area
4. Cache la Poudre Wilderness Area

#### Getting Started:
**Prerequisites**:
- Python 3.x
- Docker (for containerized deployment)

**Installation**:
1. Clone the repository.
2. Set up a virtual environment (optional but recommended).
3. Install the required dependencies from the `requirements.txt` file.

**Running the Application**:
- Using Flask: Execute `python app.py`.
- Using Docker: Build and run the Docker container.

#### Contributing:
Contributions are welcome! Feel free to fork the repository, make changes, and submit pull requests.

#### License:
The project is licensed under the MIT License.

---

**Note**: The image used in the repository is a placeholder from Unsplash. It can be replaced with any relevant image or graphic that represents the project. Emojis have been used to make the documentation more engaging and visually appealing.

---

You are currently on the free plan which is significantly limited by the number of requests. To increase your quota, you can check available plans [here](https://c7d59216ee8ec59bda5e51ffc17a994d.auth.portal-pluginlab.ai/pricing).
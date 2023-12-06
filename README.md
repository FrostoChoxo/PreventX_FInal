# PreventX_FInal
Safety Monitoring Web Application
Overview
This GitHub repository contains a web application designed for safety monitoring in industrial environments. The project includes features such as:

Detection of individuals without hardhats and vests
Counting the number of persons in a designated zone
Audio threshold detection in the environment
Additionally, the repository includes a separate module for forklift avoidance, where cameras mounted on the front and back of a forklift detect the presence of individuals in close proximity. When a person is detected, an alert sound is triggered, and the forklift stops until the situation is deemed safe.

Getting Started
Prerequisites
Before running the application, ensure you have the following installed:

Python (>=3.6)
Flask
OpenCV
Other dependencies (specified in requirements.txt)
Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/your-username/your-repo.git
cd your-repo
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Running the Application
Start the Flask server:

bash
Copy code
python app.py
Open a web browser and navigate to http://localhost:5000 to access the login page.

Log in with your credentials to access the main web pages.

Turn on the camera feed in the background to enable safety monitoring features.

Project Structure
app.py: Main Flask application file.
templates/: HTML templates for the web pages.
static/: Static files (CSS, JavaScript, images).
utils/: Utility functions for camera feed, person detection, audio threshold, etc.
forklift_avoidance/: Module for forklift avoidance system.
Forklift Avoidance System
The forklift avoidance system is implemented in the forklift_avoidance module. It includes:

Detection of persons in close proximity to the forklift using front and back cameras.
Triggering an alert sound and stopping the forklift when a person is detected.
Resuming normal operation after ensuring the area is clear.
For detailed usage and configuration, refer to the documentation in the forklift_avoidance/README.md file.

Contributors
Your Name your.email@example.com
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Mention any libraries, frameworks, or individuals whose code or work inspired or was used in this project.
User

also we have to initialize a database in mysql with certain create table statements and turn on the db server before starting the application.
ChatGPT
Safety Monitoring Web Application
Overview
This GitHub repository contains a web application designed for safety monitoring in industrial environments. The project includes features such as:

Detection of individuals without hardhats and vests
Counting the number of persons in a designated zone
Audio threshold detection in the environment
Additionally, the repository includes a separate module for forklift avoidance, where cameras mounted on the front and back of a forklift detect the presence of individuals in close proximity. When a person is detected, an alert sound is triggered, and the forklift stops until the situation is deemed safe.

Getting Started
Prerequisites
Before running the application, ensure you have the following installed:

Python (>=3.6)
Flask
OpenCV
MySQL Server
Other dependencies (specified in requirements.txt)
Database Setup
Initialize a MySQL database with the following table creation statements:

sql
Copy code
CREATE DATABASE IF NOT EXISTS safety_monitoring;
USE safety_monitoring;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL
);

-- Add more table creation statements as needed
Configure the database connection in the config.py file.

Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/your-username/your-repo.git
cd your-repo
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Running the Application
Start the MySQL server and initialize the database.

Start the Flask server:

bash
Copy code
python app.py
Open a web browser and navigate to http://localhost:5000 to access the login page.

Log in with your credentials to access the main web pages.

Turn on the camera feed in the background to enable safety monitoring features.

Project Structure
app.py: Main Flask application file.
templates/: HTML templates for the web pages.
static/: Static files (CSS, JavaScript, images).
utils/: Utility functions for camera feed, person detection, audio threshold, etc.
forklift_avoidance/: Module for forklift avoidance system.
Forklift Avoidance System
The forklift avoidance system is implemented in the forklift_avoidance module. It includes:

Detection of persons in close proximity to the forklift using front and back cameras.
Triggering an alert sound and stopping the forklift when a person is detected.
Resuming normal operation after ensuring the area is clear.
For detailed usage and configuration, refer to the documentation in the forklift_avoidance/README.md file.

Contributors
Your Name your.email@example.com
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Mention any libraries, frameworks, or individuals whose code or work inspired or was used in this project.

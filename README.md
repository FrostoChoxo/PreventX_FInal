# Safety Monitoring Web Application

## Overview

This GitHub repository contains a web application designed for safety monitoring in industrial environments. The project includes features such as:

- Detection of individuals without hardhats and vests
- Counting the number of persons in a designated zone
- Audio threshold detection in the environment

Additionally, the repository includes a separate module for forklift avoidance, where cameras mounted on the front and back of a forklift detect the presence of individuals in close proximity. When a person is detected, an alert sound is triggered, and the forklift stops until the situation is deemed safe.

## Getting Started

### Prerequisites

Before running the application, ensure you have the following installed:

- Python (>=3.6)
- Flask
- OpenCV
- MySQL Server
- Other dependencies (specified in `requirements.txt`)

### Database Setup

1. Initialize a MySQL database with the following table creation statements:

    ```sql
    CREATE DATABASE IF NOT EXISTS safety_monitoring;
    USE safety_monitoring;

    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        password VARCHAR(255) NOT NULL
    );

    CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL,
    password VARCHAR(100) NOT NULL,
    employee_id VARCHAR(10) NOT NULL
    );

    CREATE TABLE alerts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        serial_no VARCHAR(120) NOT NULL,
        alert_time DATETIME NOT NULL,
        category VARCHAR(50) NOT NULL,
        detection_type VARCHAR(50) NOT NULL,
        camera_id VARCHAR(50),
        camera_location VARCHAR(100),
        image_url VARCHAR(255)
    );


    ```

2. Configure the database connection in the `config.py` file.

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the MySQL server and initialize the database.

2. Start the Flask server:

   ```bash
   python app.py
   ```

3. Open a web browser and navigate to `http://localhost:5000` to access the login page.

4. Log in with your credentials to access the main web pages.

5. Turn on the camera feed in the background to enable safety monitoring features.

## Project Structure

- `app.py`: Main Flask application file.
- `templates/`: HTML templates for the web pages.
- `static/`: Static files (CSS, JavaScript, images).
- `utils/`: Utility functions for camera feed, person detection, audio threshold, etc.
- `forklift_avoidance/`: Module for forklift avoidance system.

## Forklift Avoidance System

The forklift avoidance system is implemented in the `forklift_avoidance` module. It includes:

- Detection of persons in close proximity to the forklift using front and back cameras.
- Triggering an alert sound and stopping the forklift when a person is detected.
- Resuming normal operation after ensuring the area is clear.

For detailed usage and configuration, refer to the documentation in the `forklift_avoidance/README.md` file.

## Contributors

- Your Name <your.email@example.com>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Mention any libraries, frameworks, or individuals whose code or work inspired or was used in this project.

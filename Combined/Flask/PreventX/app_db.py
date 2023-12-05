from ultralytics import YOLO
import cv2
import threading
import datetime
from flask import  Response, render_template, request, jsonify, redirect, url_for, make_response, session
import os
import pyaudio
import numpy as np
import math
from flask_socketio import SocketIO, emit
from flask import Flask, send_from_directory
from collections import defaultdict
from flask_mail import Mail, Message
import random
from flask_mysqldb import MySQL


import MySQLdb

app = Flask(__name__, static_url_path='/static')
app.secret_key = os.urandom(24)
socketio = SocketIO(app)


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# setting up smtp
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # e.g., 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'htacubed@gmail.com'
app.config['MAIL_PASSWORD'] = 'ycphysetncduzfyz'
app.config['MAIL_DEFAULT_SENDER'] = 'afaansubhani1965@gmail.com'

mail = Mail(app)

# Function to generate a 6-digit code
def generate_verification_code():
    return str(random.randint(100000, 999999))

# Function to send the verification code via email
def send_verification_email(email, code):
    msg = Message("Your Verification Code", recipients=[email])
    msg.body = f"Your verification code is {code}"
    mail.send(msg)

# Setting up DB
app.config['MYSQL_HOST'] = 'localhost'  # or your MySQL server address
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'haha2002'
app.config['MYSQL_DB'] = 'prevx_alerts'

mysql = MySQL(app)

max_people_allowed = 10  # Set your max people allowed
max_audio_threshold = 20000
audio_monitor = None
# Global camera feed object
camera_feed = None
first_excess_detection_time = None

#this class processes the Audio listening in the environment and checks if threshold is exceded the current magnitude
class AudioMonitor:
    def __init__(self, threshold):
        self.threshold = threshold
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.prev_avg = None

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format, channels=self.channels,
                                  rate=self.rate, input=True, frames_per_buffer=self.chunk)

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        print("Listening for crashes...")
        alert_sr_no_audio = 100
        try:
            while True:
                data = np.frombuffer(self.stream.read(self.chunk), dtype=np.int16)
                avg_magnitude = np.abs(data).mean()

                if self.prev_avg is not None:
                    if avg_magnitude > self.threshold * self.prev_avg:
                        print("Crash detected!")
                        alert_sr_no_audio += 1
                        current_time = datetime.datetime.now()
                        generate_audio_alert(alert_sr_no_audio, current_time, 'Audio', 'CRASH',
                                             'static/alert_images/audio crah/audio_alert.txt')

                        # Here, you can integrate your alert generation logic
                        # generate_alert( ... )

                self.prev_avg = avg_magnitude
        except KeyboardInterrupt:
            print("Audio monitor interrupted by user")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

    def update_threshold(self, new_threshold):
        self.threshold = new_threshold


def init_audio_monitor(threshold):
    global audio_monitor
    audio_monitor = AudioMonitor(threshold)

#this function generates the ppe,risk zone, injury alerts and stores in the database
def generate_alert(serial_no, alert_time, category, detection_type, image, camera_id, camera_location,
                   alert_image_dir):
    # Ensure the directory for alert images exists
    with app.app_context():
        if not os.path.exists(alert_image_dir):
            os.makedirs(alert_image_dir)

        alert_message = (
            f"alert_sr_no: {serial_no}, "
            f"{alert_time.strftime('%Y-%m-%d %H:%M:%S.%f')}, "
            f"category: {category},"
            f"detection: {detection_type}, "
            f"camera_id: {camera_id}, "
            f"camera_location: {camera_location}"
        )

        # Save the alert message to the specified text file

        # Construct the filename for the screenshot
        screenshot_filename = f"alert_{detection_type}_sr_no_{serial_no}.jpeg"
        screenshot_path_db = '/' + os.path.join(alert_image_dir, screenshot_filename)
        screenshot_path = os.path.join(alert_image_dir, screenshot_filename)

        # Save the screenshot image to the specified directory
        cv2.imwrite(screenshot_path, image)

        print(alert_message)  # Optionally print the message to the console

        # Database operation
        try:
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO alerts (serial_no, alert_time, category, detection_type, camera_id, camera_location, "
                "image_url) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (serial_no, alert_time, category, detection_type, camera_id, camera_location, screenshot_path_db))
            mysql.connection.commit()
            socketio.emit('new_alert', {
                'serial_no': serial_no,
                'alert_time': alert_time.strftime('%Y-%m-%d %H:%M:%S'),
                'category': category,
                'detection_type': detection_type,
                'camera_id': camera_id,
                'camera_location': camera_location,
                'image_url': screenshot_path_db  # Ensure this path is accessible via a public URL if needed
            })
            cur.close()
        except Exception as e:
            print(f"Error inserting alert into database: {e}")

#this function generates the audio alerts and stores in the database
def generate_audio_alert(serial_no, alert_time, category, detection_type, alert_text_path):
    with app.app_context():
        # Construct the alert message
        alert_message = (
            f"alert_sr_no: {serial_no}, "
            f"{alert_time.strftime('%Y-%m-%d %H:%M:%S.%f')}, "
            f"category: {category}, "
            f"detection: {detection_type}"
        )

        # Save the alert message to the specified text file
        with open(alert_text_path, "a") as file:
            file.write(alert_message + "\n")

        # Print the alert message to the console
        print(alert_message)

        # Database operation
        try:
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO alerts (serial_no, alert_time, category, detection_type) VALUES (%s, %s, %s, %s)",
                (serial_no, alert_time, category, detection_type))
            mysql.connection.commit()

            # Emitting the alert to the Flask app via socket.io
            socketio.emit('new_alert', {
                'serial_no': serial_no,
                'alert_time': alert_time.strftime('%Y-%m-%d %H:%M:%S'),
                'category': category,
                'detection_type': detection_type
                # 'image_url' is not applicable here as this is an audio alert
            })
            cur.close()
        except Exception as e:
            print(f"Error inserting alert into database: {e}")

#in this class the video processing is done on the video feed
class CameraFeed:
    def __init__(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.model = YOLO("best_anothertry.pt")  # Load your model here
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        global max_people_allowed
        classNames = ['hardhat', 'no-hardhat', 'no-vest', 'person', 'safety-vest', 'fall-detected']
        exclude_classes = ['person']

        # Set camera details
        camera_id = "Cam123"  # Replace with actual camera ID
        camera_location = "Laptop"  # Replace with actual camera location
        fall_detection_threshold = 45000
        first_fall_detection_time = None
        global max_people_allowed
        global first_excess_detection_time
        first_hardhat_detection_time = None
        first_vest_detection_time = None
        alert_sr_no_injury = 100
        alert_sr_no_ppe = 100
        alert_sr_no_risk = 100

        while True:
            success, img = self.cap.read()
            if not success:
                continue

            person_counter = 0
            no_hardhat_detected = False
            no_vest_detected = False
            fall_detected = False
            # img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

            # Perform YOLO detections
            results = self.model(img, stream=True)

            current_time = datetime.datetime.now()
            current_time_person = datetime.datetime.now()

            # Process PPE detection results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    if class_name == 'person':
                        person_counter += 1

                    if class_name in exclude_classes:
                        continue  # Skip drawing bounding boxes for excluded classes

                    # ... (rest of the detection and drawing code for PPE detection) ...
                    if class_name == 'fall-detected':
                        fall_detected = True
                        color = (0, 0, 255)  # Red for fall detected
                    elif class_name == 'no-hardhat':
                        no_hardhat_detected = True
                        color = (255, 0, 0)  # Red for no helmet
                    elif class_name == 'no-vest':
                        no_vest_detected = True
                        color = (255, 0, 0)  # Red for no vest
                    else:
                        color = (85, 45, 255)  # Default color
                    label = f'{class_name} {conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3

                    if conf > 0.6:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img, f'Persons: {person_counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                            cv2.LINE_AA)

            # Process fall detection results
            #
            # ... [Rest of your code that processes alerts based on detections] ...

            if fall_detected:
                if first_fall_detection_time is None:
                    first_fall_detection_time = current_time
                else:
                    elapsed_time = (current_time - first_fall_detection_time).total_seconds()
                    if elapsed_time >= 10:  # 10 seconds elapsed with fall detected
                        alert_sr_no_injury += 1
                        generate_alert(alert_sr_no_injury, current_time, 'Safety', 'FALL_DETECTED', img, camera_id,
                                       camera_location,
                                       r'static\alert_images\Injury')
                        # After generating alert, reset timer
                        first_fall_detection_time = None
            else:
                # Reset the timer if no fall is detected
                first_fall_detection_time = None

            if no_hardhat_detected:
                if first_hardhat_detection_time is None:
                    first_hardhat_detection_time = current_time
                else:
                    elapsed_time = (current_time - first_hardhat_detection_time).total_seconds()
                    if elapsed_time >= 10:  # 50 seconds elapsed without a hardhat
                        alert_sr_no_ppe += 1
                        generate_alert(alert_sr_no_ppe, current_time, 'PPE', 'NO-HARDHAT', img, camera_id,
                                       camera_location,
                                       r'static\alert_images\PPE')

                        first_hardhat_detection_time = current_time  # Reset timer after generating alert

            else:
                # Reset the detection timer if a hardhat is detected
                first_hardhat_detection_time = None

            if no_vest_detected:
                if first_vest_detection_time is None:
                    first_vest_detection_time = current_time
                else:
                    elapsed_time = (current_time - first_vest_detection_time).total_seconds()
                    if elapsed_time >= 10:  # 50 seconds elapsed without a hardhat
                        alert_sr_no_ppe += 1
                        generate_alert(alert_sr_no_ppe, current_time, 'PPE', 'NO-VEST', img, camera_id, camera_location,

                                       r'static\alert_images\PPE')

                        first_vest_detection_time = current_time  # Reset timer aenerating alert
            else:
                # Reset the detection timer if a hardhat is detected
                first_vest_detection_time = None

            if person_counter > max_people_allowed:
                if first_excess_detection_time is None:
                    first_excess_detection_time = current_time_person  # Start the timer
                else:
                    elapsed_time = (current_time_person - first_excess_detection_time).total_seconds()
                    if elapsed_time >= 10:  # 60 seconds elapsed with excess people count
                        alert_sr_no_risk += 1
                        generate_alert(alert_sr_no_risk, current_time, 'Risk_Zone', 'PERSON_COUNT_LIMIT', img,
                                       camera_id,
                                       camera_location,
                                       r'static\alert_images\Risk zone')
                        # After generating alert, reset timer
                        first_excess_detection_time = None
            else:
                # Reset the timer as the count is below the threshold
                first_excess_detection_time = None

            # Display the results
            # yield img
            (flag, encodedImage) = cv2.imencode(".jpg", img)
            if not flag:
                continue
            self.current_frame = img

    def get_frame(self):
        # Encode the current frame for streaming
        ret, jpeg = cv2.imencode('.jpg', self.current_frame)
        if not ret:
            return None
        return jpeg.tobytes()


def init_camera_feed(): #makes it available in the main function
    global camera_feed
    camera_feed = CameraFeed()

#this function gives the processed camera feed from backend to the frontend
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera_feed.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

#these are all routes for the webpages
@app.route('/home')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        username = request.cookies.get('username', 'Guest')  # Default to 'Guest' if the cookie is not set
        email = request.cookies.get('email', 'guest@preventx.co')
        return render_template('home.html', username=username, email=email)


@app.route('/camerafeed')
def camerafeed():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    else:
        username = request.cookies.get('username', 'Guest')  # Default to 'Guest' if the cookie is not set
        email = request.cookies.get('email', 'guest@preventx.co')
        return render_template('camerafeed.html', username=username, email=email)


@app.route('/add_user')
def add_user():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    else:
        username = request.cookies.get('username', 'Guest')  # Default to 'Guest' if the cookie is not set
        email = request.cookies.get('email', 'guest@preventx.co')
        return render_template('add_user.html', username=username, email=email)


@app.route('/alert')
def alert():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    else:
        username = request.cookies.get('username', 'Guest')  # Default to 'Guest' if the cookie is not set
        email = request.cookies.get('email', 'guest@preventx.co')
        return render_template('alert.html', username=username, email=email)


@app.route('/view_analytics')
def view_analytics():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    else:
        username = request.cookies.get('username', 'Guest')  # Default to 'Guest' if the cookie is not set
        email = request.cookies.get('email', 'guest@preventx.co')
        return render_template('view_analytics.html', username=username, email=email)

#this function sends the alerts to frontend alerts table in json format
@app.route('/get_alerts')
def get_alerts():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM alerts")
        alerts = cur.fetchall()
        cur.close()
        alerts_list = []
        for alert in alerts:
            alerts_list.append({
                'serial_no': alert[1],
                'alert_time': str(alert[2]),  # Convert datetime to string
                'category': alert[3],
                'detection_type': alert[4],
                'camera_id': alert[5],
                'camera_location': alert[6],
                'image_url': alert[7]
            })

        return jsonify(alerts_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#this handles the realtime availability of the alerts on the frontend
@socketio.on('fetch_alerts')
def handle_fetch_alerts():
    # You can call the get_alerts function here or replicate its logic
    alerts = get_alerts()
    emit('alerts_data', alerts)



#when risk zone set in frontend that data is feteched from there to backend using this
@app.route('/set_max_people', methods=['POST'])
def set_max_people_allowed():
    global max_people_allowed
    try:
        max_people_allowed = int(request.form['inputNumber'])
        return f"Max people allowed updated to {max_people_allowed}", 200
    except (ValueError, KeyError):
        return "Invalid input", 400

#when audio threshold set in frontend that data is feteched from there to backend using this
@app.route('/set_max_audio', methods=['POST'])
def set_max_audio_threshold():
    global max_audio_threshold
    try:
        max_audio_threshold = int(request.form['inputThreshold'])
        if audio_monitor:
            audio_monitor.update_threshold(max_audio_threshold)
        return f"Max audio threshold updated to {max_audio_threshold}", 200
    except (ValueError, KeyError):
        return "Invalid input", 400


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        employee_id = request.form.get('username')
        password = request.form.get('password')

        # Debug print
        print("Employee ID:", employee_id)
        print("Password:", password)

        user = query_user_by_employee_id(employee_id)

        # Debug print
        print("User from DB:", user)

        if user and verify_password(password, user[3]):  #  user[3] is the password

            # Generate and send verification code

            code = generate_verification_code()
            send_verification_email(user[2], code)  # Assuming user[2] is the email address

            # Create a response object to set a cookie
            resp = make_response(render_template('verify.html', generated_code=code, user_id=user[0]))
            # Set the cookie with the username
            resp.set_cookie('username', user[1], httponly=True)
            resp.set_cookie('email', user[2], httponly=True)
            return resp
        else:
            return render_template('login.html', error="Invalid credentials.")

    return render_template('login.html')


#authentication of 2fa
@app.route('/verify', methods=['POST'])
def verify():
    user_id = request.form.get('user_id')
    input_code = request.form.get('input_code')
    generated_code = request.form.get('generated_code')

    if input_code == generated_code:
        session['logged_in'] = True
        session['user_id'] = user_id
        # Correct code, log the user in
        return redirect(url_for('home'))  # Redirect to the home page
    else:
        # Incorrect code, show an error
        return render_template('login.html', error="Incorrect verification code.")


#sql query for data
def query_user_by_employee_id(employee_id):
    try:
        cur = mysql.connection.cursor()
        print("Executing SQL query...")
        cur.execute("SELECT * FROM users WHERE employee_id = %s", [employee_id])
        user = cur.fetchone()
        cur.close()
        return user
    except Exception as e:
        print(f"Database query error: {e}")
        return None


def verify_password(input_password, stored_password):
    # If using plain text passwords
    return input_password == stored_password


#add user function to add the user in the database
@app.route('/submit_user', methods=['POST'])
def submit_user():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    employee_id = request.form['employee_id']

    # Code to insert data into the database goes here
    try:
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (name, email, password, employee_id) VALUES (%s, %s, %s, %s)",
                    (name, email, password, employee_id))

        mysql.connection.commit()
        cur.close()
        return redirect(url_for('add_user'))

    except Exception as e:
        print(f"Error inserting data into database: {e}")
        return redirect(url_for('add_user'))



      # Redirect to home after insertion


@app.route('/logout')
def logout():
    # Redirect to login page
    session.clear()
    return redirect(url_for('login'))

#All charts in the webpage all these functions return json data to the frontend

#analytics page
@app.route('/analytics_data')
def analytics_data():
    try:
        cur = mysql.connection.cursor()
        # Modify this query based on the type of analytics you need
        cur.execute("SELECT category, COUNT(*) as count FROM alerts GROUP BY category")
        data = cur.fetchall()
        cur.close()
        # Format data for the frontend, e.g., converting it to a dictionary or JSON
        formatted_data = {d[0]: d[1] for d in data}
        return jsonify(formatted_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/alert_types_distribution')
def alert_types_distribution():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT category, COUNT(*) as count FROM alerts GROUP BY category")
        data = cur.fetchall()
        cur.close()
        formatted_data = {'labels': [d[0] for d in data], 'data': [d[1] for d in data]}
        return jsonify(formatted_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/alerts_over_time')
def alerts_over_time():
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT DATE(alert_time) as date, category, COUNT(*) as count FROM alerts GROUP BY DATE(alert_time), category")
        data = cur.fetchall()
        cur.close()

        # Transform the data
        results = defaultdict(lambda: defaultdict(int))
        dates = set()
        for row in data:
            date, detection_type, count = row
            results[detection_type][date] = count
            dates.add(date)

        sorted_dates = sorted(list(dates))
        datasets = []
        for detection_type, counts in results.items():
            dataset = {
                'label': detection_type,
                'data': [counts.get(date, 0) for date in sorted_dates],
                'fill': False,
                # You can customize the line color, etc. here
            }
            datasets.append(dataset)

        transformed_data = {
            'labels': sorted_dates,
            'datasets': datasets
        }
        return jsonify(transformed_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#dashboord graphs
@app.route('/ppe_alerts_data')
def ppe_alerts_data():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT detection_type, COUNT(*) as count FROM alerts WHERE category='PPE' AND (detection_type='NO-HARDHAT' OR detection_type='NO-VEST') GROUP BY detection_type")
        data = cur.fetchall()
        cur.close()
        formatted_data = {d[0]: d[1] for d in data}
        return jsonify(formatted_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/daily_alerts_data')
def daily_alerts_data():
    try:
        cur = mysql.connection.cursor()
        # Directly using SQL to filter today's date
        query = """SELECT category, COUNT(*) as count FROM alerts WHERE DATE(alert_time) = CURDATE() GROUP BY category """
        cur.execute(query)
        data = cur.fetchall()
        cur.close()
        formatted_data = {d[0]: d[1] for d in data}
        return jsonify(formatted_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/monthly_ppe_alerts_status')
def monthly_ppe_alerts_status():
    try:
        cur = mysql.connection.cursor()
        # Assuming 'PPE' is the category for these alerts
        cur.execute("SELECT COUNT(*) FROM alerts WHERE category='PPE' AND MONTH(alert_time) = MONTH(CURRENT_DATE()) AND YEAR(alert_time) = YEAR(CURRENT_DATE())")
        count = cur.fetchone()[0]
        cur.close()
        return jsonify({"ppe_alerts_count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# first page to render after the flask application is started
@app.route('/')
def index():
    # Redirect to the login page
    return render_template('login.html')

#MAIN
if __name__ == '__main__':
    init_camera_feed()
    init_audio_monitor(max_audio_threshold)
    socketio.run(app, host='0.0.0.0', port=8000, allow_unsafe_werkzeug=True, debug=True, use_reloader=False)

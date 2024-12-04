import os
import json
import bcrypt
import pyotp
import qrcode
import cv2
from flask import Flask, request, render_template, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "your_secret_key"

USER_DATA_FILE = "users.json"
FACE_DIR = "static/faces"

# Initialize files and directories
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "w") as f:
        json.dump({}, f)
os.makedirs(FACE_DIR, exist_ok=True)

def load_users():
    """Load user data from JSON file."""
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    """Save user data to JSON file."""
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()
        if username in users:
            flash("User already exists. Please login.", "danger")
            return redirect(url_for("login"))

        # Generate OTP secret and hash password
        otp_secret = pyotp.random_base32()
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        # Save user details
        users[username] = {"password": hashed_password, "otp_secret": otp_secret}
        save_users(users)

        # Generate QR code for OTP
        provisioning_uri = pyotp.TOTP(otp_secret).provisioning_uri(name=username, issuer_name="Secure App")
        qr = qrcode.make(provisioning_uri)
        qr_path = os.path.join("static", f"{username}_qr.png")
        qr.save(qr_path)

        # Capture face
        face_captured = capture_face(username)
        if not face_captured:
            flash("Face capture failed. Registration incomplete.", "danger")
            return redirect(url_for("register"))

        flash("Registration successful! Scan QR code and login.", "success")
        return render_template("register_success.html", qr_path=qr_path)

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()
        if username not in users:
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

        stored_password = users[username]["password"]
        if not bcrypt.checkpw(password.encode(), stored_password.encode()):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

        # Face verification
        if not verify_face(username):
            flash("Face verification failed.", "danger")
            return redirect(url_for("login"))

        # OTP Verification
        otp_secret = users[username]["otp_secret"]
        otp = request.form["otp"]
        totp = pyotp.TOTP(otp_secret)
        if not totp.verify(otp):
            flash("Invalid OTP.", "danger")
            return redirect(url_for("login"))

        flash("Login successful!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

def capture_face(username):
    """Capture and save user's face."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_path = os.path.join(FACE_DIR, f"{username}_face.jpg")
            cv2.imwrite(face_path, face)
            cap.release()
            cv2.destroyAllWindows()
            return True

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

def verify_face(username):
    """Verify user's face during login."""
    stored_face_path = os.path.join(FACE_DIR, f"{username}_face.jpg")
    if not os.path.exists(stored_face_path):
        return False

    stored_face = cv2.imread(stored_face_path, cv2.IMREAD_GRAYSCALE)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            captured_face = gray[y:y+h, x:x+w]
            captured_face = cv2.resize(captured_face, (stored_face.shape[1], stored_face.shape[0]))

            # Compare faces
            diff = cv2.absdiff(stored_face, captured_face)
            if diff.mean() < 50:  # Adjust threshold
                cap.release()
                cv2.destroyAllWindows()
                return True

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

if __name__ == "__main__":
    app.run(debug=True)

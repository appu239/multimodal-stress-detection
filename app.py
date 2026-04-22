from google.oauth2 import id_token
from google.auth.transport import requests
import jwt
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

client = MongoClient("mongodb+srv://aparnac239_db_user:woOIf19R5CeXGlnm@cluster0.cy2podc.mongodb.net/?retryWrites=true&w=majority")
db = client["stressai"]
contact_collection = db["contacts"]
users_collection = db["users"]

SECRET_KEY = "super_secret_key_123"
GOOGLE_CLIENT_ID = "575049109828-0ggp1vbn64ojk93vp71e2dp59634s3nv.apps.googleusercontent.com"
import librosa
import numpy as np
import joblib
import os
import uuid
import subprocess
import bcrypt

# ==============================
# OPTIONAL WHISPER (Speech → Text)
# ==============================
try:
    import whisper
    WHISPER_AVAILABLE = True
except:
    WHISPER_AVAILABLE = False

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app, origins=["http://localhost:3000","http://127.0.0.1:3000"])

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "stress_model.pkl")
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "text_model.pkl") # Fixed path
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl") # New path

# ==============================
# LOAD MODELS
# ==============================

# ==============================
# STRESS LABELS
# ==============================
stress_labels = {
    0: "Low Stress",
    1: "Moderate Stress",
    2: "High Stress"
}

# ==============================
# LOAD WHISPER
# ==============================
if WHISPER_AVAILABLE:
    try:
        whisper_model = whisper.load_model("tiny")
        print("OK: Whisper model loaded successfully")
    except Exception as e:
        print(f"FAIL: Whisper load failed: {e}")
        WHISPER_AVAILABLE = False
else:
    print("WARNING: Whisper is NOT installed (openai-whisper)")

from functools import wraps

# ==============================
# AUTH DECORATOR
# ==============================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        
        try:
            # Handle "Bearer <token>" format
            if token.startswith("Bearer "):
                token = token.split(" ")[1]
            
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user_email = data["email"]
        except Exception as e:
            return jsonify({"error": "Token is invalid"}), 401
        
        return f(current_user_email, *args, **kwargs)
    return decorated

# =================================================
# LOGIN ROUTE  ✅ (ADDED)
# =================================================


# ==============================
# AUDIO FEATURE EXTRACTION
# ==============================
def extract_audio_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        print(f"DEBUG: Audio loaded. Duration: {duration:.2f}s, Sample Rate: {sr}")

        if duration < 0.1:
            raise ValueError("Audio segment too short (less than 0.1s)")

        # Match training length
        if len(audio) < 3 * sr:
            audio = np.pad(audio, (0, 3 * sr - len(audio)))
        else:
            audio = audio[:3 * sr]

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.hstack((
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1)
        ))
        return features.reshape(1, -1)
    except Exception as e:
        print(f"DEBUG: Feature Extraction failed: {e}")
        raise

# ==============================
# SPEECH → TEXT
# ==============================
def speech_to_text(file_path, language=None):
    if not WHISPER_AVAILABLE:
        print("DEBUG: Whisper skip (not available)")
        return "[Transcription system unavailable]"
    
    options = {}
    if language:
        options["language"] = language
    
    try:
        # Force task to 'transcribe' to prevent translation to English
        result = whisper_model.transcribe(file_path, task="transcribe", **options)
        text = result.get("text", "").strip()
        print(f"DEBUG: Whisper transcript: {text[:50]}...")
        return text if text else "[No speech detected]"
    except Exception as e:
        print(f"DEBUG: Whisper failed: {e}")
        return "[Error during transcription]"

# =================================================
# AUDIO PREDICTION
# =================================================
@app.route("/predict", methods=["POST"])
@token_required
def predict_audio(current_user_email):
    audio_model = joblib.load(AUDIO_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    temp_wav = os.path.join(BASE_DIR, f"{uuid.uuid4().hex}.wav")
    
    audio_file.save(temp_wav)
    file_size = os.path.getsize(temp_wav)
    print(f"DEBUG: Received audio blob. Size: {file_size} bytes")

    if file_size < 100: # Sanity check for empty blobs
        return jsonify({"error": "Audio file is too small or empty"}), 400

    try:
        # Since frontend sends WAV, we don't strictly need ffmpeg for conversion
        # but we keep it here to ensure correct sample rate/channel count if needed
        # Or we can just re-save it to be sure.
        # Actually, let's just use temp_wav directly if ffmpeg is skipped, 
        # but ffmpeg is safer for normalization.
        # We'll rename it to temp_input for ffmpeg logic
        temp_input = temp_wav + ".tmp"
        os.rename(temp_wav, temp_input)
        
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", temp_input, "-ar", "16000", "-ac", "1", temp_wav],
            capture_output=True,
            text=True
        )
        if os.path.exists(temp_input): os.remove(temp_input) # Cleanup temp
        if proc.returncode != 0:
            print(f"FAIL: ffmpeg failed: {proc.stderr}")
            return jsonify({"error": "Audio conversion failed"}), 500

        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
            print(f"FAIL: ffmpeg output missing or empty: {temp_wav}")
            return jsonify({"error": "Converted audio is empty"}), 500

        features = extract_audio_features(temp_wav)
        features_scaled = scaler.transform(features)
        
        # Prediction and Confidence
        pred_probs = audio_model.predict_proba(features_scaled)[0]
        # Log precision probabilities for debugging
        print(f"DEBUG: Probabilities - Low: {pred_probs[0]:.4f}, Mod: {pred_probs[1]:.4f}, High: {pred_probs[2]:.4f}")
        
        pred_idx = np.argmax(pred_probs)
        confidence = float(pred_probs[pred_idx])
        
        stress = stress_labels[pred_idx]

        # Use language if provided
        language = request.form.get("language")
        transcript = speech_to_text(temp_wav, language=language)

        # SAVE TO DB
        assessments_collection.insert_one({
            "email": current_user_email,
            "type": "audio",
            "prediction": stress,
            "confidence": confidence,
            "language": language,
            "features": features.tolist(),
            "speech_text": transcript or "[No speech text generated]",
            "timestamp": datetime.datetime.utcnow()
        })

        return jsonify({
            "predicted_stress_level": stress,
            "confidence": confidence,
            "speech_text": transcript or "[No speech text generated]",
            "debug_probs": pred_probs.tolist()
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

# =================================================
# TEXT PREDICTION (NLP MODEL)
# =================================================
@app.route("/predict-text", methods=["POST"])
@token_required
def predict_text(current_user_email):
    text_model = joblib.load(TEXT_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        X = vectorizer.transform([text])
        
        # Prediction and Confidence
        pred_probs = text_model.predict_proba(X)[0]
        pred_idx = np.argmax(pred_probs)
        confidence = float(pred_probs[pred_idx])
        
        # Labels might be strings directly in the model or indices
        classes = text_model.classes_
        stress = classes[pred_idx]
        
        if not isinstance(stress, str):
            stress = stress_labels[int(stress)]

        # SAVE TO DB
        assessments_collection.insert_one({
            "email": current_user_email,
            "type": "text",
            "prediction": stress,
            "confidence": confidence,
            "text": text,
            "timestamp": datetime.datetime.utcnow()
        })

        return jsonify({
            "predicted_stress_level": stress,
            "confidence": confidence,
            "speech_text": text
        })

    except Exception as e:
        print("TEXT ERROR:", e)
        return jsonify({"error": str(e)}), 500
    # =================================================
# CONTACT ROUTE
# =================================================
@app.route("/api/contact", methods=["POST"])
def contact():

    data = request.get_json()

    name = data.get("name")
    email = data.get("email")
    message = data.get("message")

    contact_collection.insert_one({
        "name": name,
        "email": email,
        "message": message
    })

    return jsonify({
        "status": "success",
        "message": "Message saved in database"
    }), 200
# ==============================
# GOOGLE LOGIN (PRODUCTION LEVEL)
# ==============================

GOOGLE_CLIENT_ID = "575049109828-0ggp1vbn64ojk93vp71e2dp59634s3nv.apps.googleusercontent.com"

users_collection = db["users"]

@app.route("/api/google-login", methods=["POST"])
def google_login():

    data = request.get_json()
    token = data.get("token")

    try:
        # Verify token with Google
        idinfo = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10
        )

        email = idinfo["email"]
        name = idinfo.get("name", "")
        picture = idinfo.get("picture", "")

        # Check if user exists
        user = users_collection.find_one({"email": email})

        if not user:
            users_collection.insert_one({
                "name": name,
                "email": email,
                "picture": picture,
                "role": "USER"
            })

        # Create JWT token
        payload = {
            "email": email,
            "role": "USER",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=5)
        }

        app_token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

        return jsonify({
            "token": app_token,
            "role": "USER",
            "name": name
        })

    except Exception as e:
        print("GOOGLE LOGIN ERROR:", str(e))   # 👈 ADD THIS
        return jsonify({"error": str(e)}), 401
# ==============================
# GET USER COUNT
# ==============================
@app.route("/api/user-count", methods=["GET"])
def get_user_count():
    try:
        count = users_collection.count_documents({})
        return jsonify({
            "count": count
        }),200
    except Exception as e:
        return jsonify({"error": str(e)}),500

# ==============================
# REGISTER USER
# ==============================
@app.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.get_json()

        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not name or not email or not password:
            return jsonify({"error": "All fields required"}), 400

        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return jsonify({"error": "User already exists"}), 400

        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        users_collection.insert_one({
            "name": name,
            "email": email,
            "password": hashed_pw,
            "role": "USER"
        })

        return jsonify({"message": "User registered successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # ==============================
# LOGIN USER (MongoDB Based)
# ==============================
@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        user = users_collection.find_one({"email": email})

        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        if not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            return jsonify({"error": "Invalid credentials"}), 401

        payload = {
            "email": email,
            "role": user.get("role", "USER"),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=5)
        }

        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        
        # LOG LOGIN SUCCESS
        audit_logs_collection.insert_one({
            "event": "USER_LOGIN",
            "actor": email,
            "target": "System",
            "ip": request.remote_addr,
            "status": "Success",
            "timestamp": datetime.datetime.utcnow()
        })

        print(f"DEBUG: Login successful for {email}. Role: {user.get('role', 'USER')}") # DEBUG LOG
        return jsonify({
            "token": token,
            "role": user.get("role", "USER"),
            "name": user.get("name", "")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# FORGOT PASSWORD
# ==============================
@app.route("/api/forgot-password", methods=["POST"])
def forgot_password():
    try:
        data = request.get_json()
        email = data.get("email")
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        user = users_collection.find_one({"email": email})
        
        # We always return success to prevent email enumeration, 
        # but in a real app we'd trigger an email here.
        if user:
            print(f"DEBUG: Password reset requested for {email}")
            log_audit_event("PASSWORD_RESET_REQUESTED", "System", f"User: {email}")
            
        return jsonify({"message": "If an account exists for this email, you will receive reset instructions shortly."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# USER HISTORY
# ==============================
@app.route("/api/history", methods=["GET"])
@token_required
def get_history(current_user_email):
    try:
        from bson import ObjectId
        history = list(assessments_collection.find({"email": current_user_email}).sort("timestamp", -1))
        for item in history:
            item["_id"] = str(item["_id"])
        return jsonify(history), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history/<id>", methods=["DELETE"])
@token_required
def delete_history_item(current_user_email, id):
    try:
        from bson import ObjectId
        result = assessments_collection.delete_one({"_id": ObjectId(id), "email": current_user_email})
        if result.deleted_count == 0:
            return jsonify({"error": "Item not found or unauthorized"}), 404
        return jsonify({"message": "Deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# ADMIN ENDPOINTS
# ==============================

assessments_collection = db["assessments"]
audit_logs_collection = db["audit_logs"]

@app.route("/api/users", methods=["GET"])
def get_users():
    try:
        users = list(users_collection.find({}, {"_id": 0, "password": 0}))
        return jsonify(users), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    try:
        total_users = users_collection.count_documents({})
        total_assessments = assessments_collection.count_documents({})
        
        # Calculate high stress percentage
        high_stress_count = assessments_collection.count_documents({"prediction": {"$in": ["High Stress", "Moderate Stress"]}})
        high_stress_pct = round((high_stress_count / total_assessments * 100), 1) if total_assessments > 0 else 0
        
        # Calculate Daily Analysis Volume (Last 7 Days)
        seven_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
        
        # Daily breakdown using aggregation
        pipeline = [
            {"$match": {"timestamp": {"$gte": seven_days_ago}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        daily_results = list(assessments_collection.aggregate(pipeline))
        
        # Fill in missing days for a consistent 7-day window
        daily_counts = []
        for i in range(7):
            d = (datetime.datetime.utcnow() - datetime.timedelta(days=6-i)).strftime("%Y-%m-%d")
            match = next((item for item in daily_results if item["_id"] == d), None)
            daily_counts.append({
                "date": d,
                "label": (datetime.datetime.utcnow() - datetime.timedelta(days=6-i)).strftime("%a"),
                "count": match["count"] if match else 0
            })

        # Calculate active users (unique users who took an assessment in last 7 days)
        active_users = len(assessments_collection.distinct("email", {"timestamp": {"$gte": seven_days_ago}}))
        
        return jsonify({
            "total_users": total_users,
            "total_assessments": total_assessments,
            "high_stress_percentage": f"{high_stress_pct}%",
            "active_users": active_users,
            "daily_counts": daily_counts
        }), 200
    except Exception as e:
        print(f"ANALYTICS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/audit-logs", methods=["GET"])
def get_audit_logs():
    try:
        logs = list(audit_logs_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(50))
        return jsonify(logs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helper to log actions
def log_audit_event(event, actor, target, status="Success"):
    try:
        audit_logs_collection.insert_one({
            "event": event,
            "actor": actor,
            "target": target,
            "ip": request.remote_addr,
            "status": status,
            "timestamp": datetime.datetime.utcnow()
        })
    except:
        pass

@app.route("/api/users/<email>", methods=["GET"])
def get_user(email):
    try:
        user = users_collection.find_one({"email": email}, {"_id": 0, "password": 0})
        if not user:
            return jsonify({"error": "User not found"}), 404
        return jsonify(user), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/<email>", methods=["DELETE"])
def delete_user(email):
    try:
        result = users_collection.delete_one({"email": email})
        if result.deleted_count == 0:
            return jsonify({"error": "User not found"}), 404
        
        # Log the action
        log_audit_event("USER_DELETED", "Admin", f"User: {email}")
        return jsonify({"message": "User deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/<email>", methods=["PUT"])
def update_user(email):
    try:
        data = request.get_json()
        new_role = data.get("role")
        new_status = data.get("status")
        
        update_fields = {}
        if new_role: update_fields["role"] = new_role
        if new_status: update_fields["status"] = new_status
        
        if not update_fields:
             return jsonify({"error": "No fields to update"}), 400

        result = users_collection.update_one({"email": email}, {"$set": update_fields})
        
        if result.matched_count == 0:
            return jsonify({"error": "User not found"}), 404
            
        # Log the action
        log_audit_event("USER_UPDATED", "Admin", f"User: {email} updated {update_fields}")
        return jsonify({"message": "User updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# MODEL MANAGEMENT
# ==============================

@app.route("/api/contribute-data", methods=["POST"])
def contribute_data():
    try:
        data = request.get_json()
        text = data.get("text")
        label = data.get("label")
        
        if not text or not label:
            return jsonify({"error": "Text and label required"}), 400
            
        csv_path = os.path.join(BASE_DIR, "stress_text_data.csv")
        import csv
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([text, label])
            
        log_audit_event("DATA_CONTRIBUTED", "User/Admin", f"Label: {label}")
        return jsonify({"message": "Data contributed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/retrain", methods=["POST"])
def retrain_model():
    try:
        # Trigger training script
        script_path = os.path.join(BASE_DIR, "train_text_model.py")
        result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
        
        # Reload models
        global text_model, vectorizer
        text_model = joblib.load(TEXT_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        
        log_audit_event("MODEL_RETRAINED", "Admin", "Text Stress Model")
        return jsonify({
            "status": "success",
            "message": "Model retrained and reloaded successfully",
            "output": result.stdout
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# RUN
# ==============================
from flask import send_from_directory

@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


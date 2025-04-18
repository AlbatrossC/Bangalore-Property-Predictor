from flask import Flask, request, jsonify, render_template, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import numpy as np
import pickle
import json
import pandas as pd
from flask_cors import CORS
import requests
import sqlite3
import os
from flask_migrate import Migrate

app = Flask(__name__)

# ======================== DATABASE CONFIGURATION ========================
# Check if running on Render
is_render = os.environ.get('RENDER') == 'true'

# Database configuration
if is_render:
    # PostgreSQL configuration for Render
    db_url = os.environ.get('DATABASE_URL')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    # SQLite configuration for local development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key_here')

db = SQLAlchemy(app)
migrate = Migrate(app, db)  # For database migrations
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = "info"
NOMINATIM_API_URL = "https://nominatim.openstreetmap.org/search"
CORS(app)

# ======================== MODEL LOADING ========================
model_file = "banglore_home_prices_model.pickle"
columns_file = "columns.json"

try:
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"❌ Model file '{model_file}' not found.")
    
    if not os.path.exists(columns_file):
        raise FileNotFoundError(f"❌ Columns file '{columns_file}' not found.")

    with open(model_file, "rb") as f:
        model = pickle.load(f)
    
    with open(columns_file, "r") as f:
        data_columns = json.load(f)["data_columns"]

    if len(data_columns) < 4:
        raise ValueError("❌ Invalid data_columns format. Expected at least 4 columns (sqft, bath, bhk, property_age, locations).")

    locations = data_columns[4:]
    print(f"✅ Model and {len(locations)} locations loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    data_columns = []
    locations = []

# ======================== DATABASE MODELS ========================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    sqft = db.Column(db.Float, nullable=False)
    bhk = db.Column(db.Integer, nullable=False)
    bath = db.Column(db.Integer, nullable=False)
    property_age = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    user = db.relationship('User', backref=db.backref('favorites', lazy=True))

# ======================== DATABASE HELPER FUNCTIONS ========================
def get_db_connection():
    """Returns the appropriate database connection based on environment"""
    if is_render:
        import psycopg2
        return psycopg2.connect(os.environ.get('DATABASE_URL'))
    else:
        return sqlite3.connect("house_prices.db")

def execute_db_query(query, params=None, fetch=True):
    """Universal database query executor"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if fetch:
            result = cursor.fetchall()
        else:
            conn.commit()
            result = None
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
    
    return result

def get_location_from_db(location_name):
    """Fetch coordinates from database"""
    if is_render:
        query = """
        SELECT latitude, longitude FROM heatmap_data 
        WHERE LOWER(TRIM(location)) = LOWER(TRIM(%s))
        """
    else:
        query = """
        SELECT latitude, longitude FROM heatmap_data 
        WHERE TRIM(LOWER(location)) = LOWER(?)
        """
    
    result = execute_db_query(query, (location_name.strip(),))
    return result[0] if result else None

def save_location_to_db(location_name, lat, lon):
    """Saves a new location to the database"""
    if is_render:
        query = """
        INSERT INTO heatmap_data (location, latitude, longitude) 
        VALUES (%s, %s, %s)
        """
    else:
        query = """
        INSERT INTO heatmap_data (location, latitude, longitude) 
        VALUES (?, ?, ?)
        """
    
    execute_db_query(query, (location_name.strip(), lat, lon), fetch=False)
    print(f"✅ Saved {location_name} to database!")

# ======================== ROUTES ========================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/")
def home():
    return render_template("app.html")

@app.route("/get_locations", methods=["GET"])
def get_locations():
    return jsonify({"locations": locations})

@app.route("/login_page")
def login_page():
    return render_template("login.html")

@app.route("/register_page")
def register_page():
    return render_template("register.html")

@app.route("/predict_price", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("📥 Received Data for Prediction:", data)

        if model is None or data_columns is None:
            return jsonify({"error": "Model or data columns not loaded."}), 500

        features = np.zeros(len(data_columns))
        features[data_columns.index("total_sqft")] = float(data.get("total_sqft", 0))
        features[data_columns.index("bath")] = int(data.get("bath", 0))
        features[data_columns.index("property_age")] = int(data.get("property_age", 0))
        features[data_columns.index("bhk")] = int(data.get("bhk", 0))

        location = data.get("location", "")
        if location in data_columns:
            loc_index = data_columns.index(location)
            features[loc_index] = 1
        else:
            print(f"⚠️ Location '{location}' not found in model's data columns.")

        prediction = model.predict([features])[0]
        print("💰 Predicted Price:", prediction)

        return jsonify({"estimated_price": round(prediction, 2)})

    except Exception as e:
        print(f"❌ Error in Prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

def fetch_from_osm(location_name):
    """Fetches coordinates from OpenStreetMap API"""
    try:
        response = requests.get(NOMINATIM_API_URL, params={"q": location_name, "format": "json"}, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data:
            lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
            return lat, lon
    except Exception as e:
        print(f"🌍 OSM Fetch Error: {e}")
    return None

@app.route("/get_location_coords", methods=["GET"])
def get_location_coords():
    location = request.args.get("location", "").strip()

    if not location:
        return jsonify({"error": "Location not provided"}), 400

    coords = get_location_from_db(location)
    if coords:
        return jsonify({"lat": coords[0], "lon": coords[1]})

    coords = fetch_from_osm(location)
    if coords:
        save_location_to_db(location, coords[0], coords[1])
        return jsonify({"lat": coords[0], "lon": coords[1]})

    return jsonify({"error": "Location not found"}), 404

@app.route("/get_house_prices", methods=["GET"])
def get_house_prices():
    if is_render:
        query = "SELECT latitude, longitude, price FROM properties"
    else:
        query = "SELECT latitude, longitude, price FROM properties"
    
    rows = execute_db_query(query)
    house_prices = [{"lat": row[0], "lng": row[1], "price": row[2]} for row in rows]
    return jsonify(house_prices)

@app.route("/get_price_chart_data", methods=["GET"])
def get_price_chart_data():
    price_chart_data = {
        "locations": ["HSR Layout", "Indiranagar", "Whitefield", "Jayanagar"],
        "prices": [120, 150, 95, 110]
    }
    return jsonify(price_chart_data)

@app.route('/get_nearby_places', methods=['GET'])
def get_nearby_places():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    place_type = request.args.get('type')
    
    if not lat or not lon or not place_type:
        return jsonify({"error": "Missing parameters"}), 400
    
    overpass_url = f"https://overpass-api.de/api/interpreter?data=[out:json];node[amenity={place_type}](around:2000,{lat},{lon});out;"

    try:
        response = requests.get(overpass_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch data from Overpass API"}), 500

        data = response.json()
        places = []
        for element in data.get('elements', []):
            if 'lat' in element and 'lon' in element and 'tags' in element:
                places.append({
                    "name": element['tags'].get('name', 'Unknown'),
                    "lat": element['lat'],
                    "lon": element['lon']
                })

        return jsonify({"places": places})

    except Exception as e:
        print(f"Error fetching nearby places: {e}")
        return jsonify({"error": "Request failed"}), 500

@app.route("/save_favorite", methods=["POST"])
@login_required
def save_favorite():
    data = request.get_json()
    new_fav = Favorite(
        user_id=current_user.id,
        location=data["location"],
        sqft=data["sqft"],
        bhk=data["bhk"],
        bath=data["bath"],
        property_age=data["propertyAge"],
        price=data["price"]
    )
    db.session.add(new_fav)
    db.session.commit()
    return jsonify({"message": "Property saved to favorites!"})

@app.route("/get_favorites", methods=["GET"])
@login_required
def get_favorites():
    favorites = Favorite.query.filter_by(user_id=current_user.id).all()
    fav_list = [{
        "id": fav.id,
        "location": fav.location,
        "sqft": fav.sqft,
        "bhk": fav.bhk,
        "bath": fav.bath,
        "property_age": fav.property_age,
        "price": fav.price
    } for fav in favorites]
    return jsonify({"favorites": fav_list})

@app.route("/delete_favorite/<int:fav_id>", methods=["DELETE"])
@login_required
def delete_favorite(fav_id):
    favorite = Favorite.query.get_or_404(fav_id)
    if favorite.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403

    db.session.delete(favorite)
    db.session.commit()
    return jsonify({"message": "Favorite deleted successfully"})

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    existing_user = User.query.filter_by(email=data["email"]).first()
    if existing_user:
        return jsonify({"error": "User already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(data["password"]).decode('utf-8')
    new_user = User(name=data["name"], email=data["email"], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully!"})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data["email"]).first()

    if user and bcrypt.check_password_hash(user.password, data["password"]):
        login_user(user)
        return jsonify({"message": "Login successful", "user": user.name})
    return jsonify({"error": "Invalid email or password"}), 401

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

@app.route("/check_session", methods=["GET"])
def check_session():
    if current_user.is_authenticated:
        return jsonify({"logged_in": True, "user": current_user.name})
    return jsonify({"logged_in": False})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=not is_render)  # Disable debug mode in production
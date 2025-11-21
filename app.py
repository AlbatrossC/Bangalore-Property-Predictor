from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import os
import time
import json
import pickle
import numpy as np
import requests
from sqlalchemy import text

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- Database Configuration ---
# Fix Render PostgreSQL URL
db_url = os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url or 'postgresql://postgres:password@localhost/banglore_property'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', 'dev_secret_key')

db = SQLAlchemy(app)
CORS(app)

# --- APIs ---
NOMINATIM_API_URL = "https://nominatim.openstreetmap.org/search"

# --- Global Variables for ML Model ---
model = None
data_columns = []
locations = []

# --- Load Model (FIXED: Uses absolute paths & runs globally) ---
def load_saved_artifacts():
    global model, data_columns, locations
    print("⏳ Loading model artifacts...")
    try:
        # Get absolute path to the directory where app.py is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        path_model = os.path.join(base_dir, "banglore_home_prices_model.pickle")
        path_cols = os.path.join(base_dir, "columns.json")
        
        if os.path.exists(path_model) and os.path.exists(path_cols):
            with open(path_cols, "r") as f:
                data_columns = json.load(f)["data_columns"]
                # First 3 columns are total_sqft, bath, bhk, property_age etc.
                # Adjust index based on your actual columns.json structure
                # Usually locations start after the non-location features
                locations = data_columns[3:] # Adjust this index if needed based on your json
            
            with open(path_model, "rb") as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully.")
        else:
            print(f"⚠️ Model files not found at: {path_model}")
            print("⚠️ Prediction will use fallback logic.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# 🔥 CRITICAL FIX: Call this HERE so it runs on Render/Gunicorn
load_saved_artifacts()

# --- Database Models ---
class HeatmapData(db.Model):
    __tablename__ = 'heatmap_data'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(200), nullable=False, unique=True, index=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

# --- Database Initialization (Runs on Startup) ---
with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables created.")
        
        # Sync Postgres sequence (Fix for duplicate key errors)
        print("🔧 Syncing Database ID Sequence...")
        db.session.execute(text("SELECT setval('heatmap_data_id_seq', COALESCE((SELECT MAX(id)+1 FROM heatmap_data), 1), false);"))
        db.session.commit()
        print("✅ Database Sequence Synced.")
    except Exception as e:
        print(f"ℹ️ Database setup note: {e}")
        # Don't rollback here necessarily, just continue if DB is already set

# --- Helper Functions ---
def get_location_coords_db(location_name):
    try:
        res = HeatmapData.query.filter(
            db.func.lower(HeatmapData.location) == location_name.strip().lower()
        ).first()
        return (res.latitude, res.longitude) if res else None
    except Exception:
        return None

def save_location_db(location_name, lat, lon):
    try:
        exists = HeatmapData.query.filter(
            db.func.lower(HeatmapData.location) == location_name.strip().lower()
        ).first()
        
        if not exists:
            new_loc = HeatmapData(location=location_name.strip(), latitude=lat, longitude=lon)
            db.session.add(new_loc)
            db.session.commit()
            print(f"💾 Saved coordinates for: {location_name}")
    except Exception as e:
        db.session.rollback()
        print(f"⚠️ DB Save Error: {e}")

# --- Routes ---

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/get_location_names')
def get_location_names():
    # Check if locations loaded, if not retry loading
    global locations
    if not locations:
        load_saved_artifacts()
        
    response = jsonify({'locations': locations})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        total_sqft = float(data.get('total_sqft'))
        location = data.get('location')
        bhk = int(data.get('bhk'))
        bath = int(data.get('bath'))
        age = int(data.get('property_age', 0))

        if model:
            try:
                loc_index = data_columns.index(location.lower())
            except:
                loc_index = -1

            x = np.zeros(len(data_columns))
            x[0] = total_sqft
            x[1] = bath
            x[2] = bhk
            # Check your columns.json to see where 'age' fits or if it was used in training
            # Assuming standard Banglore dataset often used: x[0]=sqft, x[1]=bath, x[2]=bhk
            
            if loc_index >= 0:
                x[loc_index] = 1

            price = model.predict([x])[0]
        else:
            # Fallback
            price = (total_sqft * 5000) + (bhk * 500000)
            
        # Apply Tier Logic
        tier_multipliers = {'Whitefield': 1.2, 'Indiranagar': 1.5, 'Koramangala': 1.4}
        multiplier = tier_multipliers.get(location, 1.0)
        final_price = max(0, price * multiplier)

        return jsonify({
            'estimated_price': round(final_price, 2),
            'details': {'location_tier_mult': multiplier}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/get_location_coords", methods=["GET"])
def get_location_coords():
    location = request.args.get("location", "").strip()
    if not location:
        return jsonify({"error": "Location required"}), 400

    # 1. Check Database
    coords = get_location_coords_db(location)
    if coords:
        return jsonify({"lat": coords[0], "lon": coords[1], "source": "db"})

    # 2. Fetch from Nominatim (OSM)
    try:
        headers = {'User-Agent': 'BangalorePropPredictor/1.0'}
        params = {'q': f"{location}, Bangalore, India", 'format': 'json', 'limit': 1}
        response = requests.get(NOMINATIM_API_URL, params=params, headers=headers, timeout=5)
        data = response.json()

        if data:
            lat, lon = float(data[0]['lat']), float(data[0]['lon'])
            save_location_db(location, lat, lon)
            return jsonify({"lat": lat, "lon": lon, "source": "api"})
            
    except Exception as e:
        print(f"Nominatim Error: {e}")

    return jsonify({"error": "Coordinates not found"}), 404

@app.route('/get_nearby_places', methods=['GET'])
def get_nearby_places():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    p_type = request.args.get('type', 'school')
    radius = request.args.get('radius', 2000)

    if not lat or not lon:
        return jsonify({'error': 'Missing lat/lon'}), 400

    type_map = {
        'school': '"amenity"~"school|college|university"',
        'hospital': '"amenity"~"hospital|clinic"',
        'restaurant': '"amenity"~"restaurant|cafe"',
        'mall': '"shop"~"mall|supermarket"',
        'park': '"leisure"="park"'
    }
    
    query_filter = type_map.get(p_type, f'"amenity"="{p_type}"')

    overpass_query = f"""
    [out:json][timeout:25];
    (
      node[{query_filter}](around:{radius},{lat},{lon});
      way[{query_filter}](around:{radius},{lat},{lon});
      relation[{query_filter}](around:{radius},{lat},{lon});
    );
    out center 30;
    """

    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter"
    ]

    places = []

    # Try Overpass
    for url in overpass_urls:
        try:
            resp = requests.post(url, data={'data': overpass_query}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                for el in data.get('elements', []):
                    tags = el.get('tags', {})
                    name = tags.get('name', 'Unnamed')
                    
                    if 'lat' in el:
                        plat, plon = el['lat'], el['lon']
                    elif 'center' in el:
                        plat, plon = el['center']['lat'], el['center']['lon']
                    else:
                        continue
                        
                    places.append({'name': name, 'lat': plat, 'lon': plon, 'type': p_type})
                
                if places:
                    return jsonify({'places': places})
        except Exception:
            continue

    # Fallback: Nominatim
    try:
        headers = {'User-Agent': 'BangalorePropPredictor/1.0'}
        viewbox_delta = 0.02 
        params = {
            'q': f"{p_type}",
            'format': 'json',
            'limit': 20,
            'viewbox': f"{float(lon)-viewbox_delta},{float(lat)+viewbox_delta},{float(lon)+viewbox_delta},{float(lat)-viewbox_delta}",
            'bounded': 1
        }
        resp = requests.get(NOMINATIM_API_URL, params=params, headers=headers, timeout=10)
        data = resp.json()
        for item in data:
            places.append({
                'name': item.get('display_name', '').split(',')[0],
                'lat': float(item['lat']),
                'lon': float(item['lon']),
                'type': p_type
            })
    except Exception as e:
        print(f"Nominatim Fallback Error: {e}")

    return jsonify({'places': places})

if __name__ == "__main__":
    # This block ONLY runs when you type 'python app.py' locally
    # Render does NOT run this block.
    print("Starting Python Flask Server Locally...")
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
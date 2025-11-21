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

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost/banglore_property')
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

# --- Load Model ---
def load_saved_artifacts():
    global model, data_columns, locations
    print("⏳ Loading model artifacts...")
    try:
        path_model = "banglore_home_prices_model.pickle"
        path_cols = "columns.json"
        
        if os.path.exists(path_model) and os.path.exists(path_cols):
            with open(path_cols, "r") as f:
                data_columns = json.load(f)["data_columns"]
                locations = data_columns[4:] # First 4 are numeric features
            
            with open(path_model, "rb") as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully.")
        else:
            print("⚠️ Model files not found. Prediction will use fallback logic.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# --- Database Models ---
class HeatmapData(db.Model):
    __tablename__ = 'heatmap_data'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(200), nullable=False, unique=True, index=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)

# Note: User and Favorite models kept for schema compatibility but ignored for logic
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

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
        # Double check existence to prevent race conditions
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
        print(f"⚠️ DB Save Error (likely duplicate): {e}")

# --- Routes ---

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/get_location_names')
def get_location_names():
    return jsonify({'locations': locations})

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
            x[3] = age
            if loc_index >= 0:
                x[loc_index] = 1

            price = model.predict([x])[0]
        else:
            # Fallback logic if model is missing
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
            # Save to DB for future use
            save_location_db(location, lat, lon)
            return jsonify({"lat": lat, "lon": lon, "source": "api"})
            
    except Exception as e:
        print(f"Nominatim Error: {e}")

    return jsonify({"error": "Coordinates not found"}), 404

@app.route('/get_nearby_places', methods=['GET'])
def get_nearby_places():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    p_type = request.args.get('type', 'school') # school, hospital, mall
    radius = request.args.get('radius', 2000)

    if not lat or not lon:
        return jsonify({'error': 'Missing lat/lon'}), 400

    # Map simpler types to Overpass Key-Values
    # We use stricter filters to avoid 400 Bad Request
    type_map = {
        'school': '"amenity"~"school|college|university"',
        'hospital': '"amenity"~"hospital|clinic"',
        'restaurant': '"amenity"~"restaurant|cafe"',
        'mall': '"shop"~"mall|supermarket"',
        'park': '"leisure"="park"'
    }
    
    query_filter = type_map.get(p_type, f'"amenity"="{p_type}"')

    # Construct Strict Overpass QL
    # Using f-string with minimal whitespace issues
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
            print(f"🔎 Querying Overpass: {url}")
            resp = requests.post(url, data={'data': overpass_query}, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for el in data.get('elements', []):
                    tags = el.get('tags', {})
                    name = tags.get('name', 'Unnamed')
                    
                    # Get coords (center for ways/relations, lat/lon for nodes)
                    if 'lat' in el:
                        plat, plon = el['lat'], el['lon']
                    elif 'center' in el:
                        plat, plon = el['center']['lat'], el['center']['lon']
                    else:
                        continue
                        
                    places.append({
                        'name': name,
                        'lat': plat,
                        'lon': plon,
                        'type': p_type
                    })
                
                if places:
                    print(f"✅ Found {len(places)} places via Overpass")
                    return jsonify({'places': places})
            else:
                print(f"⚠️ Overpass Error {resp.status_code}")
                
        except Exception as e:
            print(f"⚠️ Overpass Exception: {e}")

    # Fallback: Nominatim (if Overpass fails)
    print("⚠️ Switching to Nominatim Fallback")
    try:
        headers = {'User-Agent': 'BangalorePropPredictor/1.0'}
        # Viewbox is approx 2km box
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

# --- Application Startup ---
if __name__ == "__main__":
    load_saved_artifacts()
    
    with app.app_context():
        db.create_all()
        
        # -----------------------------------------------------------
        # 🔥 CRITICAL FIX FOR: duplicate key value violates unique constraint
        # This syncs the Postgres sequence with the actual data ID
        # -----------------------------------------------------------
        try:
            print("🔧 Syncing Database ID Sequence...")
            db.session.execute(text("SELECT setval('heatmap_data_id_seq', COALESCE((SELECT MAX(id)+1 FROM heatmap_data), 1), false);"))
            db.session.commit()
            print("✅ Database Sequence Synced.")
        except Exception as e:
            print(f"ℹ️ Sequence sync note: {e}")
            db.session.rollback()
        # -----------------------------------------------------------

    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
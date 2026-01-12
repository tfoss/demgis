"""Load capitals from JSON file."""
import json
import os

def load_capitals():
    """Load capitals dictionary from capitals.json."""
    capitals_path = os.path.join(os.path.dirname(__file__), 'capitals.json')
    with open(capitals_path, 'r') as f:
        caps_json = json.load(f)
    # Convert from JSON format to tuple format: {country: (city, lon, lat)}
    return {country: (data['city'], data['lon'], data['lat']) for country, data in caps_json.items()}

CAPITALS = load_capitals()

if __name__ == '__main__':
    print(f"Loaded {len(CAPITALS)} capitals")
    print(f"Turkey in CAPITALS: {'Turkey' in CAPITALS}")
    if 'Turkey' in CAPITALS:
        print(f"Turkey: {CAPITALS['Turkey']}")

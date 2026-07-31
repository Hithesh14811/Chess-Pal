from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/start-cropper", methods=["POST"])
def start_cropper():
    return jsonify({"status": "success", "message": "Cropper route works!"})

@app.route("/check", methods=["GET"])
def check():
    return jsonify({
        "status": "ok",
        "routes": [str(rule) for rule in app.url_map.iter_rules()]
    })

if __name__ == "__main__":
    print("🚀 Running test_routes.py on http://localhost:5002")
    for rule in app.url_map.iter_rules():
        print(f"✔️ {rule}")
    app.run(host="0.0.0.0", port=5002, debug=True)

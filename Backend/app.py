# backend/app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from datetime import timedelta
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

# ----- Config -----
app.config['SECRET_KEY'] = 'change-me-to-a-strong-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'change-this-too'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=6)

db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app)  # allow frontend files opened from file:// or served from another origin

# ----- Models -----
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(180), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(120), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# ----- DB init -----
@app.before_first_request
def create_tables():
    db.create_all()

# ----- API endpoints -----
@app.route('/api/auth/register', methods=['POST'])
def register():
    """
    Expects JSON: { "email": "...", "password": "...", "name": "optional" }
    """
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    name = data.get('name')

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "user with this email already exists"}), 409

    user = User(email=email, name=name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "user created", "email": user.email}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    """
    Expects JSON: { "email": "...", "password": "..." }
    Returns: { "access_token": "..." }
    """
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "invalid credentials"}), 401

    token = create_access_token(identity=user.id)
    return jsonify({"access_token": token}), 200

@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "user not found"}), 404
    return jsonify({"id": user.id, "email": user.email, "name": user.name}), 200

# quick health check
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# ----- Run -----
if __name__ == '__main__':
    # development server
    app.run(host='127.0.0.1', port=5000, debug=True)

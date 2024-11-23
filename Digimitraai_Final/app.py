from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, flash, session
from controller.manageAgent import agent_retrive_generate_response
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

app = Flask(__name__)

# Set a random secret key for session management
app.secret_key = secrets.token_hex(16)  # Generating a secure random key for session

# Dummy user database
users = {
    "admin": {"name": "Admin", "password": generate_password_hash("admin"),"role":"superadmin"},
    "pradeep": {"name": "Pradeep S", "password": generate_password_hash("pradeep"),"role":"governance"},
    "amit": {"name": "Amit S", "password": generate_password_hash("amit"),"role":"customersupport"},
    
}


@app.route("/chat", methods=["POST"])
def chat():
    # prepare the conversion chain  
    data = request.json
    user_message = data.get("message")
    user_name = data.get("username")
    role = users.get(user_name)["role"]
    print(role)
    try:
        response = agent_retrive_generate_response(role, user_message)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"response": response})

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'], role=session['role'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Authenticate user
        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session['role'] = user["role"]
            flash(f"Welcome, {user['name']}!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
  

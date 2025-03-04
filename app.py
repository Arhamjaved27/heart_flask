from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/add', methods=['POST'])
def add_numbers():
    data = request.get_json()
    num1 = data.get("num1")
    num2 = data.get("num2")

    if num1 is None or num2 is None:
        return jsonify({"error": "Missing numbers"}), 400

    try:
        result = float(num1) + float(num2)
    except ValueError:
        return jsonify({"error": "Invalid number format"}), 400

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)

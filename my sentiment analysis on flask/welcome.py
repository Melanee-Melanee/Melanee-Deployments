#a simple flask project to see the result on URL tab of web browser

from flask import Flask, jsonify, request


app =Flask(__name__)

# @app.route('/', methods=['GET'])
# def index():
#     return '<h1>welcomeee</h1>'


@app.route('/predict', methods=['GET'])
def predict():
    a = request.args.get('a', '')
    b = request.args.get('b', '')
    result = int(a) + int(b)
    print(result)


    return jsonify(result), 200

if __name__ == "__main__":
    app.run(debug=True)
    
    
  # to see the result on web brwoser type: http://127.0.0.1:5000/predict?a=4&b=6  
  # i typed 4 and 6 as my favourite inputs

#a simple sentiment analysis project
# 
from flask import Flask, jsonify, request
from textblob import TextBlob
import flask

app = Flask(__name__)

@app.route("/predict", methods=['GET'])
def predict():
    text = request.args.get("text")
 
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Polarity: {polarity}")
    print(f"Subjectivity: {subjectivity}")
    

    response = {}
    response ['response:'] ={
        'text' : str(text),
        'Sentiment': str(sentiment),
        'Polarity' : str(polarity),
        'Subjectivity': str(subjectivity)
    }

    return jsonify(response), 200



if __name__ == "__main__":
    app.run(debug=True)

# to see the result type : http://127.0.0.1:5000/predict?text=I love you
# I type 'I love you' as my favourite input in URL tab
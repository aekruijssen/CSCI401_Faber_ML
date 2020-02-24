from flask import Flask, request, jsonify

from recommendation_engine import RecommendationEngine

app = Flask(__name__)
model = RecommendationEngine()

@app.route('/', methods=['GET'])
def hello_world():
    '''
    Hello world.
    '''
    return "Hello world."

@app.route('/predict_api',methods=['POST'])
def predict_score():
    '''
    Predicts score given (user, item) pair.
    
    Sample Input (in json format):
    { 
        "user": {
            "latitude": 40,
            "longitude": -80,
            "...": (other info),
            "reviews": [
                {
                    "item_id": "A",
                    "rating": 4.5,
                    "text": "Blah."
                }
            ]
        },
        "item_id": "asdf"
    }
    '''
    content = request.json
    return jsonify(model.predict_score(content["user"], content["item_id"]))
    
if __name__ == "__main__":
    app.run(port=6010, threaded=True, host='0.0.0.0')
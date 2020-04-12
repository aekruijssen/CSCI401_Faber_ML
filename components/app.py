from flask import Flask, request, jsonify

from methods.cf.cfa_recommendation_engine import CFARecommendationEngine as RE

app = Flask(__name__)
model = RE()

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
            "location": {
                "latitude": 40,
                "longitude": -80
            },
            "...": (other info),
            "reviews": [
                {
                    "business_id": "A",
                    "rating": 4.5,
                    "content": "Blah."
                }
            ]
        },
        "business_id": "asdf"
    }
    '''
    content = request.json
    return jsonify(model.predict_score(content["user"], content["business_id"]))

@app.route('/make_recommendations', methods=['POST'])
def make_recommendations():
    '''
    Makes recommendations for items given a user.

    Sample input is the same as the obj["user"] part of the inupt above.
    '''
    content = request.json
    return jsonify(model.make_recommendations(content))

    
if __name__ == "__main__":
    app.run(port=6010, threaded=True, host='0.0.0.0')

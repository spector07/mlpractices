from flask import  Flask, render_template, jsonify, request
from flask_cors import  cross_origin, CORS
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__); # flask application created

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def prediction():
    if request.method == 'POST':
        gre_score = float(request.form['gre_score'])
        toefl_score = float(request.form['toefl_score'])
        university_rating = float(request.form['university_rating'])
        sop = float(request.form['sop'])
        lor = float(request.form['lor'])
        cgpa = float(request.form['cgpa'])
        research = request.form['research']
        if(research == 'yes'):
            research  = 1
        else:
            research = 0
        filename = 'finalized_model.pickle'
        loadmodel = pickle.load(open(filename,'rb'))
        scaler = StandardScaler()
        prediction =  loadmodel.predict(scaler.transform([[gre_score,toefl_score,university_rating,sop,lor,cgpa,research]]))
        return  render_template('results.html',prediction=round(100*prediction[0]))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

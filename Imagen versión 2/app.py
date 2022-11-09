from flask import Flask, request, jsonify, render_template, session, redirect, url_for, session
import requests
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, template_folder='template')
@app.route('/',  methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        pclass = request.form['pclass']
        sex = request.form['sex']
        age = request.form['age']
        sibsp = request.form['sibsp']
        parch = request.form['parch']
        ticket = request.form['ticket']
        fare = request.form['fare']
        cabin = request.form['cabin']
        embarked = request.form['embarked']
        return redirect(url_for('result',pclass=pclass,sex=sex,age=age,sibsp=sibsp,parch=parch,ticket=ticket,fare=fare,cabin=cabin,embarked=embarked))
    return render_template('index.html')

@app.route('/result/<int:pclass>/<int:sex>/<int:age>/<int:sibsp>/<int:parch>/<int:ticket>/<int:fare>/<int:cabin>/<embarked>', methods = ['GET','POST'])

def result(pclass,sex,age,sibsp,parch,ticket,fare,cabin,embarked):
        # Put inputs to dataframe
    model_rf = joblib.load("cfk.pkl")
    X = pd.DataFrame([[pclass,sex,age,sibsp,parch,ticket,fare,cabin,embarked]], columns = ["pclass","sex","age","sibsp","parch", "ticket","fare","cabin","embarked"])
    # Get prediction
    prediction = model_rf.predict(X)[0]
    if prediction == 1:
        prediction = "survived"
    else:
        prediction = "died"
    return render_template('result.html', res = prediction)

if __name__ =='__main__':
    app.run(debug=True,host="0.0.0.0", port=8000)
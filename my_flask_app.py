from flask import Flask,redirect,render_template,url_for,session
import numpy as np
import joblib
from flask_wtf import FlaskForm
from wtforms import SubmitField,TextAreaField,StringField
from tensorflow.keras.models import load_model

def return_prediction(model,scaler,sample_json):
    sep_len=sample_json['sepal_length']
    sep_wid=sample_json['sepal_width']
    pet_len=sample_json['petal_length']
    pet_wid=sample_json['petal_width']
    
    flower=[[sep_len,sep_wid,pet_len,pet_wid]]
    flower=scaler.transform(flower)
    classes=np.array(['setosa', 'versicolor', 'virginica'])
    prediction=model.predict(flower)
    prediction_class_index=np.argmax(prediction,axis=1)
    return classes[prediction_class_index][0]

app=Flask(__name__)
app.config["SECRET_KEY"]='mysecretkey'

class flowerform(FlaskForm):

    sep_len=StringField("Sepal Length")
    sep_wid=StringField("Sepal Width")
    pet_len=StringField("Petal Length")
    pet_wid=StringField("Petal Width")

    submit=SubmitField("Analyse")

@app.route("/", methods=['GET','POST'])
def index():
    form=flowerform()

    if form.validate_on_submit():

        session["sep_len"]=form.sep_len.data
        session["sep_wid"]=form.sep_wid.data
        session["pet_len"]=form.pet_len.data
        session["pet_wid"]=form.pet_wid.data
        return redirect(url_for("prediction"))
    
    return render_template("home.html",form=form)

flower_model=load_model("myModel.h5")
flower_scaler=joblib.load("myScaler.pk1")

@app.route("/prediction")
def prediction():
    content={}
    content["sepal_length"]=float(session["sep_len"])
    content["sepal_width"]=float(session["sep_wid"])
    content["petal_length"]=float(session["pet_len"])
    content["petal_width"]=float(session["pet_wid"])
    results=return_prediction(flower_model,flower_scaler,content)
    return render_template("prediction.html",results=results)

if __name__=='__main__':
    app.run()
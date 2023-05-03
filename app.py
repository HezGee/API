from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('Stroke.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Stroke Disease Prediction App"

@app.route('/predict',methods=['POST'])

#'gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status'
#[0,108,1,0,250,160,1,1.5] 0
#[3,150,0,0,233,145,1,2.3] 1

def predict():
    gender= request.form.get('gender')
    age = request.form.get('age')
    hypertension = request.form.get('hypertension')
    heart_disease = request.form.get('heart_disease')
    ever_married = request.form.get('ever_married')
    work_type = request.form.get('work_type')
    Residence_type = request.form.get('Residence_type')
    avg_glucose_level = request.form.get('avg_glucose_level')
    bmi = request.form.get('bmi')
    smoking_status = request.form.get('smoking_status')

    #result = {'gender':gender,'age':age,'hypertension':hypertension,'heart_disease':heart_disease,'ever_married':ever_married,'work_type':work_type,'Residence_type':Residence_type,'avg_glucose_level':avg_glucose_level,'bmi':bmi,'smoking_status':smoking_status}

    input_query = np.array([[gender,age,hypertension,heart_disease,ever_married,Residence_type,avg_glucose_level,bmi,smoking_status]])

    result = model.predict(input_query)[0]

    return jsonify({'Stroke_Disease': str(result)})

    

if __name__=='__main__':
    app.run(debug=True)

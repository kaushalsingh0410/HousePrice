import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template,redirect,url_for,jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=('GET','POST'))
def predict():
    # if request.method == 'POST':     #
    if request.method =='POST':
        CRIM = float(request.form['CRIM'])
        ZN = float(request.form['ZN'])
        INDUS = float(request.form['INDUS'])
        CHAS = float(request.form['CHAS'])
        NOX = float(request.form['NOX'])
        RM = float(request.form['RM'])
        AGE = float(request.form['AGE'])
        DIS = float(request.form['DIS'])
        RAD = float(request.form['RAD'])
        TAX = float(request.form['TAX'])
        PTRATIO = float(request.form['PTRATIO'])
        B = float(request.form['B'])
        LSTAT = float(request.form['LSTAT'])

        print('formdata',request.form)

        # data = pd.DataFrame([[CRIM, ZN,	INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
        #                     columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
        #                              'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
        
        
        
        data = pd.DataFrame([[CRIM, ZN,	INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
                            columns=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
                                        'ptratio', 'b', 'lstat'])
        
        
        # lin_reg = joblib.load('real.joblib')
        lin_reg = joblib.load('Real.joblib')

        numerical_transformer = joblib.load('Labels.joblib')
        column_transformer = joblib.load('Pipeline_house.joblib')
        Scaler = joblib.load('Scaler.joblib')


        # print('type1',type(column_transformer))
        # data = data.astype('object')
        # print('type1',type(column_transformer))
        # prepared_data = column_transformer.transform(data)



        prepared_data = Scaler.transform(data)


        output = lin_reg.predict(prepared_data)
        print('output',output)
        

        # final_output = numerical_transformer.inverse_transform([output])
        # final_output = '{:.2f}'.format(final_output[0, 0])

        # return render_template('index.html', result=final_output)
        response = {
            'result': round(output[0], 2)  # Round the output to 2 decimal places
        }
        
        return jsonify(response)
    return redirect(url_for('index'))




if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, url_for, request
app = Flask(__name__)


@app.route("/")

@app.route("/introduction")
def introduction():
    return render_template('introduction.html')

@app.route("/form")
def form():
    return render_template('forms.html')

@app.route("/benign")
def benign():
    return render_template('benign.html')

@app.route("/malignant")
def malignant():
    return render_template('malignant.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/signin")
def signin():
    return render_template('signin.html')

@app.route("/signup")
def signup():
    return render_template('signup.html')

@app.route("/ADoubleGordianKnot")
def problem():
    return render_template('problem.html')

@app.route("/HowPatalahaimstoovercomeproblems")
def solution():
    return render_template('solution.html')

@app.route("/support")
def support():
    return render_template('support.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

    import numpy as np
    import pandas as pd
    
    
    dataset = pd.read_csv('wdbcdata(edited).csv')
    X = dataset.iloc[:, 1:31].values
    Y = dataset.iloc[:, 31].values
    

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:31])
    X[:, 1:31] = imputer.transform(X[:, 1:31])

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    

    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn. ensemble import VotingClassifier

    lr = LogisticRegression()
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    svm = SVC(kernel = 'linear')
    ksvm = SVC(kernel = 'rbf')

    evc = VotingClassifier( estimators= [('lr',lr),('knn',knn),('svm',svm),('ksvm',ksvm)], voting = 'hard')
    evc.fit(X,Y)
    
    v = []

    for i in request.form.values():
        v.append(i)

    a=[]
    my_prediction = [-1]
    a.append(v)

    a = sc.transform(a)
    my_prediction = evc.predict(a)
    print(my_prediction)


    if my_prediction == 1:
        return render_template('malignant.html')
    elif my_prediction == 0:
        return render_template('benign.html' )

    

if __name__ == '__main__':
    app.run(debug=True)
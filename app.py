from flask import Flask,render_template,request,url_for,redirect
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    # return "Hello World"
    print("aaaaaaaaaa")
    return render_template('index.html')

# @app.route('/predict')
# def predict():
#     data = ['This is so bad']
#     with open('CountVector_stock_data.hd5', 'rb') as f:
#         vectorizer = pickle.load(f)
#     with open('Logis_stock_data.hd5', 'rb') as f:
#         logis = pickle.load(f)

#     data_lower = [sentence.lower() for sentence in data]

#     vectorized_data = vectorizer.transform(data_lower)

#     scaled_data = vectorized_data 

#     predict_ans = logis.predict(scaled_data)

#     return str(predict_ans)

@app.route('/predict_from', methods=['POST', 'GET'])
def predict_from():
    if request.method == 'POST':
        selected_option = request.form['option']
        sentences = request.form['keyword']

        if selected_option == 'option1':
            with open('CountVector_stock_data.hd5', 'rb') as f:
                vectorizer = pickle.load(f)

            with open('Logis_stock_data.hd5', 'rb') as f:
                model = pickle.load(f)
            data_lower = [sentences.lower()]
            vectorized_data = vectorizer.transform(data_lower)
            scaled_data = vectorized_data

            predict_ans = model.predict(scaled_data)

            return render_template('index.html', pre = predict_ans[0])
        elif selected_option == 'option2':
            # Load the model and training data
            print("asdasdasd")
            # Load the model and training data
            # Load the model

            # โหลดโมเดล

            # Load the model
            with open('naive_bayes_model.pkl', 'rb') as model_file:
                model = pickle.load(model_file)

            # Load the 'TfidfVectorizer'
            with open('data_n2.pkl', 'rb') as vectorizer_file:
                vectorizer = pickle.load(vectorizer_file)

            sentences = "Your sentence here"  # Replace with your actual sentence
            data_lower = [sentences.lower()]

            # Transform the input data using the loaded vectorizer
            vectorized_data = vectorizer.transform(data_lower)

            # Convert the sparse matrix to a dense NumPy array
            dense_vectorized_data = vectorized_data.toarray()

            # Predict using the loaded model
            predict_ans = model.predict(dense_vectorized_data)
            return render_template('index.html', pre=predict_ans[0])



        else:
            return "Option ไม่ถูกต้อง"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
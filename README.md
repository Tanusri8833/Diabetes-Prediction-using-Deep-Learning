# 🩺 Diabetes Predictor

This project is a machine learning-based web application that predicts whether a person is likely to have diabetes, using medical attributes as inputs. The model is trained on the Pima Indians Diabetes Dataset and deployed via a simple web interface using **Streamlit**.

## 📂 Project Structure

```
.
├── app.py                    # Streamlit web app script
├── Diabetes Prediction.ipynb # Jupyter notebook for data analysis and model training
├── diabetes.csv              # Dataset used for training and testing
└── README.md                 # Project documentation
```

---

## 🚀 Features

- User-friendly interface to input medical data
- Predicts diabetes based on model trained on 768 real cases
- Built using Python, Pandas, Scikit-learn, and Streamlit
- Clean visualizations and interactive form for input

---

## 📊 Dataset

The dataset (`diabetes.csv`) consists of the following medical parameters:

```
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0: No diabetes, 1: Diabetes)
```

Source: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## ⚙️ Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/ImSahil25/dDiabetes-Prediction-using-Deep-Learning.git
   cd Diabetes-Prediction-using-Deep-Learning
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not present, install manually:
   ```
   pip install pandas numpy scikit-learn streamlit
   ```

3. **Run the app:**
   ```
   streamlit run app.py
   ```

---

## 📌 Usage

1. Open the app in a browser after running.
2. Input the required medical data in the form.
3. The model will display whether the patient is likely to have diabetes.

---

## 🔮 Future Scope

- Integrate more sophisticated models like XGBoost or Neural Networks
- Add login/authentication system
- Visual analytics dashboard for healthcare professionals
- Deploy using cloud platforms (e.g., AWS, Heroku)

---

## 🧩 Real-World Applications

- Assist doctors in early diabetes diagnosis
- Patient self-assessment tool
- Health camps and screening programs

---

## 🤔 Known Issues / Challenges

- Dataset has missing or zero values that need careful treatment
- Limited size of dataset may affect generalizability
- Basic UI – can be enhanced with CSS or React integration

---

## 👨‍💻 Author

**Tanusri Jana**  
Feel free to reach out for improvements, collaborations, or feedback!

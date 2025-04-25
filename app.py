import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
rf_model = joblib.load('credit_risk_model.pkl')

# Add custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stSidebar {
            background-color: #f0f0f0;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #333333;
        }
        .subtitle {
            font-size: 18px;
            color: #666666;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<div class='title'>⚖️ RiskRater - Credit Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>This app predicts whether a loan applicant is a <b>Good</b> or <b>Bad Credit Risk</b> based on their details.</div>", unsafe_allow_html=True)

# Sidebar for instructions
st.sidebar.header("📋 Applicant Information")
st.sidebar.markdown("""
### Instructions:
1. Enter the applicant's details in the sidebar.
2. Click on the **Predict Credit Risk** button.
3. View the prediction and explanation below.

### About the Model:
- Trained using the German Credit Dataset.
- Predicts credit risk based on financial and personal details.
""")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.selectbox("Job (1=Unskilled, 2=Skilled, 3=Highly Skilled)", ["1", "2", "3"])
    housing = st.selectbox("Housing", ["own", "free", "rent"])

with col2:
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich", "NA"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "NA"])
    credit_amount = st.number_input("Credit Amount (€)", min_value=0, value=1000)
    duration = st.slider("Loan Duration (Months)", 6, 72, 12)

purpose = st.selectbox("Purpose of Loan", ["radio/TV", "education", "furniture/equipment", "business", "car"])

# Prepare input data
input_data = pd.DataFrame([{
    'Age': age,
    'Sex': 0 if sex == "male" else 1,
    'Job': int(job) - 1,
    'Housing': {"own": 0, "free": 1, "rent": 2}[housing],
    'Saving accounts': {"little": 0, "moderate": 1, "rich": 2, "quite rich": 3, "NA": 4}[saving_accounts],
    'Checking account': {"little": 0, "moderate": 1, "rich": 2, "NA": 3}[checking_account],
    'Credit amount': credit_amount,
    'Duration': duration,
    'Purpose': {"radio/TV": 0, "education": 1, "furniture/equipment": 2, "business": 3, "car": 4}[purpose],
    'Credit_Duration_Ratio': credit_amount / duration
}])

# Predict button
if st.button("🔍 Predict Credit Risk"):
    prediction = rf_model.predict(input_data)
    st.write("### Prediction Result")
    if prediction[0] == 1:
        st.success("✅ Good Credit Risk")
        st.info("The applicant is likely to repay the loan on time.")
    else:
        st.error("❌ Bad Credit Risk")
        st.warning("The applicant may struggle to repay the loan.")

    st.write("### Model Info")
    st.markdown("""
    - Trained with Random Forest Classifier  
    - Based on German Credit Dataset  
    - Feature importance was visualized to interpret results  
    """)

# Footer
st.sidebar.markdown("""
---
**Developed by Ananya Dikshit**
""")
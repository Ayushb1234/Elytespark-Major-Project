import streamlit as st


st.title("Business Insights")


st.header("Key Findings")

st.markdown(
    """
    ### Healthcare Risk Factors

    - Glucose and HbA1c levels are important metabolic indicators.
    - BMI provides information about potential obesity-related risk.
    - Blood-pressure measurements contribute to cardiovascular risk analysis.
    - Cholesterol, HDL, LDL and triglycerides provide additional cardiovascular information.
    - Smoking, physical activity, sleep and stress provide lifestyle-related signals.
    - Family history provides additional risk context.
    """
)


st.header("Recommended Actions")

st.markdown(
    """
    **1. Early screening**

    High-risk patients can be prioritized for additional medical assessment.

    **2. Lifestyle monitoring**

    Patients with unfavorable lifestyle indicators can be encouraged
    to improve physical activity, sleep and dietary habits.

    **3. Clinical review**

    Predictions should be reviewed by qualified healthcare professionals.

    **4. Continuous monitoring**

    Model performance should be monitored as new patient data becomes available.
    """
)


st.warning(
    "This system is a decision-support prototype and should not replace professional medical diagnosis."
)

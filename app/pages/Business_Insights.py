import streamlit as st

st.title("💡 Business Insights")

st.markdown(
    """
    ### Key observations
    - Glucose, HbA1c, BMI, and blood-pressure-related features are strong predictors.
    - Lifestyle and family-history variables also contribute meaningfully.
    - SHAP/LIME can be used to explain individual patient predictions.

    ### Recommendations
    - Prioritize early screening for high-risk patients.
    - Monitor patients with elevated glucose and HbA1c.
    - Encourage lifestyle changes: sleep, activity, diet, and stress management.
    - Use the model as a clinical decision-support tool, not as a replacement for doctors.
    """
)

st.subheader("Risk management actions")
st.write(
    """
    - Deploy the dashboard for early triage.
    - Flag high-confidence abnormal cases for manual review.
    - Track model predictions over time for auditability.
    - Retrain with fresh data when more records are available.
    """
)
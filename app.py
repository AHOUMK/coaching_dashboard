import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Coaching Forecast & Recommendation App")

st.markdown("""
Upload your KPI data to forecast the next QS_Adoption_Score and automatically generate coaching recommendations.
""")

uploaded_file = st.file_uploader("Upload your cleaned KPI CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Data Preview")
    st.dataframe(df.head())

    required_columns = ["AHT", "ACW", "RONA", "Schedule_Adherence", "QS_Adoption_Score"]
    if not all(col in df.columns for col in required_columns):
        st.error("The uploaded file is missing one or more required columns.")
    else:
        df["Next_QS_Score"] = df["QS_Adoption_Score"].shift(-1)
        df = df.dropna(subset=["Next_QS_Score"]).reset_index(drop=True)

        X = df[required_columns]
        y = df["Next_QS_Score"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.subheader("2. Model Evaluation")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        st.subheader("3. Feature Importance")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": required_columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        st.bar_chart(imp_df.set_index("Feature"))

        st.subheader("4. Actual vs Predicted QS Score")
        fig, ax = plt.subplots()
        ax.plot(df["QS_Adoption_Score"].values, label="Actual QS", marker="o")
        df["Predicted_Next_QS"] = model.predict(scaler.transform(X))
        ax.plot(df["Predicted_Next_QS"].values, label="Predicted Next QS", marker="x")
        ax.axhline(90, color="red", linestyle="--", label="Coaching Threshold")
        ax.set_xlabel("Index")
        ax.set_ylabel("QS Score")
        ax.set_title("Actual vs Predicted Next QS Score")
        ax.legend()
        st.pyplot(fig)

        st.subheader("5. Coaching Report")

        df["Coaching_Recommended"] = df["Predicted_Next_QS"].apply(lambda x: "Yes" if x < 90 else "No")

        def coaching_reason(row):
            if row["Predicted_Next_QS"] < 90:
                if row["RONA"] > 1:
                    return "RONA too high"
                elif row["AHT"] > 500:
                    return "AHT above threshold"
                elif row["ACW"] > 150:
                    return "ACW too long"
                else:
                    return "Low predicted QS score"
            return "On Track"

        df["Coaching_Reason"] = df.apply(coaching_reason, axis=1)

        report_df = df[[
            "Date", "AHT", "ACW", "RONA", "Schedule_Adherence", 
            "QS_Adoption_Score", "Predicted_Next_QS", 
            "Coaching_Recommended", "Coaching_Reason"
        ]] if "Date" in df.columns else df[[
            "AHT", "ACW", "RONA", "Schedule_Adherence", 
            "QS_Adoption_Score", "Predicted_Next_QS", 
            "Coaching_Recommended", "Coaching_Reason"
        ]]

        st.dataframe(report_df)

        csv = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Coaching Report",
            data=csv,
            file_name="coaching_report.csv",
            mime="text/csv"
        )
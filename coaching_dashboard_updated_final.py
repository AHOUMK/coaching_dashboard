import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Set wide layout
st.set_page_config(layout="wide")

# Title and instructions
st.title("Advisor Coaching Prediction Dashboard")
st.markdown("Upload your KPI data to predict which advisors may need coaching based on QS_Adoption_Score.")

# File uploader
uploaded_file = st.file_uploader("Upload cleaned KPI CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Data Preview")
    st.dataframe(df.head())

    features = ["AHT", "ACW", "RONA", "Schedule_Adherence"]
    if not all(col in df.columns for col in features + ["QS_Adoption_Score"]):
        st.error("Missing required columns in the CSV. Please check your data.")
    else:
        # Prepare target variable
        df = df.dropna(subset=features + ["QS_Adoption_Score"])
        df["Needs_Coaching_Predicted"] = (df["QS_Adoption_Score"] < 90).astype(int)
        X = df[features]
        y = df["Needs_Coaching_Predicted"]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluation report
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.subheader("2. Model Performance")
        st.dataframe(report_df)

        # Feature importance
        st.subheader("3. Feature Importance")
        feature_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        st.bar_chart(feature_imp)

        # Predict on full dataset
        df["Prediction_Probability"] = model.predict_proba(scaler.transform(df[features]))[:, 1]
        df["Prediction_Label"] = df["Needs_Coaching_Predicted"].apply(lambda x: "Needs Coaching" if x == 1 else "On Track")

        # Filters
        st.subheader("4. Filter & View Results")
        filter_choice = st.selectbox("Show advisors who are:", ["All", "Needs Coaching", "On Track"])
        if filter_choice != "All":
            df_filtered = df[df["Prediction_Label"] == filter_choice]
        else:
            df_filtered = df

        st.dataframe(df_filtered[["Date", "QS_Adoption_Score", "Prediction_Label", "Prediction_Probability"] + features])

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Predictions CSV", csv, "coaching_predictions_output.csv", "text/csv")

        # Visualizations
        st.subheader("5. QS_Adoption_Score vs KPIs")
        for col in features:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col, y="QS_Adoption_Score", hue="Prediction_Label", style="Prediction_Label", palette="Set1", ax=ax)
            plt.title(f"{col} vs QS_Adoption_Score")
            st.pyplot(fig)
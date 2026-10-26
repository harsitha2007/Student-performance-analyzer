import streamlit as st
import pandas as pd 
from sklearn.linear_model
import LinearRegression
st.set_page_config(page_title="Student Performance Analyzer",
                   page_icon="Heart",
                   layout="Wide"
                  )
st.title("Heart Student Perfomance Analyzer")
st.caption("A BTS-Themed Data Science learning Project")
st.info("This project Uses fictional practice data.")
data = {"name":["JK","jimin","Namjoon","yoongi","jin","JHope"],"study_hours":[2,4,8,7,6,5],"attendance":[65,70,95,50,78,68],"marks":[55,64,80,88,69,67]}
df = pd.DataFrame(data)
col1,col2,col3 = st.column(3)
col1.metric("Students",len(df))
col2.metric("Average Marks",f"{df['Marks'].mean():.1f}")
col3.metric("Average Study Hours",f"{df['study_hours'].mean():.1f}")
st.subheader("Student Marks")
st.bar_chart(df.set_index("name")["marks"])
st.subheader("Study Hours VS Marks")
st.scatter_chart(df,x="study_hours",y="marks",color="attendance")
st.subheader("📋 Student Dataset")
st.dataframe(df, use_container_width=True)

X = df[["study_hours", "attendance"]]
y = df["marks"]

model = LinearRegression()
model.fit(X, y)

st.subheader("🤖 Try a Prediction")

hours = st.slider("Study hours per day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 85)

if st.button("Estimate Marks"):
    new_student = pd.DataFrame({
        "study_hours": [hours],
        "attendance": [attendance]
    })

    prediction = model.predict(new_student)[0]
    prediction = max(0, min(100, prediction))

    st.metric("Estimated Marks", f"{prediction:.1f} / 100")
    st.warning(
        "Demonstration only: this model uses six fictional "
        "records and is not validated for real-world predictions."
    )

st.divider()
st.caption("Built with Python, Pandas, scikit-learn and Streamlit")

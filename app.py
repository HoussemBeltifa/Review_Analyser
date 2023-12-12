import streamlit as st
from ML import *

st.set_page_config(page_title="Review Scanner", page_icon="✨")


st.title("Welcome to the Review Scanner")
st.subheader("A place to analyze text reviews")
st.subheader("Created by Mohamed Houssem Beltifa")
st.write("Enter your text in the box below and click 'Submit' to analyze the review.")
st.write("Note : the accuracy of the model is just 50%")

review_text = st.text_area("Enter your review here:", max_chars=500)


if st.button("Submit"):
    result = analyze_review(review_text)

    st.success(f"Analysis Result: {result}")




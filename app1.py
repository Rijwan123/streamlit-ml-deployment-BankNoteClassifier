import streamlit as st
import pickle
from pydantic import BaseModel
from BankNotes import BankNote

def load_classifier():
    with open('classifier.pkl', 'rb') as file:
        classifier = pickle.load(file)
    return classifier

def main():
    st.title("Bank Note Classifier")
    st.write("Enter the values for the banknote features:")

    variance = st.number_input("Variance", step=0.01)
    skewness = st.number_input("Skewness", step=0.01)
    curtosis = st.number_input("Curtosis", step=0.01)
    entropy = st.number_input("Entropy", step=0.01)

    banknote = BankNote(variance=variance, skewness=skewness, curtosis=curtosis, entropy=entropy)

    if st.button("Predict"):
        classifier = load_classifier()
        features = [[banknote.variance, banknote.skewness, banknote.curtosis, banknote.entropy]]
        prediction = classifier.predict(features)
        #st.write("Prediction:", prediction)

        print('prediction===', prediction)
        print('prediction[0]===', prediction[0])

        if (prediction[0] > 0.5):
            prediction = 'Fake Note'
        else:
            prediction = "It's a Bank note"
        return {
            st.success('Prediction: {} '.format(prediction))
        }

if __name__ == "__main__":
    main()

    # To test Streamlit API, run the command below on the terminal-
    #run:  streamlit run app1.py

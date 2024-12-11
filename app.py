import os
import json
import datetime
import random
import csv
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("punkt")


# Load intents
def load_intents(csv_file: str = None):
    file_path = os.path.abspath("intents.json")
    with open(file_path, "r") as file:
        intents = json.load(file)

        return intents


# Train model
def train_model(intents: dict):
    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    regressor = LogisticRegression(random_state=0, max_iter=10000)

    tags = []
    patterns = []
    for intent in intents:
        for pattern in intent["patterns"]:
            tags.append(intent["tag"])
            patterns.append(pattern)

    x = vectorizer.fit_transform(patterns)
    y = tags
    regressor.fit(x, y)
    return vectorizer, regressor


# Chatbot response function
@st.cache_resource()
def chatbot(input_text):
    intents = load_intents()
    vectorizer, model = train_model(intents)
    input_vector = vectorizer.transform([input_text])
    predicted_tag = model.predict(input_vector)[0]
    for intent in intents:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I don't understand that."


# Function to log chat to CSV
def save_to_csv(user_input, response):
    if not os.path.exists("chat_log.csv"):
        with open("chat_log.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["User Input", "Chatbot Response", "Timestamp"])
    with open("chat_log.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([user_input, response, timestamp])


st.set_page_config(page_title="Chatbot", layout="centered")

counter = 0


# Streamlit UI
def main():

    global counter

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("🌿 Healthy Chat", anchor="chatbot")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area(
                "Chatbot:",
                value=response,
                height=120,
                max_chars=None,
                key=f"chatbot_response_{counter}",
            )

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ["goodbye", "bye"]:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.markdown(
                        f"**User:** {row[0]}  \n**Bot:** {row[1]}  \n*{row[2]}*",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.write(
            "The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications."
        )

        st.subheader("Project Overview:")

        st.write(
            """
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """
        )

        st.subheader("Dataset:")

        st.write(
            """
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """
        )

        st.subheader("Streamlit Chatbot Interface:")

        st.write(
            "The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input."
        )

        st.subheader("Conclusion:")

        st.write(
            "In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms."
        )


if __name__ == "__main__":
    main()

# app.py
# This is the backend server for the AI Chatbot.
# It uses Flask to create a web server and an API endpoint for the chat functionality.
# It also uses SQLite to store the chat history.

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# Configure logging to show info-level messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Flask App and CORS ---
# We create a Flask app to handle web requests.
# CORS (Cross-Origin Resource Sharing) is enabled to allow the frontend,
# which will be on a different "origin" (like a file on your computer),
# to communicate with this backend server.
app = Flask(__name__)
CORS(app)

# --- Database Setup ---
# This function initializes the SQLite database and creates a 'chat_logs' table
# if it doesn't already exist. This table will store the conversation history.
def init_db():
    try:
        conn = sqlite3.connect('chatbot.db')
        cursor = conn.cursor()
        # The table stores a timestamp, the user's message, and the bot's response.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        logging.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")

# --- Load Conversational AI Model ---
# **UPGRADE**: We are switching to a larger, more capable model to improve response quality.
# This model will provide more coherent and context-aware answers.
try:
    logging.info("Loading upgraded model and tokenizer (microsoft/DialoGPT-large)...")
    # The line below is the only change: switching from "medium" to "large"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    logging.info("Upgraded model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    # If the model can't be loaded, we'll exit or handle it gracefully.
    model = None
    tokenizer = None

# --- Chat API Endpoint ---
# This is the main API endpoint for the chatbot. It listens for POST requests at '/chat'.
@app.route('/chat', methods=['POST'])
def chat():
    if not model or not tokenizer:
        return jsonify({"error": "Model is not available"}), 500

    # Get the user's message and history from the incoming JSON request.
    data = request.get_json()
    user_message = data.get('message')
    chat_history_ids_list = data.get('chat_history_ids')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # 1. Encode the new user input, adding the end-of-string token.
        new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')

        # 2. Append the new user input tokens to the chat history.
        if chat_history_ids_list:
            history_tensor = torch.tensor([chat_history_ids_list])
            bot_input_ids = torch.cat([history_tensor, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # **IMPROVEMENT**: Truncate history to prevent context window overflow.
        max_history_length = 900
        if bot_input_ids.shape[-1] > max_history_length:
            bot_input_ids = bot_input_ids[:, -max_history_length:]

        # 3. Generate a response.
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1024, # Set max_length for the entire sequence
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )

        # 4. Decode the response, skipping the prompt to get only the new reply.
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        logging.info(f"Bot response: {bot_response}")

        # --- Log interaction to Database ---
        try:
            conn = sqlite3.connect('chatbot.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO chat_logs (user_message, bot_response) VALUES (?, ?)", (user_message, bot_response))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Database logging error: {e}")

        # Return the bot's response and the updated chat history to the frontend.
        return jsonify({
            "response": bot_response,
            "chat_history_ids": chat_history_ids.tolist()[0]
        })

    except Exception as e:
        logging.error(f"Error during chat generation: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate response"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize the database when the server starts.
    init_db()
    # Run the Flask app. `debug=True` allows for auto-reloading when you save changes.
    app.run(port=5000, debug=True)

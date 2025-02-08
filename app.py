import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer from Hugging Face's Model Hub
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Set up Streamlit interface
def chat_with_bot():
    st.title("AI Healthcare Assistant Chatbot")
    st.write("Hello! I am your healthcare assistant. Ask me about common health queries.")

    # Input field for user query
    user_input = st.text_input("You: ", "")

    if 'history' not in st.session_state:
        st.session_state.history = []

    if user_input:
        # Append the user input to history
        st.session_state.history.append(f"You: {user_input}")

        # Tokenize the new user input
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Tokenize the previous chat history (limit to last 5 exchanges)
        history_ids = []
        for chat in st.session_state.history[-5:]:
            history_ids.append(tokenizer.encode(chat + tokenizer.eos_token, return_tensors='pt'))

        # Concatenate chat history and new user input
        bot_input_ids = torch.cat(history_ids + [new_user_input_ids], dim=-1)

        # Generate a response from the model
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, top_p=0.92, top_k=50, temperature=0.7)

        # Get the chatbot's reply and decode it
        bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Append the bot's response to the history
        st.session_state.history.append(f"Assistant: {bot_output}")

        # Display the conversation history
        for chat in st.session_state.history:
            st.write(chat)

if __name__ == "__main__":
    chat_with_bot()

# ---------------------------------------------------------
# Project 1: Rule-Based AI Chatbot
# DecodeLabs - Artificial Intelligence
# ---------------------------------------------------------

print("=" * 55)
print("        RULE-BASED AI CHATBOT")
print("=" * 55)
print("Chatbot: Hello! I am your Rule-Based AI Chatbot.")
print("Chatbot: You can ask me about myself, AI, or request help.")
print("Chatbot: Type 'bye' or 'exit' to end the conversation.")
print("=" * 55)


# ---------------------------------------------------------
# KNOWLEDGE BASE
# Dictionary containing predefined intents and responses
# ---------------------------------------------------------

responses = {

    # Greeting intent
    "hello": "Hello! How can I help you?",
    "hi": "Hi there! How can I help you?",
    "hey": "Hey! Nice to meet you.",

    # Well-being intent
    "how are you": "I am doing well! Thanks for asking.",

    # Name / identity intent
    "what is your name": "I am a Rule-Based AI Chatbot.",
    "your name": "I am a Rule-Based AI Chatbot.",
    "who are you": "I am a simple chatbot that responds using predefined rules.",

    # Artificial Intelligence intent
    "what is ai": (
        "AI stands for Artificial Intelligence. "
        "It enables computers to perform tasks that normally "
        "require human intelligence."
    ),

    "define ai": (
        "AI stands for Artificial Intelligence. "
        "It enables computers to perform tasks that normally "
        "require human intelligence."
    ),

    # Help intent
    "help": (
        "You can greet me, ask my name, ask how I am, "
        "ask about AI, or say goodbye."
    ),

    # Thanks intent
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",

    # Goodbye intent
    "goodbye": "Goodbye! Have a nice day."
}


# ---------------------------------------------------------
# CONTINUOUS INPUT LOOP
# The chatbot continues running until an exit command
# is received.
# ---------------------------------------------------------

while True:

    # -----------------------------------------------------
    # INPUT SANITIZATION
    # lower() handles different letter cases.
    # strip() removes unnecessary spaces.
    # -----------------------------------------------------

    raw_input = input("You: ")
    user_input = raw_input.lower().strip()


    # -----------------------------------------------------
    # EXIT STRATEGY
    # Uses if-else decision logic and break command.
    # -----------------------------------------------------

    if user_input == "bye" or user_input == "exit":

        print("Chatbot: Goodbye! Have a nice day.")
        break

    else:

        # -------------------------------------------------
        # RESPONSE LOOKUP
        # Dictionary .get() performs direct rule matching.
        # If the input is not found, the fallback message
        # is returned.
        # -------------------------------------------------

        response = responses.get(
            user_input,
            "Sorry, I don't understand that. Please try another question."
        )

        print("Chatbot:", response)


# ---------------------------------------------------------
# END OF CHATBOT
# ---------------------------------------------------------
print("=" * 55)
print("Chatbot session ended.")
print("=" * 55)

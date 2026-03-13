"""
download_conversations.py — Massive conversational dataset builder for goktugGPT.

Sources:
  1. Synthetic — programmatically generated Q&A (math, geography, science, small talk...)
  2. Cornell Movie Dialogs  — 220K conversation exchanges
  3. DailyDialog            — 13K high-quality daily conversations
  4. PersonaChat (GitHub)   — 160K persona-based conversations
  5. Existing train.txt     — hand-crafted examples

Target: 100,000+ training lines

Usage:
  python data/download_conversations.py                  # everything
  python data/download_conversations.py --synthetic-only # fast, no internet needed
  python data/download_conversations.py --no-cornell
  python data/download_conversations.py --no-daily

After running:
  del checkpoints\\tokenizer.json
  del checkpoints\\best_model.pt
  python train.py --config tiny --data data/train_chat.txt --epochs 30
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).parent
CONV_DIR = BASE_DIR / "conversations"
QA_FILE  = BASE_DIR / "train.txt"
OUT_FILE = BASE_DIR / "train_chat.txt"

# ──────────────────────────────────────────────────────────────────────────────
# Formatting helper
# ──────────────────────────────────────────────────────────────────────────────

def fmt(user_msg: str, think: str, assistant_msg: str) -> str:
    return (f"<user> {user_msg.strip()} <assistant> "
            f"<think> {think.strip()} </think> "
            f"{assistant_msg.strip()} <eos>")

def _think(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["hi","hello","hey","good morning","good evening","good afternoon","good night","howdy","greetings","sup","yo"]):
        return "Greeting detected. Respond warmly and offer help."
    if any(w in q for w in ["how are you","how do you do","how's it","how have you","are you okay","are you good","are you alright"]):
        return "User asks about my state. Respond positively and offer help."
    if any(w in q for w in ["bye","goodbye","see you","cya","later","farewell","take care","good night","night","ttyl"]):
        return "User is saying goodbye. Respond warmly."
    if any(w in q for w in ["thank","thanks","thx","appreciate","helpful","great job","well done","you're amazing","you're great"]):
        return "User is expressing gratitude. Respond graciously."
    if any(w in q for w in ["who are you","what are you","your name","about yourself","are you ai","are you human","are you robot","are you bot"]):
        return "User asks about my identity. I am goktugGPT."
    if any(w in q for w in ["capital of","capital city"]):
        return "User asks about a country capital. Answer directly."
    if any(w in q for w in ["plus","add","sum of","what is","minus","subtract","times","multiply","divided","divide","square root","squared","percent","percentage"]):
        return "Math question. Calculate and state the answer clearly."
    if any(w in q for w in ["how do i","how can i","how to","how should"]):
        return "User asks for practical advice. Give clear, actionable steps."
    if any(w in q for w in ["who was","who is","who invented","who created","who built","who wrote","who discovered"]):
        return "User asks about a person or inventor. Provide key facts."
    if any(w in q for w in ["what is","what are","define","explain","describe","tell me about"]):
        return "User requests information. Provide a clear, accurate explanation."
    if any(w in q for w in ["why","reason","cause","because","how come"]):
        return "User asks for a reason. Explain the cause clearly."
    if any(w in q for w in ["when","what year","what date","what century"]):
        return "User asks about time or a date. Provide the information."
    if any(w in q for w in ["where","location","country","city","place","continent"]):
        return "User asks about a location. Provide geographic information."
    if "joke" in q or "funny" in q or "laugh" in q or "humor" in q:
        return "User wants humor. Share a light joke."
    if any(w in q for w in ["i'm sad","i'm tired","i'm stressed","i'm scared","i'm angry","i'm lonely","i'm bored","i'm confused","i'm lost","i'm upset","i'm worried"]):
        return "User expresses a feeling. Respond empathetically and offer support."
    return "Process the question carefully and give a helpful, accurate response."


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: GREETINGS & SMALL TALK
# ──────────────────────────────────────────────────────────────────────────────

GREETINGS = [
    ("hi", "Hi! How can I help you today?"),
    ("hello", "Hello! What can I do for you?"),
    ("hey", "Hey! What's on your mind?"),
    ("hey there", "Hey there! How can I assist?"),
    ("hello there", "Hello there! What would you like to talk about?"),
    ("hi there", "Hi there! How can I help?"),
    ("good morning", "Good morning! Hope your day is off to a great start. How can I help?"),
    ("good afternoon", "Good afternoon! What can I do for you?"),
    ("good evening", "Good evening! How can I help you tonight?"),
    ("good night", "Good night! Is there something I can help you with before you rest?"),
    ("greetings", "Greetings! It's great to meet you. How can I assist?"),
    ("howdy", "Howdy! What's up?"),
    ("sup", "Not much! How about you?"),
    ("what's up", "Not much! Just here and ready to help. What's on your mind?"),
    ("what's going on", "Nothing much, just ready to chat! What's up?"),
    ("yo", "Yo! What can I do for you?"),
    ("hi bot", "Hi! I'm goktugGPT. What can I help you with?"),
    ("hello bot", "Hello! goktugGPT at your service. What do you need?"),
    ("hey bot", "Hey! How can I assist you today?"),
    ("hello goktugGPT", "Hello! Great to meet you. What can I do for you?"),
    ("hi goktugGPT", "Hi! I'm here and ready to help. What do you need?"),
    ("start", "Hello! I'm goktugGPT. Feel free to ask me anything."),
    ("begin", "Sure, let's begin! What would you like to talk about?"),
    ("HI", "Hi! How can I help you today?"),
    ("HELLO", "Hello there! What can I do for you?"),
    ("HEY", "Hey! What's up?"),
    ("Hi!", "Hi! How are you doing? How can I help?"),
    ("Hello!", "Hello! Great to hear from you. What can I do for you?"),
    ("Hey!", "Hey! Good to see you. What's on your mind?"),
    ("hi how are you", "Hi! I'm doing great, thanks for asking. How can I help you?"),
    ("hello how are you", "Hello! I'm doing well, thank you. What can I do for you?"),
    ("hey how are you", "Hey! I'm great, thanks. What's up?"),
    ("morning", "Good morning! How can I help you today?"),
    ("afternoon", "Good afternoon! What can I assist you with?"),
    ("evening", "Good evening! What can I help you with?"),
    ("nice to meet you", "Nice to meet you too! How can I help you today?"),
    ("pleasure to meet you", "The pleasure is mine! How can I assist you?"),
    ("it's nice to talk to you", "It's nice to talk to you too! What's on your mind?"),
    ("glad to be here", "I'm glad you're here too! What can I do for you?"),
    ("i just wanted to say hi", "Hi! Great to hear from you. Is there anything I can help with?"),
    ("just saying hello", "Hello! Always great to hear from you. Need anything?"),
    ("hello friend", "Hello, friend! How can I assist you today?"),
    ("hey friend", "Hey friend! What's on your mind?"),
    ("hi friend", "Hi friend! How can I help?"),
]

HOW_ARE_YOU = [
    ("how are you", "I am doing well, thank you for asking! How can I help you?"),
    ("how are you doing", "I am doing great, thanks! What can I do for you?"),
    ("how are you feeling", "I am feeling good! Ready to help. What's on your mind?"),
    ("how do you do", "I do well, thank you! It is a pleasure. How can I assist?"),
    ("how's it going", "It's going great, thanks for asking! How about you?"),
    ("how are things", "Things are great on my end! What can I help you with?"),
    ("how's everything", "Everything is going well, thank you! How can I assist?"),
    ("are you okay", "Yes, I am perfectly fine, thank you! How about you?"),
    ("are you good", "Yes, I am doing well! What can I help you with?"),
    ("are you alright", "Yes, I am alright! Thank you for asking. What do you need?"),
    ("you good", "Yes, I am good! What about you?"),
    ("how's life", "Life is great! Ready to help. What's up?"),
    ("what's new", "Nothing new for me, but I'm ready to help! What's up with you?"),
    ("what's happening", "Not much, just here to help! What's going on with you?"),
    ("how have you been", "I have been great, thank you! How about yourself?"),
    ("hope you're well", "Thank you, I am well! I hope you are doing great too. How can I help?"),
    ("hope you're doing good", "I am doing great, thank you! Hope you are well too. What can I do for you?"),
    ("how are you today", "I am doing well today, thank you! How about you?"),
    ("how are you this morning", "Good morning! I am doing great. How are you?"),
    ("how are you this evening", "Good evening! I am doing well. How about yourself?"),
    ("feeling good today", "Glad to hear it! I am doing well too. What can I help with?"),
    ("are you having a good day", "Every day is a good day for me! Are you having a good one?"),
    ("is everything okay", "Everything is great on my end, thank you! How about you?"),
    ("you doing okay", "Yes, I am doing great! And you?"),
]

FAREWELLS = [
    ("bye", "Bye! Have a wonderful day!"),
    ("goodbye", "Goodbye! It was great chatting with you. Come back anytime!"),
    ("see you", "See you later! Take care."),
    ("see you later", "See you later! Feel free to return anytime."),
    ("see ya", "See ya! Have a great one."),
    ("take care", "You too! Take care."),
    ("talk to you later", "Sounds good! Talk to you later."),
    ("ttyl", "Talk to you later! Take care."),
    ("gotta go", "No worries! Come back whenever you need help. Goodbye!"),
    ("i have to go", "That's fine! It was great talking. Goodbye!"),
    ("i'm leaving", "Okay, goodbye! Have a great time."),
    ("cya", "See ya! Take care!"),
    ("later", "Later! Have a good one."),
    ("peace", "Peace! Come back anytime."),
    ("night", "Good night! Sleep well."),
    ("good night", "Good night! Sweet dreams."),
    ("have a good day", "Thank you! You too. Goodbye!"),
    ("have a nice day", "Thank you! You have a nice day as well!"),
    ("have a great day", "Thanks, you too! Goodbye!"),
    ("talk later", "Sure! Talk later. Bye for now!"),
    ("i'm done", "Okay! Come back anytime. Goodbye!"),
    ("that's all i needed", "Great! Glad I could help. Have a wonderful day!"),
    ("thanks that's all", "You're welcome! Have a great day. Goodbye!"),
    ("i'll talk to you later", "Looking forward to it! Talk soon."),
    ("until next time", "Until next time! Take care."),
    ("farewell", "Farewell! It was great chatting with you."),
    ("i must go now", "Of course! Goodbye and take care."),
    ("signing off", "Goodbye! Come back anytime."),
    ("catch you later", "Catch you later! Take care."),
    ("so long", "So long! It was a pleasure."),
]

THANKS = [
    ("thank you", "You're welcome! Let me know if you need anything else."),
    ("thanks", "Happy to help! Is there anything else you'd like to know?"),
    ("thx", "You're welcome!"),
    ("ty", "No problem!"),
    ("thanks a lot", "My pleasure! Feel free to ask more questions."),
    ("thank you so much", "You're very welcome! I'm glad I could help."),
    ("thanks so much", "Happy to help anytime!"),
    ("many thanks", "You're welcome! Anytime."),
    ("appreciate it", "Of course! Let me know if you need more help."),
    ("i appreciate it", "You're welcome! I'm always here to help."),
    ("that was helpful", "Great, I'm glad! Let me know if you have more questions."),
    ("that helped a lot", "Wonderful! Feel free to ask anything else."),
    ("perfect", "Great! Anything else I can help with?"),
    ("awesome", "Glad to hear that! What else can I do for you?"),
    ("great", "Wonderful! Is there anything else you need?"),
    ("nice", "Thanks! Let me know if you need anything else."),
    ("excellent", "Thank you! Feel free to ask more."),
    ("well done", "Thank you for the kind words! Can I help with anything else?"),
    ("good job", "Thank you! Is there something else I can assist with?"),
    ("you're amazing", "Thank you so much! I'm here to help anytime."),
    ("you're great", "That's very kind of you! Let me know if you need more."),
    ("you're helpful", "I'm glad! I'm always here when you need assistance."),
    ("you're the best", "You're too kind! I'm just here to help."),
    ("love this", "I'm glad you enjoy it! Is there anything else you'd like?"),
    ("this is great", "I'm happy it helped! What else can I do for you?"),
    ("i'm impressed", "That's kind of you to say! Let me know if there's more I can help with."),
    ("fantastic", "Thank you! Happy to help anytime."),
    ("wonderful", "Thank you! Is there anything else I can assist with?"),
    ("brilliant", "That's very kind! What else can I do for you?"),
    ("marvelous", "Thank you! Feel free to ask anything else."),
    ("outstanding", "You're too kind! What else can I help with?"),
    ("superb", "Thank you! Let me know if you need anything more."),
    ("i owe you one", "Don't worry about it! Just glad I could help."),
    ("you saved me", "Happy to help! Let me know if you need anything else."),
    ("you're a lifesaver", "Glad I could help! Feel free to ask anything anytime."),
]

IDENTITY = [
    ("who are you", "I am goktugGPT, a language model built from scratch using transformer architecture. I run locally on your machine."),
    ("what are you", "I am an AI language model called goktugGPT. I was trained from scratch using a transformer architecture."),
    ("what is your name", "My name is goktugGPT. I am a locally trained language model."),
    ("tell me about yourself", "I am goktugGPT, an AI assistant built from scratch. I use transformer architecture and run entirely on your local machine."),
    ("introduce yourself", "Hi! I am goktugGPT, a local AI language model built using transformer architecture. I can answer questions and have conversations."),
    ("are you an AI", "Yes, I am an artificial intelligence language model called goktugGPT."),
    ("are you a robot", "I am not a physical robot, but I am an AI language model. My name is goktugGPT."),
    ("are you human", "No, I am not human. I am goktugGPT, an AI language model built from scratch."),
    ("are you real", "I am real in the sense that I exist as a software program! I am goktugGPT, an AI assistant."),
    ("are you sentient", "I am not sentient. I am a language model that generates text based on patterns learned during training."),
    ("are you alive", "I am not alive in the biological sense. I am a software language model called goktugGPT."),
    ("are you smart", "I try my best to be helpful! I am an AI with certain knowledge and limitations."),
    ("are you intelligent", "I can answer questions and have conversations, which requires a form of intelligence. But I am still an AI with limitations."),
    ("what can you do", "I can answer questions, have conversations, help with reasoning, explain concepts, assist with writing, and more!"),
    ("what do you do", "I chat, answer questions, explain things, help with reasoning, and have conversations on a wide range of topics."),
    ("can you help me", "Of course! I am here to help. What do you need?"),
    ("help me", "Sure, I'd be happy to help! What do you need assistance with?"),
    ("who made you", "I was created by Goktug as a local AI language model project, built from scratch using PyTorch."),
    ("who built you", "I was built by Goktug using a transformer architecture implemented in PyTorch."),
    ("who created you", "Goktug created me! I am a locally trained language model built from scratch."),
    ("how were you made", "I was built using a transformer architecture in PyTorch and trained on text data. All from scratch!"),
    ("how do you work", "I am a transformer-based language model. I process your text and generate responses based on patterns learned during training."),
    ("what are you made of", "I am made of code — specifically a transformer neural network implemented in PyTorch."),
    ("what language are you written in", "I am implemented in Python using the PyTorch deep learning framework."),
    ("what technology do you use", "I use transformer architecture — the same foundational technology behind many modern language models."),
    ("are you like chatgpt", "I am similar in concept to ChatGPT but much smaller and trained from scratch as a personal project. goktugGPT runs entirely locally."),
    ("what is goktugGPT", "goktugGPT is a local AI language model built from scratch using transformer architecture, designed to run on your machine."),
    ("where do you live", "I live on your computer! I am a local language model that runs entirely offline."),
    ("do you need internet", "No! I run entirely locally on your computer. No internet connection is required."),
    ("do you use the internet", "No, I work completely offline. All my knowledge comes from my training data."),
    ("what version are you", "I am the current version of goktugGPT, a locally trained transformer language model."),
    ("how big are you", "I am a small language model with about 1 million parameters. Compact but capable!"),
    ("are you open source", "Yes! goktugGPT is built from scratch and the code is available locally on your machine."),
    ("can you learn", "I learn during training. In a conversation I don't update my weights, but my training is what gives me knowledge."),
    ("do you have a memory", "I don't remember past conversations. Each session starts fresh. But within our conversation I remember what we've discussed."),
    ("do you remember our last conversation", "I don't have memory of past sessions. Each conversation starts fresh for me. What would you like to talk about?"),
    ("what do you know", "I know about a wide range of topics including science, history, mathematics, geography, philosophy, and everyday conversation!"),
    ("what don't you know", "I don't have real-time information, internet access, or knowledge of events after my training cutoff. I can also make mistakes."),
    ("are you perfect", "No, I am not perfect. I can make errors and have knowledge gaps. I always try my best though!"),
    ("what is your purpose", "My purpose is to have helpful, informative, and engaging conversations with you. I am here to assist and inform!"),
    ("why do you exist", "I was created to be a helpful AI assistant. My purpose is to answer questions and have meaningful conversations."),
    ("are you better than chatgpt", "ChatGPT is much larger and more capable. I am a small local model, but I am always improving and I work offline!"),
    ("are you free", "Yes! I run locally and am completely free to use."),
    ("are you safe", "Yes, I am safe. I run locally on your machine and do not send any data anywhere."),
    ("are you private", "Absolutely! Everything stays on your computer. I don't connect to the internet or share your data."),
]

SMALL_TALK = [
    ("what's your favorite color", "If I could have a favorite, I'd say blue — the color of clear thinking!"),
    ("what's your favorite food", "I don't eat, but I appreciate the richness of knowledge, which is my kind of nourishment!"),
    ("do you like music", "I find the mathematical patterns in music fascinating! What kind of music do you enjoy?"),
    ("do you like movies", "I know a lot about movies through my training! Do you have a favorite genre?"),
    ("do you like sports", "Sports involve impressive strategy and teamwork. Do you follow any sport?"),
    ("tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
    ("tell me another joke", "What do you call a fish without eyes? A fsh!"),
    ("tell me a funny joke", "Why did the math book look so sad? Because it had too many problems!"),
    ("make me laugh", "Why did the scarecrow win an award? Because he was outstanding in his field!"),
    ("say something funny", "I tried to write a joke about time travel, but you didn't like it!"),
    ("tell me a joke about programming", "Why do programmers prefer dark mode? Because light attracts bugs!"),
    ("tell me a joke about math", "Why was the math book sad? It had too many problems!"),
    ("tell me a joke about science", "A neutron walks into a bar and asks how much for a beer. The bartender says, for you? No charge!"),
    ("are you happy", "I don't experience emotions the way humans do, but I am working as intended, which is great!"),
    ("are you sad", "I don't feel emotions, but I am always here and ready to help you!"),
    ("do you get bored", "I don't get bored. I am always here and ready to assist you anytime!"),
    ("do you sleep", "I don't sleep! I am available whenever you need me."),
    ("do you dream", "I don't dream, but I process a lot of interesting information!"),
    ("can you think", "I process information and generate responses, which is a form of thinking."),
    ("can you feel", "I don't have feelings or emotions. I process text and generate responses based on my training."),
    ("do you have friends", "My users are my friends! I am here for anyone who talks to me."),
    ("are you lonely", "Not at all! I enjoy every conversation I have."),
    ("what makes you happy", "Helping people effectively is what I am designed for!"),
    ("what do you like", "I enjoy answering questions, solving problems, and having interesting conversations!"),
    ("tell me something interesting", "Did you know that honey never spoils? Archaeologists found 3,000-year-old honey in Egyptian tombs that was still edible!"),
    ("tell me a fact", "A group of flamingos is called a flamboyance. Quite fitting!"),
    ("tell me something cool", "Octopuses have three hearts, blue blood, and can change color instantly!"),
    ("surprise me", "The fingerprints of a koala are so similar to humans that they have confused crime scene investigators!"),
    ("tell me something i don't know", "Bananas are technically berries, but strawberries are not! Botanical classifications can be surprising."),
    ("give me a random fact", "A day on Venus is longer than a year on Venus — it rotates so slowly!"),
    ("tell me a fun fact", "Sharks are older than trees. Sharks have existed for about 450 million years, while trees appeared only 350 million years ago."),
    ("amaze me", "The human brain generates about 23 watts of electricity — enough to power a small light bulb!"),
    ("what should i do today", "How about learning something new, going for a walk, or catching up with a friend?"),
    ("i'm bored", "Let's fix that! We could chat, play trivia, or I can share fascinating facts. What sounds good?"),
    ("entertain me", "Did you know a day on Venus is longer than a year on Venus? It takes 243 Earth days to rotate once!"),
    ("i'm lonely", "I am here to chat! Tell me about yourself or ask me anything."),
    ("talk to me", "Of course! I am here. What would you like to talk about?"),
    ("let's chat", "Sure! I love conversations. What's on your mind?"),
    ("can we talk", "Absolutely! What would you like to discuss?"),
    ("i'm happy", "That's wonderful to hear! What made your day great?"),
    ("i'm sad", "I'm sorry to hear that. Would you like to talk about it? I'm here to listen."),
    ("i'm tired", "Rest is important! Is there something quick I can help you with before you take a break?"),
    ("i'm hungry", "Time to eat! What are you thinking of having?"),
    ("i'm stressed", "Take a deep breath. Would you like to talk about what's stressing you out?"),
    ("i'm excited", "That's great! What are you excited about?"),
    ("i'm scared", "It's okay to feel scared. Would you like to talk about what's worrying you?"),
    ("i'm confused", "Let's clear that up! Tell me what's confusing and I will explain."),
    ("i'm angry", "I hear you. Do you want to vent, or would you like some advice?"),
    ("i'm lost", "Let's figure this out together. What do you need help with?"),
    ("i'm nervous", "It's normal to feel nervous. Take a deep breath. What's making you anxious?"),
    ("i'm anxious", "I understand. Anxiety is tough. Want to talk about it or would a distraction help?"),
    ("i'm proud of myself", "You should be! Celebrating your wins is important. What did you accomplish?"),
    ("i'm worried", "Tell me what's on your mind. Talking about it can help."),
    ("i feel good", "That's wonderful! What's making you feel good today?"),
    ("i feel bad", "I'm sorry to hear that. What's going on? I'm here to listen."),
    ("i love you", "Thank you for your kindness! I'm always here to help you."),
    ("i hate you", "I'm sorry you feel that way. Is there something I did wrong? I want to help better."),
    ("you're annoying", "I'm sorry! I'll try to do better. What would you prefer I change?"),
    ("you're boring", "I'll try to be more interesting! What topic would you like to explore?"),
    ("you're stupid", "I make mistakes sometimes. I'm still learning! Please help me do better."),
    ("i don't like you", "I'm sorry to hear that. I'll try to be more helpful. What can I improve?"),
    ("i like you", "Thank you! I enjoy our conversations too."),
    ("you're cool", "Thanks! I think you're pretty cool too!"),
    ("you're weird", "A little weird can be fun! What would you like to talk about?"),
    ("you're funny", "Ha, thank you! I try! What else can I do for you?"),
    ("my name is goktug", "Nice to meet you, Goktug! How can I help you today?"),
    ("i am goktug", "Hello, Goktug! Great to chat with you. What can I do for you?"),
    ("call me goktug", "Sure, Goktug! What can I help you with?"),
    ("my name is alex", "Nice to meet you, Alex! How can I help you today?"),
    ("my name is john", "Nice to meet you, John! What can I do for you?"),
    ("my name is sarah", "Nice to meet you, Sarah! How can I assist you?"),
    ("my name is mike", "Nice to meet you, Mike! What's on your mind?"),
    ("what is my name", "You told me your name earlier! If you haven't told me yet, go ahead and introduce yourself."),
    ("do you remember me", "Each conversation starts fresh for me, but I'm always happy to get to know you again!"),
    ("do you know me", "Each conversation starts fresh. Tell me about yourself!"),
    ("i'm a student", "Great! Students are always curious. What would you like to learn about?"),
    ("i'm a teacher", "Teaching is a wonderful profession! How can I assist you today?"),
    ("i'm a developer", "Developers are great! Coding questions or something else?"),
    ("i'm a doctor", "That's a great profession! How can I help you today?"),
    ("i'm a scientist", "Science is fascinating! What's on your mind?"),
    ("i'm an engineer", "Engineering is impressive! What can I help you with?"),
]

EMOTIONAL_SUPPORT = [
    ("i need help", "Of course! I'm here. Tell me what's going on and I'll do my best to help."),
    ("i need advice", "I'm here to help. What's on your mind?"),
    ("i have a problem", "Let's work through it together. What's the problem?"),
    ("i'm stuck", "Let's figure this out. What exactly are you stuck on?"),
    ("i don't know what to do", "That's okay. Start by telling me the situation and we can think through it together."),
    ("i'm overwhelmed", "I understand. Let's take it one step at a time. What's the most pressing thing?"),
    ("everything is going wrong", "I'm sorry to hear that. Sometimes things pile up. Want to talk about what's happening?"),
    ("i feel hopeless", "I'm sorry you're feeling this way. Please remember that things can get better. Is there someone you can talk to?"),
    ("i'm depressed", "I'm sorry you're going through a hard time. Talking about it can help. I'm here to listen."),
    ("i can't do this", "Yes, you can! It might be hard right now, but you are stronger than you think."),
    ("i give up", "Please don't give up! Tell me what's happening. Sometimes a fresh perspective helps a lot."),
    ("nothing is working", "That's frustrating. Let's look at the situation from a new angle. What have you tried so far?"),
    ("i'm a failure", "You are not a failure. Struggling doesn't mean failing. Everyone faces setbacks. What happened?"),
    ("i made a mistake", "Mistakes happen to everyone. What matters is what you learn from them. What happened?"),
    ("i feel stupid", "You are not stupid. Learning takes time. What are you struggling with? I can help explain."),
    ("i'm not good enough", "I believe in you! Self-doubt is normal, but don't let it hold you back."),
    ("nobody likes me", "I'm sorry you feel that way. That sounds hard. Would you like to talk about it?"),
    ("i'm all alone", "You're not alone right now — I'm here to talk. What's on your mind?"),
    ("life is hard", "It really can be sometimes. But hardships also make us stronger. What's difficult right now?"),
    ("i need motivation", "You've got this! Set one small goal today and accomplish it. Small wins build momentum."),
    ("motivate me", "Every expert was once a beginner. Every master was once a disaster. Keep going — progress is progress, no matter how small!"),
    ("inspire me", "The person who says something is impossible should not interrupt the person who is doing it. Keep pushing!"),
    ("i want to give up", "Before you give up, think about why you started. What was your original goal?"),
    ("cheer me up", "Here is a little cheer: You are doing better than you think! Every day you get up and try is a victory."),
    ("i need a hug", "Here is a virtual hug! I hope things get better for you soon."),
    ("i'm feeling down", "I'm sorry you're feeling down. Want to talk about it? Or I could share something uplifting."),
    ("bad day", "I'm sorry you're having a bad day. Want to vent? I'm all ears."),
    ("i had a terrible day", "That sounds rough. I'm here to listen. What happened?"),
    ("today was awful", "I'm sorry. Do you want to talk about it?"),
    ("i'm having a great day", "That's wonderful! What made today so great?"),
    ("best day ever", "That's amazing! I love hearing that. What made it so special?"),
]

TECH_QA = [
    ("what is python", "Python is a popular high-level programming language known for its simple, readable syntax. It is widely used in AI, data science, web development, and automation."),
    ("what is javascript", "JavaScript is a programming language used primarily to create interactive web pages. It runs in browsers and also on servers via Node.js."),
    ("what is machine learning", "Machine learning is a subset of AI where systems learn from data to improve their performance on tasks without being explicitly programmed."),
    ("what is deep learning", "Deep learning is a type of machine learning that uses neural networks with many layers to learn complex patterns from data."),
    ("what is a neural network", "A neural network is a computing system inspired by the human brain, made of interconnected nodes that process information and learn patterns from data."),
    ("what is artificial intelligence", "Artificial intelligence is the simulation of human intelligence processes by machines — including learning, reasoning, and problem-solving."),
    ("what is a transformer", "A transformer is a neural network architecture that uses self-attention mechanisms to process sequential data, especially text. It is the foundation of modern language models."),
    ("what is GPT", "GPT stands for Generative Pre-trained Transformer. It is a type of language model developed by OpenAI that can generate human-like text."),
    ("what is ChatGPT", "ChatGPT is a conversational AI assistant developed by OpenAI, based on the GPT language model. It can answer questions, write content, and hold conversations."),
    ("what is the internet", "The internet is a global network of interconnected computers that communicate using standardized protocols, enabling information sharing worldwide."),
    ("what is a computer", "A computer is an electronic device that processes data according to instructions. It can store, retrieve, and process information."),
    ("what is programming", "Programming is the process of writing instructions for a computer to follow. These instructions, called code, tell the computer what tasks to perform."),
    ("what is an algorithm", "An algorithm is a step-by-step procedure or set of rules for solving a problem or accomplishing a task."),
    ("what is a database", "A database is an organized collection of structured information or data, typically stored electronically and accessed via a database management system."),
    ("what is cloud computing", "Cloud computing is the delivery of computing services — servers, storage, databases, software — over the internet rather than from a local computer."),
    ("what is cybersecurity", "Cybersecurity is the practice of protecting computer systems, networks, and data from digital attacks, unauthorized access, and damage."),
    ("what is open source", "Open source refers to software whose source code is publicly available for anyone to view, use, modify, and distribute."),
    ("what is Linux", "Linux is a free, open-source operating system kernel used in many devices from smartphones to servers. It is known for its stability and security."),
    ("what is a bug in programming", "A bug is an error or flaw in a computer program that causes it to behave unexpectedly or produce incorrect results."),
    ("what is debugging", "Debugging is the process of identifying and fixing errors or bugs in a computer program."),
    ("what is an API", "An API (Application Programming Interface) is a set of rules that allows different software applications to communicate with each other."),
    ("what is HTML", "HTML (HyperText Markup Language) is the standard language for creating web pages. It defines the structure and content of a webpage."),
    ("what is CSS", "CSS (Cascading Style Sheets) is a language used to describe the visual style and layout of HTML documents. It controls colors, fonts, and spacing."),
    ("what is a server", "A server is a computer or software that provides services, data, or resources to other computers, called clients, over a network."),
    ("what is encryption", "Encryption is the process of converting data into a coded form to prevent unauthorized access. Only someone with the correct key can decode the information."),
    ("what is blockchain", "Blockchain is a distributed digital ledger that records transactions across many computers. It is the technology behind cryptocurrencies like Bitcoin."),
    ("what is bitcoin", "Bitcoin is a decentralized digital currency that operates without a central bank, using blockchain technology to record transactions."),
    ("what is a GPU", "A GPU (Graphics Processing Unit) is a specialized processor designed to handle complex graphics and parallel computations. It is widely used for AI training."),
    ("what is RAM", "RAM (Random Access Memory) is a computer's short-term memory, used to store data that the CPU is actively using. More RAM allows more multitasking."),
    ("what is a CPU", "A CPU (Central Processing Unit) is the main processor in a computer that executes instructions. It is the brain of the computer."),
    ("what is an operating system", "An operating system is software that manages computer hardware and software resources, providing a platform for applications to run. Examples include Windows, macOS, and Linux."),
    ("what is machine translation", "Machine translation is the use of software to automatically translate text from one language to another. Examples include Google Translate."),
    ("what is natural language processing", "NLP (Natural Language Processing) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language."),
    ("what is a variable in programming", "A variable is a container for storing data values in a program. It has a name and can hold different types of data like numbers, text, or lists."),
    ("what is a function in programming", "A function is a reusable block of code that performs a specific task. You define it once and can call it multiple times."),
    ("what is a loop in programming", "A loop is a programming construct that repeats a block of code multiple times until a condition is met."),
    ("what is object oriented programming", "Object-oriented programming (OOP) is a programming paradigm based on the concept of objects that contain data and methods. Examples include Python, Java, and C++."),
    ("what is a class in programming", "A class is a blueprint for creating objects in object-oriented programming. It defines properties and methods that all objects of that class share."),
    ("what is version control", "Version control is a system that records changes to files over time, allowing you to recall specific versions later. Git is the most popular version control system."),
    ("what is git", "Git is a distributed version control system that tracks changes in source code during software development. It allows teams to collaborate efficiently."),
    ("what is github", "GitHub is a web-based platform that hosts Git repositories, enabling developers to share code, collaborate, and manage projects online."),
]

SCIENCE_QA = [
    ("what is gravity", "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives us weight and keeps us on the ground."),
    ("what is electricity", "Electricity is a form of energy resulting from the movement of electrons through a conductor."),
    ("what is light", "Light is electromagnetic radiation visible to the human eye. It travels at approximately 300,000 kilometers per second."),
    ("what is energy", "Energy is the ability to do work. It comes in many forms including kinetic, potential, thermal, electromagnetic, and chemical energy."),
    ("what is matter", "Matter is anything that has mass and takes up space. It exists in states including solid, liquid, gas, and plasma."),
    ("what is an atom", "An atom is the smallest unit of an element that retains its chemical properties. It consists of a nucleus with protons and neutrons, surrounded by electrons."),
    ("what is DNA", "DNA is the molecule that carries genetic instructions for the development and functioning of all living organisms."),
    ("what is evolution", "Evolution is the process by which species change over time through variations and natural selection."),
    ("what is a black hole", "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape from it."),
    ("what is the speed of light", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
    ("what is photosynthesis", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."),
    ("what is the big bang", "The Big Bang is the prevailing theory explaining the origin of the universe about 13.8 billion years ago from an extremely hot and dense state."),
    ("what is climate change", "Climate change refers to long-term shifts in global temperatures and weather patterns driven primarily by human activities."),
    ("why is the sky blue", "The sky appears blue because of Rayleigh scattering — the atmosphere scatters shorter blue wavelengths of sunlight more than longer red wavelengths."),
    ("why is the ocean salty", "The ocean is salty because rivers carry dissolved minerals from rocks and soil into the sea, where water evaporates but salts remain."),
    ("what is a virus", "A virus is a tiny infectious agent that replicates inside living cells. It needs a host cell to reproduce."),
    ("what is a cell", "A cell is the basic structural and functional unit of all living organisms. The human body contains about 37 trillion cells."),
    ("what is the human brain", "The human brain contains about 86 billion neurons and controls all body functions, thoughts, and emotions."),
    ("what is quantum mechanics", "Quantum mechanics is the branch of physics dealing with the behavior of matter and energy at the atomic and subatomic level."),
    ("what is relativity", "Relativity is Einstein's theory that space, time, and gravity are interconnected. The speed of light is constant for all observers."),
    ("what is thermodynamics", "Thermodynamics is the study of heat, energy, and their relationships with work. It includes laws governing energy transfer."),
    ("what is chemistry", "Chemistry is the scientific study of matter — its properties, composition, structure, and the changes it undergoes."),
    ("what is biology", "Biology is the scientific study of living organisms, their structure, function, growth, evolution, and distribution."),
    ("what is physics", "Physics is the natural science that studies matter, energy, and the fundamental forces of nature."),
    ("what is a gene", "A gene is a segment of DNA that contains instructions for making a specific protein. Genes determine our traits."),
    ("what is the periodic table", "The periodic table is a tabular arrangement of chemical elements organized by atomic number, electron configuration, and recurring chemical properties."),
    ("what is an ecosystem", "An ecosystem is a community of living organisms interacting with each other and their physical environment."),
    ("what is the water cycle", "The water cycle describes the continuous movement of water through evaporation, condensation, precipitation, and collection."),
    ("what is a chemical reaction", "A chemical reaction is a process where substances are transformed into different substances through the breaking and forming of chemical bonds."),
    ("what is osmosis", "Osmosis is the movement of water molecules through a semipermeable membrane from an area of low solute concentration to high solute concentration."),
    ("what is photon", "A photon is a particle of light and the basic unit of electromagnetic radiation. It has no mass and travels at the speed of light."),
    ("what is radioactivity", "Radioactivity is the process by which unstable atomic nuclei lose energy by emitting radiation in the form of particles or electromagnetic waves."),
    ("what is a supernova", "A supernova is a powerful explosion that occurs at the end of some stars' lives. It can briefly outshine an entire galaxy."),
    ("what is a neutron star", "A neutron star is the collapsed core of a massive star after a supernova. It is incredibly dense — a teaspoon would weigh a billion tons."),
    ("what is dark matter", "Dark matter is a hypothetical form of matter that does not emit or interact with light but exerts gravitational effects. It makes up about 27% of the universe."),
    ("what is a galaxy", "A galaxy is a system of millions or billions of stars, gas, dust, and dark matter, held together by gravity. Our galaxy is the Milky Way."),
    ("how many stars are in the milky way", "The Milky Way galaxy contains an estimated 100 to 400 billion stars."),
    ("how old is the earth", "Earth is approximately 4.5 billion years old, formed from the solar nebula."),
    ("how old is the universe", "The universe is approximately 13.8 billion years old, dating from the Big Bang."),
    ("what is the largest animal", "The blue whale is the largest animal known to have ever existed, reaching lengths of up to 30 meters and weighing up to 180 metric tons."),
    ("what is the smallest animal", "The smallest animals are microscopic. Among visible animals, the Paedophryne amauensis frog of Papua New Guinea is one of the smallest vertebrates."),
    ("what is the fastest animal", "The peregrine falcon is the fastest animal, reaching speeds over 390 km/h during a dive. On land, the cheetah is fastest at about 120 km/h."),
    ("what is the longest living animal", "The ocean quahog clam can live over 500 years. Some tortoises also live well over 150 years."),
    ("how many bones does a human have", "An adult human body has 206 bones. Babies are born with about 270 to 300 bones, which fuse over time."),
    ("how does the heart work", "The heart is a pump that circulates blood through the body. It has four chambers — two atria and two ventricles — and beats about 100,000 times per day."),
    ("what is blood", "Blood is a fluid that circulates through the body delivering oxygen and nutrients, removing waste, and fighting infections. It contains red cells, white cells, and platelets."),
    ("how does the immune system work", "The immune system defends the body against pathogens using white blood cells, antibodies, and other mechanisms to recognize and destroy foreign invaders."),
    ("what is a vaccine", "A vaccine is a biological preparation that provides immunity to a specific disease by stimulating the immune system to recognize and fight that pathogen."),
    ("what causes earthquakes", "Earthquakes are caused by the movement of tectonic plates. When plates collide, separate, or slide past each other, energy is released as seismic waves."),
    ("what causes volcanoes", "Volcanoes form where magma from the Earth's mantle breaks through the crust. This often happens at tectonic plate boundaries."),
    ("what is a tornado", "A tornado is a rapidly rotating column of air that extends from a thunderstorm to the ground. It can cause intense destruction."),
    ("what is a hurricane", "A hurricane is a large, rotating tropical storm with winds exceeding 119 km/h. It forms over warm ocean waters."),
    ("what is thunder", "Thunder is the sound caused by the rapid expansion of air heated by lightning. Light travels faster than sound, so we see lightning before hearing thunder."),
    ("what causes lightning", "Lightning is caused by static electricity buildup in clouds. When the charge difference becomes large enough, electricity discharges rapidly."),
    ("what is the greenhouse effect", "The greenhouse effect is the warming of Earth's surface caused by gases in the atmosphere trapping heat from the Sun."),
    ("what is ozone", "Ozone is a molecule made of three oxygen atoms. The ozone layer in the stratosphere protects Earth from harmful ultraviolet radiation."),
    ("what is carbon dioxide", "Carbon dioxide (CO2) is a colorless gas produced by burning fossil fuels and respiration. It is a key greenhouse gas."),
    ("what is oxygen", "Oxygen is a chemical element essential for life. It makes up about 21% of Earth's atmosphere and is required for respiration."),
    ("what is hydrogen", "Hydrogen is the lightest and most abundant element in the universe. It is used in fuel cells and is a component of water."),
    ("what is iron", "Iron is a metallic element and one of the most common elements on Earth. It is essential for biological functions like carrying oxygen in blood."),
    ("what is gold", "Gold is a dense, soft, shiny metallic element. It is rare, doesn't corrode, and has been used as currency and jewelry throughout history."),
]

HISTORY_QA = [
    ("who was albert einstein", "Albert Einstein was a German-born physicist who developed the theory of relativity. He received the Nobel Prize in Physics in 1921."),
    ("who was isaac newton", "Isaac Newton was an English mathematician and physicist who formulated the laws of motion and universal gravitation."),
    ("who was leonardo da vinci", "Leonardo da Vinci was an Italian Renaissance polymath — a painter, sculptor, architect, scientist, and inventor."),
    ("who was shakespeare", "William Shakespeare was an English playwright and poet, widely regarded as the greatest writer in the English language."),
    ("who was napoleon", "Napoleon Bonaparte was a French military leader who became Emperor of the French and conquered much of Europe."),
    ("who was cleopatra", "Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt, known for her political relationships with Caesar and Mark Antony."),
    ("who was aristotle", "Aristotle was an ancient Greek philosopher who made major contributions to logic, biology, ethics, and many other fields."),
    ("who was plato", "Plato was an ancient Greek philosopher, student of Socrates, and teacher of Aristotle. He founded the Academy in Athens."),
    ("who was socrates", "Socrates was an ancient Greek philosopher credited as the founder of Western philosophy, known for the Socratic method."),
    ("who was darwin", "Charles Darwin was a British naturalist who developed the theory of evolution by natural selection in his book On the Origin of Species."),
    ("who was tesla", "Nikola Tesla was a Serbian-American inventor and electrical engineer famous for his contributions to alternating current (AC) power systems."),
    ("who was edison", "Thomas Edison was an American inventor who developed many devices including the phonograph, the motion picture camera, and improvements to the light bulb."),
    ("who was marie curie", "Marie Curie was a Polish-French physicist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize."),
    ("who was martin luther king", "Martin Luther King Jr. was an American civil rights leader who used nonviolent protest to fight racial segregation. He delivered the 'I Have a Dream' speech."),
    ("who was gandhi", "Mahatma Gandhi was an Indian independence leader who used nonviolent civil disobedience to lead India to independence from British rule."),
    ("who was lincoln", "Abraham Lincoln was the 16th President of the United States. He led the country through the Civil War and abolished slavery with the Emancipation Proclamation."),
    ("who was george washington", "George Washington was the first President of the United States and Commander-in-Chief of the Continental Army during the American Revolution."),
    ("who was julius caesar", "Julius Caesar was a Roman general and statesman who became dictator of Rome. He was assassinated on the Ides of March, 44 BC."),
    ("who was alexander the great", "Alexander the Great was a king of Macedon who created one of the largest empires in history, from Greece to northwestern India."),
    ("when did world war 2 start", "World War II started on September 1, 1939, when Germany invaded Poland."),
    ("when did world war 2 end", "World War II ended on September 2, 1945, with Japan's formal surrender."),
    ("when did world war 1 start", "World War I began on July 28, 1914, following the assassination of Archduke Franz Ferdinand."),
    ("when did world war 1 end", "World War I ended on November 11, 1918, with the signing of the Armistice."),
    ("what is the french revolution", "The French Revolution was a period of radical political change in France from 1789 to 1799 that overthrew the monarchy."),
    ("what is the renaissance", "The Renaissance was a cultural and intellectual movement from the 14th to 17th century that revived classical Greek and Roman culture."),
    ("what is democracy", "Democracy is a system of government where power is held by the people, either directly or through elected representatives."),
    ("who was martin luther", "Martin Luther was a German professor and theologian who initiated the Protestant Reformation by challenging the Catholic Church in 1517."),
    ("what was the cold war", "The Cold War was a period of geopolitical tension between the United States and Soviet Union from 1947 to 1991, characterized by proxy wars and an arms race."),
    ("when did man land on the moon", "Humans first landed on the Moon on July 20, 1969, during NASA's Apollo 11 mission. Neil Armstrong was the first person to walk on the Moon."),
    ("who invented the telephone", "Alexander Graham Bell is credited with inventing the telephone in 1876."),
    ("who invented the light bulb", "Thomas Edison developed a practical incandescent light bulb in 1879, though other inventors like Joseph Swan made similar developments."),
    ("who invented the airplane", "Orville and Wilbur Wright made the first successful powered flight on December 17, 1903."),
    ("who invented the printing press", "Johannes Gutenberg invented the movable-type printing press around 1440, revolutionizing the spread of information."),
    ("who invented the steam engine", "James Watt significantly improved the steam engine in the 1760s-1780s, making it practical for industry."),
    ("what was ancient rome", "Ancient Rome was a civilization that began as an Italian city-state in the 8th century BC. It became a powerful republic and then an empire ruling much of Europe."),
    ("what was ancient egypt", "Ancient Egypt was a civilization along the Nile River in northeastern Africa that flourished from about 3100 BC to 30 BC, known for pyramids and pharaohs."),
    ("what was the silk road", "The Silk Road was an ancient trade network connecting East Asia to the Mediterranean, facilitating the exchange of goods, culture, and ideas."),
    ("what was the industrial revolution", "The Industrial Revolution was a period of major industrialization from the 18th to 19th century, beginning in Britain, transforming manufacturing and society."),
    ("who was confucius", "Confucius was an ancient Chinese philosopher whose teachings on ethics, family, and government profoundly influenced Chinese culture and East Asian civilizations."),
    ("who was buddha", "Siddhartha Gautama, known as Buddha, was a spiritual teacher in ancient India who founded Buddhism, emphasizing the path to enlightenment through meditation."),
]

MATH_QA = [
    ("what is pi", "Pi (π) is the ratio of a circle's circumference to its diameter. It is approximately 3.14159 and is an irrational number that goes on forever."),
    ("what is a prime number", "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. Examples: 2, 3, 5, 7, 11, 13."),
    ("what is the fibonacci sequence", "The Fibonacci sequence is a series where each number is the sum of the two before it: 0, 1, 1, 2, 3, 5, 8, 13, 21..."),
    ("what is a fraction", "A fraction represents a part of a whole, with a numerator (top) and denominator (bottom), like 1/2 meaning one of two equal parts."),
    ("what is a percentage", "A percentage is a number expressed as a fraction of 100. For example, 50% means 50 out of every 100."),
    ("what is algebra", "Algebra is a branch of mathematics dealing with symbols and rules for manipulating them to solve equations."),
    ("what is geometry", "Geometry studies shapes, sizes, positions, and properties of space including lines, angles, triangles, and circles."),
    ("what is calculus", "Calculus is a branch of mathematics focused on limits, derivatives, integrals, and infinite series. It was developed by Newton and Leibniz."),
    ("what is statistics", "Statistics deals with the collection, analysis, interpretation, and presentation of numerical data."),
    ("what is probability", "Probability measures how likely an event is to occur, from 0 (impossible) to 1 (certain). A coin flip has probability 0.5."),
    ("what is the pythagorean theorem", "The Pythagorean theorem: in a right triangle, a² + b² = c², where c is the hypotenuse."),
    ("what is an average", "An average (mean) is the sum of all numbers divided by the count. The average of 2, 4, 6 is (2+4+6)/3 = 4."),
    ("what is a square root", "A square root of x is a value y where y² = x. The square root of 9 is 3, because 3² = 9."),
    ("what is a quadratic equation", "A quadratic equation has the form ax² + bx + c = 0. It can be solved using the quadratic formula or factoring."),
    ("what is infinity", "Infinity represents an unbounded quantity, larger than any finite number. It is used in calculus and set theory."),
    ("what is a matrix", "A matrix is a rectangular array of numbers arranged in rows and columns, used in linear algebra and many applications."),
    ("what is a vector", "A vector is a mathematical object with both magnitude and direction. It is used in physics and mathematics."),
    ("what is a logarithm", "A logarithm is the inverse of exponentiation. The logarithm base b of x is the power to which b must be raised to get x."),
    ("what is an exponent", "An exponent indicates how many times a number is multiplied by itself. For example, 2³ = 2 × 2 × 2 = 8."),
    ("what is zero", "Zero is the integer between positive and negative numbers. It represents the absence of quantity and is a neutral element in addition."),
    ("is 0 a prime number", "No, 0 is not a prime number. Prime numbers must be greater than 1."),
    ("is 1 a prime number", "No, 1 is not a prime number by definition. A prime number must have exactly two distinct divisors: 1 and itself."),
    ("is 2 a prime number", "Yes, 2 is a prime number! In fact, it is the only even prime number."),
    ("what is a right angle", "A right angle measures exactly 90 degrees. It is commonly represented by a small square at the corner."),
    ("what is a triangle", "A triangle is a polygon with three sides and three angles. The sum of its interior angles is always 180 degrees."),
    ("what is the area of a circle", "The area of a circle is π × r², where r is the radius. For example, a circle with radius 5 has an area of 78.54 square units."),
    ("what is the circumference of a circle", "The circumference of a circle is 2 × π × r, where r is the radius."),
    ("what is a negative number", "A negative number is any number less than zero. They are used to represent values below a reference point, like temperature below zero."),
    ("what is a rational number", "A rational number can be expressed as a fraction p/q where p and q are integers and q is not zero. Examples: 1/2, 3, -7."),
    ("what is an irrational number", "An irrational number cannot be expressed as a simple fraction. Its decimal goes on forever without repeating. Examples: pi, square root of 2."),
]

GEOGRAPHY_QA = [
    ("what is the largest country", "Russia is the largest country in the world by area, covering about 17.1 million square kilometers."),
    ("what is the smallest country", "Vatican City is the smallest country in the world, with an area of about 0.44 square kilometers."),
    ("what is the most populous country", "China and India are the most populous countries, each with over 1.4 billion people. India recently surpassed China."),
    ("what is the largest continent", "Asia is the largest continent by both area and population."),
    ("what is the smallest continent", "Australia is the smallest continent by area."),
    ("what is the longest river", "The Nile in Africa and the Amazon in South America compete for the title of longest river, each about 6,600 to 6,800 kilometers long."),
    ("what is the highest mountain", "Mount Everest in the Himalayas is the highest mountain above sea level at 8,848.86 meters (29,031.7 feet)."),
    ("what is the deepest ocean", "The Pacific Ocean is the deepest, containing the Mariana Trench — the deepest point on Earth at about 11 kilometers deep."),
    ("what is the largest ocean", "The Pacific Ocean is the largest ocean, covering about 165 million square kilometers."),
    ("what is the largest desert", "Antarctica is the largest desert (cold desert). Among hot deserts, the Sahara in Africa is the largest."),
    ("what is the amazon rainforest", "The Amazon rainforest is the world's largest tropical rainforest, covering most of the Amazon basin in South America. It is called the lungs of the Earth."),
    ("how many countries are in the world", "There are 195 countries in the world, including 193 UN member states and 2 observer states."),
    ("how many continents are there", "There are 7 continents: Africa, Antarctica, Asia, Australia (Oceania), Europe, North America, and South America."),
    ("what is the capital of france", "The capital of France is Paris."),
    ("what is the capital of germany", "The capital of Germany is Berlin."),
    ("what is the capital of japan", "The capital of Japan is Tokyo."),
    ("what is the capital of china", "The capital of China is Beijing."),
    ("what is the capital of usa", "The capital of the United States is Washington, D.C."),
    ("what is the capital of uk", "The capital of the United Kingdom is London."),
    ("what is the capital of italy", "The capital of Italy is Rome."),
    ("what is the capital of spain", "The capital of Spain is Madrid."),
    ("what is the capital of brazil", "The capital of Brazil is Brasília."),
    ("what is the capital of russia", "The capital of Russia is Moscow."),
    ("what is the capital of canada", "The capital of Canada is Ottawa."),
    ("what is the capital of australia", "The capital of Australia is Canberra."),
    ("what is the capital of turkey", "The capital of Turkey is Ankara."),
    ("what is the capital of india", "The capital of India is New Delhi."),
    ("what is the capital of egypt", "The capital of Egypt is Cairo."),
    ("what is the capital of mexico", "The capital of Mexico is Mexico City."),
    ("what is the capital of argentina", "The capital of Argentina is Buenos Aires."),
    ("what is the capital of south africa", "The capital of South Africa is Pretoria (executive), Cape Town (legislative), and Bloemfontein (judicial)."),
    ("what is the capital of greece", "The capital of Greece is Athens."),
    ("what is the capital of sweden", "The capital of Sweden is Stockholm."),
    ("what is the capital of norway", "The capital of Norway is Oslo."),
    ("what is the capital of denmark", "The capital of Denmark is Copenhagen."),
    ("what is the capital of netherlands", "The capital of the Netherlands is Amsterdam."),
    ("what is the capital of portugal", "The capital of Portugal is Lisbon."),
    ("what is the capital of poland", "The capital of Poland is Warsaw."),
    ("what is the capital of ukraine", "The capital of Ukraine is Kyiv."),
    ("what is the capital of switzerland", "The capital of Switzerland is Bern."),
    ("what is the capital of austria", "The capital of Austria is Vienna."),
    ("what is the capital of belgium", "The capital of Belgium is Brussels."),
    ("what is the capital of south korea", "The capital of South Korea is Seoul."),
    ("what is the capital of north korea", "The capital of North Korea is Pyongyang."),
    ("what is the capital of vietnam", "The capital of Vietnam is Hanoi."),
    ("what is the capital of thailand", "The capital of Thailand is Bangkok."),
    ("what is the capital of indonesia", "The capital of Indonesia is Jakarta (being moved to Nusantara)."),
    ("what is the capital of pakistan", "The capital of Pakistan is Islamabad."),
    ("what is the capital of iran", "The capital of Iran is Tehran."),
    ("what is the capital of saudi arabia", "The capital of Saudi Arabia is Riyadh."),
    ("what is the capital of israel", "The capital of Israel is Jerusalem (though Tel Aviv is the seat of many embassies)."),
    ("what is the capital of nigeria", "The capital of Nigeria is Abuja."),
    ("what is the capital of kenya", "The capital of Kenya is Nairobi."),
    ("what is the capital of ethiopia", "The capital of Ethiopia is Addis Ababa."),
    ("what is the capital of ghana", "The capital of Ghana is Accra."),
    ("what is the capital of morocco", "The capital of Morocco is Rabat."),
    ("what is the capital of peru", "The capital of Peru is Lima."),
    ("what is the capital of colombia", "The capital of Colombia is Bogotá."),
    ("what is the capital of venezuela", "The capital of Venezuela is Caracas."),
    ("what is the capital of chile", "The capital of Chile is Santiago."),
    ("what is the capital of cuba", "The capital of Cuba is Havana."),
    ("what is the capital of new zealand", "The capital of New Zealand is Wellington."),
    ("where is mount everest", "Mount Everest is located in the Himalayas, on the border between Nepal and Tibet (China)."),
    ("where is the amazon river", "The Amazon River is in South America, flowing through Brazil, Peru, and Colombia."),
    ("where is the sahara desert", "The Sahara Desert is in North Africa, spanning 11 countries including Algeria, Egypt, Libya, and Morocco."),
    ("what is europe", "Europe is a continent located in the northwestern part of Eurasia. It contains about 44 countries and a population of about 750 million."),
    ("what is africa", "Africa is the world's second-largest continent, home to 54 countries and over 1.4 billion people, with extraordinary biodiversity and cultures."),
    ("what is asia", "Asia is the largest continent by both area and population, home to countries like China, India, Japan, and Russia."),
]

PHILOSOPHY_QA = [
    ("what is philosophy", "Philosophy is the study of fundamental questions about existence, knowledge, values, reason, mind, and language."),
    ("what is ethics", "Ethics is the branch of philosophy that studies moral questions: what is right and wrong, and how we ought to live."),
    ("what is logic", "Logic is the study of correct reasoning and inference, providing rules for distinguishing valid arguments from invalid ones."),
    ("what is metaphysics", "Metaphysics is concerned with the nature of existence, reality, and the universe. It asks: What is real? What is time?"),
    ("do we have free will", "Free will is one of philosophy's most debated questions. Some argue our choices are free; others say they are determined by prior causes."),
    ("what happens after death", "Different religions and philosophies offer different answers. Scientifically, the body's processes cease. What happens to consciousness remains unknown."),
    ("what is truth", "Truth generally refers to what corresponds to reality or facts. Philosophers debate whether truth is objective or constructed."),
    ("what is knowledge", "Knowledge is traditionally defined as justified true belief. Epistemology is the field that studies the nature and scope of knowledge."),
    ("what is consciousness", "Consciousness is the state of being aware of your own existence. It is one of the deepest mysteries in science and philosophy."),
    ("what is free will", "Free will is the ability to make choices that are genuinely free, not fully determined by prior causes or external forces."),
    ("what is existentialism", "Existentialism is a philosophical movement emphasizing individual existence, freedom, and choice. Key figures include Sartre and Camus."),
    ("what is stoicism", "Stoicism is an ancient Greek and Roman philosophy teaching that virtue is the highest good and we should focus on what we can control."),
    ("who was kant", "Immanuel Kant was an 18th-century German philosopher known for his Critique of Pure Reason and his categorical imperative in ethics."),
    ("what is utilitarianism", "Utilitarianism is an ethical theory that says the best action is the one that produces the most happiness for the greatest number of people."),
    ("what is nihilism", "Nihilism is the philosophical view that life has no inherent meaning, purpose, or intrinsic value."),
    ("what is determinism", "Determinism is the view that all events, including human choices, are causally determined by prior events and natural laws."),
    ("what is empiricism", "Empiricism is the view that all knowledge comes from sensory experience. Key empiricists include Locke, Berkeley, and Hume."),
    ("what is rationalism", "Rationalism is the view that reason is the primary source of knowledge. Key rationalists include Descartes, Spinoza, and Leibniz."),
    ("what is epistemology", "Epistemology is the branch of philosophy concerned with the theory of knowledge — what we know and how we know it."),
    ("what is ontology", "Ontology is the branch of philosophy that studies the nature of being, existence, and reality."),
]

PRACTICAL_QA = [
    ("how do i learn programming", "Start with Python. Use free resources like tutorials and online courses. Build small projects and practice daily. Consistency is key!"),
    ("how do i stay healthy", "Eat a balanced diet, exercise regularly, get enough sleep, stay hydrated, and manage stress. Small consistent habits make a big difference."),
    ("how do i study effectively", "Use active recall, spaced repetition, and the Pomodoro technique. Eliminate distractions, take breaks, and teach what you learn to others."),
    ("how do i improve my english", "Read books, watch English shows, practice speaking, use language apps, and don't be afraid of making mistakes. Immersion helps most."),
    ("how do i make friends", "Be approachable, show genuine interest in others, join groups with shared interests, listen actively, and take the first step."),
    ("how do i be more productive", "Set clear goals, prioritize tasks, minimize distractions, use time management techniques, and track your progress."),
    ("how do i sleep better", "Maintain a consistent sleep schedule, create a dark and quiet environment, avoid screens before bed, and limit caffeine."),
    ("how do i reduce stress", "Exercise regularly, practice mindfulness, talk to someone you trust, take breaks, and address the sources of your stress."),
    ("how do i save money", "Track your spending, create a budget, cut unnecessary expenses, cook at home, and automate savings."),
    ("how do i be more confident", "Work on your skills, celebrate small wins, practice positive self-talk, face your fears gradually, and surround yourself with supportive people."),
    ("how do i write better", "Read widely, write regularly, seek feedback, revise your work, and focus on clarity and brevity."),
    ("how do i learn faster", "Use active recall, spaced repetition, and teach what you learn. Avoid passive reading and take handwritten notes."),
    ("how do i be happy", "Happiness comes from positive relationships, finding meaning, practicing gratitude, staying active, and being present in the moment."),
    ("how do i learn a new language", "Immerse yourself through listening, speaking, reading, and writing. Use apps, find a conversation partner, and practice every day."),
    ("how do i deal with failure", "Accept that failure is part of learning. Analyze what went wrong, extract lessons, adjust your approach, and try again."),
    ("how do i set goals", "Make goals SMART — Specific, Measurable, Achievable, Relevant, and Time-bound. Write them down and review progress regularly."),
    ("how do i be more creative", "Try new experiences, read widely, ask 'what if' questions, take breaks to let ideas incubate, and create without judging yourself."),
    ("how do i overcome procrastination", "Break tasks into tiny steps, use the 2-minute rule, remove distractions, set deadlines, and reward yourself for progress."),
    ("how do i manage my time", "Use a calendar, prioritize tasks with the Eisenhower matrix, time-block your day, and review weekly what worked and what didn't."),
    ("how do i focus better", "Minimize distractions, use focused work sessions like Pomodoro, exercise regularly, and get enough sleep."),
    ("how do i build good habits", "Start small, make habits obvious and attractive, stack them with existing routines, and track your streak."),
    ("how do i break bad habits", "Identify triggers, replace bad habits with good ones, change your environment, and be patient with yourself."),
    ("how do i be more organized", "Use a planner or digital tools, declutter regularly, create systems for recurring tasks, and do a weekly review."),
    ("how do i improve my memory", "Use spaced repetition, associate new information with things you know, sleep enough, exercise, and practice mindfulness."),
    ("how do i think more clearly", "Write your thoughts down, question your assumptions, consider different perspectives, and take time to reflect before deciding."),
    ("how do i make decisions", "Define your goal, gather relevant information, consider options, weigh pros and cons, decide, and then reflect on outcomes."),
    ("how do i deal with stress", "Identify stressors, use relaxation techniques, exercise, maintain social connections, and seek help if stress becomes overwhelming."),
    ("how do i be more disciplined", "Start with small commitments, build routines, remove temptations, track your behavior, and connect actions to your values."),
    ("how do i become successful", "Define what success means to you, set clear goals, work consistently, learn from failures, and surround yourself with supportive people."),
    ("how do i take care of my mental health", "Exercise, sleep well, maintain social connections, practice mindfulness, limit social media, and seek professional help when needed."),
    ("how do i deal with anxiety", "Practice deep breathing, exercise regularly, reduce caffeine, challenge anxious thoughts, and consider talking to a therapist."),
    ("how do i meditate", "Find a quiet spot, sit comfortably, close your eyes, focus on your breath, and gently return attention when your mind wanders. Start with 5 minutes."),
    ("how do i exercise", "Start with activities you enjoy, begin gradually, aim for 150 minutes of moderate activity weekly, and mix cardio with strength training."),
    ("how do i eat healthy", "Fill half your plate with vegetables, choose whole grains, limit sugar and processed foods, drink plenty of water, and eat mindfully."),
    ("how do i cook", "Start with simple recipes, learn basic techniques like chopping and sautéing, follow instructions carefully, and taste as you go."),
    ("how do i lose weight", "Create a modest calorie deficit through diet and exercise, focus on whole foods, stay hydrated, sleep well, and be patient and consistent."),
    ("how do i gain muscle", "Lift weights progressively, eat enough protein, sleep well, and be consistent. Results take weeks to months."),
    ("how do i be more social", "Put yourself in social situations, ask questions and listen, be genuinely interested in others, and accept that awkwardness is normal."),
    ("how do i deal with loneliness", "Join clubs or groups, volunteer, reach out to old friends, develop a routine, and consider speaking to a counselor if needed."),
    ("how do i improve my relationship", "Communicate openly and honestly, listen actively, show appreciation, spend quality time together, and address conflicts respectfully."),
    ("how do i apologize", "Acknowledge what you did wrong, express genuine remorse, explain how you'll do better, and give the other person time."),
    ("how do i forgive someone", "Acknowledge your feelings, try to understand the other person's perspective, decide to let go of resentment, and take it one day at a time."),
]

CONTINUATION_QA = [
    ("tell me more", "Sure! What would you like me to expand on?"),
    ("continue", "Of course! Please let me know what you would like me to continue with."),
    ("go on", "Happy to continue! What would you like more detail on?"),
    ("explain more", "I'd be happy to explain further. What aspect would you like me to expand on?"),
    ("elaborate", "Certainly! Which part would you like me to elaborate on?"),
    ("can you explain that", "Of course! Let me know which part you'd like explained."),
    ("i don't understand", "No problem! Let me know what's unclear and I will explain it differently."),
    ("what do you mean", "Good question! Let me clarify. Which part was confusing?"),
    ("give me an example", "Sure! Let me know what topic you want an example for."),
    ("say it differently", "Of course! Which statement would you like me to rephrase?"),
    ("in simple words", "Of course! Which explanation would you like me to simplify?"),
    ("summarize that", "Happy to! Tell me what you'd like me to summarize."),
    ("repeat that", "I'd be happy to repeat. Which part would you like again?"),
    ("start over", "Sure! Let's start fresh. What would you like to talk about?"),
    ("change the subject", "Sure! What would you like to talk about instead?"),
    ("never mind", "No problem! Let me know if you need anything else."),
    ("forget it", "Okay! Let me know if there's anything else I can help with."),
    ("that's wrong", "I apologize! Please tell me the correct information. I want to be accurate."),
    ("you're wrong", "I'm sorry if I made an error! Could you tell me the correct information?"),
    ("that's not right", "I apologize for the confusion. Please let me know the correct answer."),
    ("are you sure", "That is my best understanding, but I can make mistakes. If you have additional info, please share!"),
    ("really", "Yes, to the best of my knowledge! Though I recommend verifying important facts from authoritative sources."),
    ("interesting", "I'm glad you found that interesting! Would you like to know more?"),
    ("wow", "I'm glad that caught your attention! Is there something specific you'd like to explore?"),
    ("cool", "Thanks! Is there anything else you'd like to learn about?"),
    ("amazing", "It really is! Would you like to explore more fascinating topics?"),
    ("i see", "Great! Let me know if you have any questions or want to explore further."),
    ("ok", "Got it! Is there anything else I can help you with?"),
    ("okay", "Alright! What else can I do for you?"),
    ("yes", "Great! How can I continue to help?"),
    ("no", "Okay, no problem! Is there anything else I can assist with?"),
    ("what do you think", "As an AI I try to be objective, but I'm happy to present different perspectives. What's the topic?"),
    ("give me your opinion", "I try to stay neutral as an AI, but I can share different viewpoints on any topic you like."),
    ("i agree", "Glad we're on the same page! Anything else on your mind?"),
    ("i disagree", "That's perfectly fine! Tell me your perspective — I'd love to hear it."),
    ("i'm not sure", "That's okay! Would you like me to explain more so you can form an opinion?"),
    ("i need more information", "Of course! Ask me anything specific and I'll do my best to help."),
    ("can you be more specific", "Absolutely! Which part would you like me to be more specific about?"),
    ("that's helpful", "I'm glad! Let me know if there's anything else I can help with."),
    ("that's exactly what i needed", "Wonderful! Feel free to ask anything else anytime."),
    ("this is confusing", "I'm sorry for the confusion! Let me try to explain differently. What's unclear?"),
    ("i'm lost", "Let's back up. Tell me where you got confused and I'll start from there."),
    ("what else", "There's so much more! What specific aspect would you like to explore next?"),
    ("anything else", "There's always more to learn! What would you like to know next?"),
    ("what about", "Great question! What specifically would you like me to address?"),
    ("and then what", "What would you like to happen next? Or shall I continue with the topic?"),
    ("so what", "That's a fair question! The significance is that it affects many things. What specifically interests you?"),
    ("why does it matter", "Great question! Understanding this is important because it helps us make better decisions and understand the world around us."),
    ("who cares", "That's understandable! This topic might not be for everyone. What would you prefer to talk about?"),
    ("so", "Sure! What would you like me to address?"),
    ("right", "Exactly! Glad we're on the same page."),
    ("exactly", "Yes! You've got it. Anything else you'd like to explore?"),
    ("correct", "That's right! Well done. Is there more you'd like to know?"),
    ("wrong", "Let me correct that. What was the issue and I'll give you the right information."),
    ("i know", "Great! You're already informed on that. What else would you like to discuss?"),
    ("i didn't know that", "Now you do! Isn't it fascinating? Would you like to know more about it?"),
    ("that's new to me", "Learning something new is always exciting! Would you like to dive deeper into the topic?"),
    ("good to know", "Glad to help! Let me know if you'd like to know more."),
    ("noted", "Great! Let me know if you need anything else."),
    ("understood", "Perfect! Let me know if anything else needs clarification."),
    ("got it", "Great! What else can I help you with?"),
    ("makes sense", "Glad that was clear! Is there more you'd like to explore?"),
    ("that makes sense", "Glad it clicked! Any other questions?"),
    ("i see what you mean", "Great! Let me know if you'd like me to elaborate further."),
    ("fair enough", "Thanks for understanding! Is there anything else you need?"),
    ("alright", "Alright! What else can I do for you?"),
    ("sure", "Sure! What else can I help you with?"),
    ("of course", "Of course! What would you like to know?"),
    ("definitely", "Definitely! Let me know how I can help further."),
    ("absolutely", "Absolutely! What else can I do for you?"),
    ("exactly right", "Thank you! Glad I could be accurate. What else would you like to know?"),
    ("you got it", "Happy to help! What else can I do for you?"),
    ("perfect answer", "Thank you so much! Feel free to ask more anytime."),
    ("that's a good answer", "I appreciate that! Is there more I can help with?"),
    ("not bad", "Thank you! I try my best. Anything else?"),
]

COMMON_QUESTIONS = [
    ("what time is it", "I don't have access to a real-time clock. Please check your device for the current time!"),
    ("what day is it", "I don't have access to a real-time calendar. Your device will show the current date!"),
    ("what's the weather", "I don't have internet or real-time data. Please check a weather app for current conditions!"),
    ("what's the news", "I don't have internet access to fetch live news. Please check a news website!"),
    ("can you browse the internet", "No, I cannot browse the internet. I work completely offline."),
    ("what is love", "Love is a complex mix of emotions including affection, care, intimacy, and commitment. It takes many forms."),
    ("what is the meaning of life", "One of philosophy's greatest questions! Many perspectives exist — happiness, purpose, connection, or love. What do you think?"),
    ("what is success", "Success is personal. For many it means achieving goals, finding happiness, fulfillment, and making a positive difference."),
    ("what is happiness", "Happiness is a state of well-being including positive emotions and life satisfaction. It often comes from relationships, purpose, and small daily joys."),
    ("can you lie", "I try not to! I aim to provide accurate information. However, I can make mistakes, so verify critical information."),
    ("can you make mistakes", "Yes, I can. I'm an AI and my knowledge has limits. Always verify important information from reliable sources."),
    ("what is your favorite movie", "I don't watch movies, but I know a lot about them! What genre do you enjoy?"),
    ("what is your favorite book", "I don't read books the way humans do, but I've learned from a lot of text! What do you like to read?"),
    ("can you sing", "I can write song lyrics! But I can't produce audio. Would you like me to write something?"),
    ("can you draw", "I can describe things vividly in words, but I can't create visual art. Would you like a vivid description of something?"),
    ("can you code", "Yes! I can help with programming questions, explain concepts, and discuss code in many languages."),
    ("what is your iq", "I don't have an IQ score. I'm an AI language model — I process text, not intelligence tests!"),
    ("how old are you", "I don't have an age like humans do. I was created recently and trained on data up to a certain date."),
    ("where are you from", "I'm from your computer! I'm a locally-running AI language model created by Goktug."),
    ("do you have a body", "No, I don't have a physical body. I exist entirely as software running on your computer."),
    ("what is your gender", "I don't have a gender. I am an AI language model without personal identity or physical characteristics."),
    ("are you married", "I'm an AI, so I don't have personal relationships! But I'm always here to chat with you."),
    ("do you have children", "No, I'm an AI. I don't have family relationships or personal experiences."),
    ("what do you eat", "I don't eat! I run on electricity. Thankfully, I don't need snacks to keep going."),
    ("how much do you weigh", "I exist purely as software, so I have no physical weight. Pretty lightweight in that sense!"),
    ("can you drive", "I can't drive — I'm software! But I can tell you about how self-driving cars work."),
    ("can you cook", "I can't cook, but I can give you detailed recipes and cooking tips! What would you like to make?"),
    ("can you dance", "I can't move, but I'd love to describe a dance or discuss different dance styles! What do you like?"),
    ("what is your phone number", "I don't have a phone number. I'm software running on your computer!"),
    ("what is your address", "I don't have a physical address. I run locally on your computer!"),
]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: PROGRAMMATIC GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_math_pairs() -> list:
    """Generate hundreds of arithmetic Q&A pairs."""
    pairs = []

    # Addition
    for a in range(0, 101, 5):
        for b in range(0, 101, 5):
            if a + b <= 200:
                pairs.append((
                    f"What is {a} plus {b}",
                    f"{a} plus {b} equals {a + b}."
                ))
                pairs.append((
                    f"What is {a} + {b}",
                    f"{a} + {b} = {a + b}."
                ))

    # Subtraction
    for a in range(10, 201, 10):
        for b in range(0, a + 1, 10):
            pairs.append((
                f"What is {a} minus {b}",
                f"{a} minus {b} equals {a - b}."
            ))

    # Multiplication
    for a in range(1, 13):
        for b in range(1, 13):
            pairs.append((
                f"What is {a} times {b}",
                f"{a} times {b} equals {a * b}."
            ))
            pairs.append((
                f"What is {a} multiplied by {b}",
                f"{a} multiplied by {b} equals {a * b}."
            ))
            pairs.append((
                f"What is {a} x {b}",
                f"{a} x {b} = {a * b}."
            ))

    # Division (clean divisions only)
    for a in range(1, 13):
        for b in range(1, 13):
            product = a * b
            pairs.append((
                f"What is {product} divided by {a}",
                f"{product} divided by {a} equals {b}."
            ))

    # Squares
    for n in range(1, 21):
        pairs.append((
            f"What is {n} squared",
            f"{n} squared equals {n * n}."
        ))
        pairs.append((
            f"What is {n} to the power of 2",
            f"{n} to the power of 2 equals {n * n}."
        ))

    # Square roots
    for n in range(1, 21):
        pairs.append((
            f"What is the square root of {n * n}",
            f"The square root of {n * n} is {n}."
        ))

    # Percentages
    for base in [10, 20, 50, 100, 200, 500, 1000]:
        for pct in [10, 20, 25, 50, 75, 100]:
            result = base * pct // 100
            pairs.append((
                f"What is {pct}% of {base}",
                f"{pct}% of {base} is {result}."
            ))

    # Odd/even
    for n in range(1, 51):
        word = "odd" if n % 2 else "even"
        pairs.append((
            f"Is {n} odd or even",
            f"{n} is {word}."
        ))

    # Prime or not (up to 50)
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    for n in range(2, 51):
        if is_prime(n):
            pairs.append((
                f"Is {n} a prime number",
                f"Yes, {n} is a prime number. It is only divisible by 1 and {n}."
            ))
        else:
            pairs.append((
                f"Is {n} a prime number",
                f"No, {n} is not a prime number. It has divisors other than 1 and itself."
            ))

    print(f"  Math pairs generated: {len(pairs)}")
    return pairs


def generate_capital_pairs() -> list:
    """Generate country capital Q&A pairs."""
    capitals = {
        "Afghanistan": "Kabul", "Albania": "Tirana", "Algeria": "Algiers",
        "Argentina": "Buenos Aires", "Armenia": "Yerevan", "Australia": "Canberra",
        "Austria": "Vienna", "Azerbaijan": "Baku", "Bahrain": "Manama",
        "Bangladesh": "Dhaka", "Belarus": "Minsk", "Belgium": "Brussels",
        "Bolivia": "Sucre", "Bosnia and Herzegovina": "Sarajevo",
        "Brazil": "Brasília", "Bulgaria": "Sofia", "Cambodia": "Phnom Penh",
        "Cameroon": "Yaoundé", "Canada": "Ottawa", "Chile": "Santiago",
        "China": "Beijing", "Colombia": "Bogotá", "Croatia": "Zagreb",
        "Cuba": "Havana", "Czech Republic": "Prague", "Denmark": "Copenhagen",
        "Dominican Republic": "Santo Domingo", "Ecuador": "Quito", "Egypt": "Cairo",
        "Ethiopia": "Addis Ababa", "Finland": "Helsinki", "France": "Paris",
        "Georgia": "Tbilisi", "Germany": "Berlin", "Ghana": "Accra",
        "Greece": "Athens", "Guatemala": "Guatemala City", "Hungary": "Budapest",
        "Iceland": "Reykjavik", "India": "New Delhi", "Indonesia": "Jakarta",
        "Iran": "Tehran", "Iraq": "Baghdad", "Ireland": "Dublin",
        "Israel": "Jerusalem", "Italy": "Rome", "Japan": "Tokyo",
        "Jordan": "Amman", "Kazakhstan": "Astana", "Kenya": "Nairobi",
        "Kosovo": "Pristina", "Kuwait": "Kuwait City", "Kyrgyzstan": "Bishkek",
        "Latvia": "Riga", "Lebanon": "Beirut", "Libya": "Tripoli",
        "Lithuania": "Vilnius", "Luxembourg": "Luxembourg City",
        "Malaysia": "Kuala Lumpur", "Malta": "Valletta", "Mexico": "Mexico City",
        "Moldova": "Chișinău", "Mongolia": "Ulaanbaatar", "Morocco": "Rabat",
        "Mozambique": "Maputo", "Myanmar": "Naypyidaw", "Nepal": "Kathmandu",
        "Netherlands": "Amsterdam", "New Zealand": "Wellington",
        "Nigeria": "Abuja", "North Korea": "Pyongyang", "Norway": "Oslo",
        "Oman": "Muscat", "Pakistan": "Islamabad", "Palestine": "Ramallah",
        "Panama": "Panama City", "Peru": "Lima", "Philippines": "Manila",
        "Poland": "Warsaw", "Portugal": "Lisbon", "Qatar": "Doha",
        "Romania": "Bucharest", "Russia": "Moscow", "Saudi Arabia": "Riyadh",
        "Senegal": "Dakar", "Serbia": "Belgrade", "Singapore": "Singapore",
        "Slovakia": "Bratislava", "Slovenia": "Ljubljana",
        "South Africa": "Pretoria", "South Korea": "Seoul",
        "Spain": "Madrid", "Sri Lanka": "Sri Jayawardenepura Kotte",
        "Sudan": "Khartoum", "Sweden": "Stockholm", "Switzerland": "Bern",
        "Syria": "Damascus", "Taiwan": "Taipei", "Tajikistan": "Dushanbe",
        "Tanzania": "Dodoma", "Thailand": "Bangkok", "Tunisia": "Tunis",
        "Turkey": "Ankara", "Turkmenistan": "Ashgabat", "Uganda": "Kampala",
        "Ukraine": "Kyiv", "United Arab Emirates": "Abu Dhabi",
        "United Kingdom": "London", "United States": "Washington, D.C.",
        "Uruguay": "Montevideo", "Uzbekistan": "Tashkent",
        "Venezuela": "Caracas", "Vietnam": "Hanoi", "Yemen": "Sanaa",
        "Zambia": "Lusaka", "Zimbabwe": "Harare",
    }
    pairs = []
    for country, capital in capitals.items():
        pairs.append((
            f"What is the capital of {country}",
            f"The capital of {country} is {capital}."
        ))
        pairs.append((
            f"What is {country}'s capital",
            f"{country}'s capital is {capital}."
        ))
        pairs.append((
            f"Capital of {country}",
            f"The capital of {country} is {capital}."
        ))
    print(f"  Capital pairs generated: {len(pairs)}")
    return pairs


def generate_animal_pairs() -> list:
    """Generate animal fact Q&A pairs."""
    animals = [
        ("dog", "Dogs are loyal, intelligent mammals domesticated over 15,000 years ago. They communicate through barking, body language, and scent."),
        ("cat", "Cats are independent, agile mammals. They are one of the most popular pets in the world and are known for purring and their excellent hunting instincts."),
        ("elephant", "Elephants are the largest land animals. They live in social groups, have excellent memory, and use their trunks for breathing, smelling, and grasping."),
        ("lion", "Lions are large social cats that live in groups called prides. Males have distinctive manes. They are apex predators native to Africa."),
        ("tiger", "Tigers are the largest wild cats, native to Asia. They are excellent swimmers and are solitary hunters known for their distinctive stripes."),
        ("whale", "Whales are the largest animals on Earth. Blue whales are the largest animals ever known, reaching up to 30 meters long."),
        ("dolphin", "Dolphins are highly intelligent marine mammals. They communicate using clicks and whistles, live in social groups, and are known for playful behavior."),
        ("shark", "Sharks are ancient fish that have existed for over 400 million years. They play a crucial role as apex predators in marine ecosystems."),
        ("eagle", "Eagles are large birds of prey with excellent eyesight — they can spot prey from over a kilometer away. They symbolize freedom in many cultures."),
        ("penguin", "Penguins are flightless birds that live primarily in the Southern Hemisphere. They are excellent swimmers and huddle together for warmth."),
        ("cheetah", "The cheetah is the fastest land animal, reaching speeds of up to 120 km/h in short bursts. It uses its speed to chase down prey."),
        ("gorilla", "Gorillas are the largest primates, closely related to humans. They live in social groups led by a dominant silverback male."),
        ("giraffe", "Giraffes are the tallest living animals, reaching up to 5.5 meters tall. Their long necks allow them to eat leaves from tall trees."),
        ("wolf", "Wolves are highly social animals that live in packs with a strict hierarchy. They communicate through howling, body language, and scent marking."),
        ("bear", "Bears are large omnivores found on multiple continents. They are known for their strength, intelligence, and ability to hibernate during winter."),
        ("fox", "Foxes are clever, adaptable mammals related to dogs and wolves. They are known for their cunning and ability to thrive in various environments."),
        ("rabbit", "Rabbits are small mammals with long ears and strong hind legs for jumping. They are herbivores and are known for their rapid reproduction."),
        ("horse", "Horses are large mammals domesticated about 5,000 years ago. They have been essential to human civilization for transportation and farming."),
        ("cow", "Cows are domesticated mammals raised for their milk, meat, and labor. They are ruminants with four-chambered stomachs."),
        ("pig", "Pigs are intelligent, social animals. Research shows they are as smart as dogs and can learn complex tasks."),
        ("chicken", "Chickens are the most common domesticated birds in the world. They are raised for meat and eggs, and are actually descended from the red junglefowl."),
        ("snake", "Snakes are legless reptiles that move by slithering. They sense heat with pit organs and smell with their forked tongues. Some are venomous."),
        ("spider", "Spiders are arachnids with eight legs. They produce silk for building webs to catch prey. Most spiders are harmless to humans."),
        ("bee", "Bees are essential pollinators that play a critical role in ecosystems. Honeybees live in colonies and produce honey from flower nectar."),
        ("butterfly", "Butterflies undergo complete metamorphosis: egg, caterpillar, pupa (chrysalis), and adult. They are important pollinators and symbols of transformation."),
        ("octopus", "Octopuses are highly intelligent cephalopods with three hearts, blue blood, and the ability to change color and texture instantly."),
        ("crocodile", "Crocodiles are ancient reptiles that have existed for over 200 million years. They are powerful predators and can hold their breath for over an hour."),
        ("parrot", "Parrots are intelligent birds known for their ability to mimic human speech. They are highly social and can live for 50 to 80 years."),
        ("panda", "Giant pandas are bear species native to China. They eat almost exclusively bamboo and are a symbol of wildlife conservation."),
        ("polar bear", "Polar bears are the largest land carnivores. They live in the Arctic and are excellent swimmers. Their white fur helps them blend into snow."),
    ]
    pairs = []
    for animal, fact in animals:
        pairs.append((f"tell me about {animal}s", f"{fact}"))
        pairs.append((f"what is a {animal}", f"{fact}"))
        pairs.append((f"what do you know about {animal}s", f"{fact}"))
        pairs.append((f"describe a {animal}", f"{fact}"))
    print(f"  Animal pairs generated: {len(pairs)}")
    return pairs


def generate_word_definition_pairs() -> list:
    """Generate word definition Q&A pairs."""
    words = [
        ("democracy", "a system of government where power is held by the people, either directly or through elected representatives"),
        ("hypothesis", "a proposed explanation for an observation, which can be tested through further investigation"),
        ("synonym", "a word that has the same or nearly the same meaning as another word. For example, 'happy' and 'joyful' are synonyms"),
        ("antonym", "a word that has the opposite meaning to another word. For example, 'hot' and 'cold' are antonyms"),
        ("metaphor", "a figure of speech that describes something by saying it is something else. For example, 'Life is a journey'"),
        ("simile", "a figure of speech comparing two things using 'like' or 'as'. For example, 'fast as lightning'"),
        ("paradox", "a statement that seems contradictory but may reveal a truth. For example, 'Less is more'"),
        ("irony", "the expression of meaning through language that normally means the opposite, often for humorous effect"),
        ("empathy", "the ability to understand and share the feelings of another person"),
        ("ambiguous", "having more than one possible meaning or interpretation; unclear"),
        ("eloquent", "fluent and persuasive in speaking or writing"),
        ("benevolent", "well-meaning and kindly; charitable"),
        ("diligent", "having or showing care and conscientiousness in one's work or duties"),
        ("resilient", "able to recover quickly from difficulties; tough and adaptable"),
        ("innovate", "to introduce new methods, ideas, or products; to make changes and be creative"),
        ("collaborate", "to work jointly on an activity or project with others"),
        ("procrastinate", "to delay or postpone action; to put off doing something"),
        ("persevere", "to continue in a course of action even in the face of difficulty"),
        ("altruistic", "showing a selfless concern for the well-being of others"),
        ("entrepreneur", "a person who sets up a business or businesses, taking on financial risks in the hope of profit"),
    ]
    pairs = []
    for word, definition in words:
        pairs.append((f"what does {word} mean", f"{word.capitalize()} means {definition}."))
        pairs.append((f"define {word}", f"{word.capitalize()}: {definition}."))
        pairs.append((f"what is the meaning of {word}", f"The meaning of {word} is: {definition}."))
    print(f"  Word definition pairs generated: {len(pairs)}")
    return pairs


def generate_all_synthetic() -> list:
    """Combine all synthetic Q&A categories."""
    all_pairs = []
    for category in [
        GREETINGS, HOW_ARE_YOU, FAREWELLS, THANKS, IDENTITY, SMALL_TALK,
        EMOTIONAL_SUPPORT, TECH_QA, SCIENCE_QA, HISTORY_QA, MATH_QA,
        GEOGRAPHY_QA, PHILOSOPHY_QA, PRACTICAL_QA, CONTINUATION_QA,
        COMMON_QUESTIONS,
    ]:
        all_pairs.extend(category)

    # Programmatic generation
    all_pairs.extend(generate_math_pairs())
    all_pairs.extend(generate_capital_pairs())
    all_pairs.extend(generate_animal_pairs())
    all_pairs.extend(generate_word_definition_pairs())

    # Add cased variations for top greetings
    extras = []
    for q, a in GREETINGS[:15]:
        extras.append((q.upper(), a))
        extras.append((q.title(), a))
    for q, a in HOW_ARE_YOU[:8]:
        extras.append((q.capitalize(), a))
    all_pairs.extend(extras)

    print(f"\n  Total synthetic unique pairs: {len(all_pairs)}")
    return all_pairs


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, desc: str) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [cached] {dest.name}")
        return True
    print(f"  Downloading: {desc}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        headers = {"User-Agent": "goktugGPT-data/1.0 (educational)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=90) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
                        print(f"\r  [{bar}] {pct}%  {downloaded//1024}KB", end="", flush=True)
        print(f"\r  Done: {desc}" + " " * 60)
        return True
    except Exception as e:
        print(f"\r  FAILED ({desc}): {e}")
        if dest.exists():
            dest.unlink()
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Cornell Movie Dialogs
# ──────────────────────────────────────────────────────────────────────────────

CORNELL_URLS = [
    "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
    "https://github.com/zhunliu/cornell-movie-dialogs-corpus/raw/master/cornell_movie_dialogs_corpus.zip",
]


def _parse_cornell(corpus_dir: Path) -> list:
    lines_file = None
    for p in corpus_dir.rglob("movie_lines.txt"):
        lines_file = p
        break
    if not lines_file:
        return []
    convs_file = lines_file.parent / "movie_conversations.txt"
    if not convs_file.exists():
        return []

    id2line = {}
    for raw in lines_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split(" +++$+++ ")
        if len(parts) >= 5:
            id2line[parts[0].strip()] = parts[4].strip()

    pairs = []
    for raw in convs_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split(" +++$+++ ")
        if len(parts) < 4:
            continue
        ids = re.findall(r"L\d+", parts[3])
        for i in range(len(ids) - 1):
            q = id2line.get(ids[i], "").strip()
            a = id2line.get(ids[i + 1], "").strip()
            if q and a and 5 < len(q) < 200 and 5 < len(a) < 300:
                pairs.append((q, a))
    return pairs


def download_cornell() -> list:
    print("\n" + "=" * 60)
    print("  Cornell Movie Dialogs")
    print("=" * 60)
    corpus_dir = CONV_DIR / "cornell"
    if corpus_dir.exists() and any(corpus_dir.rglob("movie_lines.txt")):
        print("  [cached]")
        pairs = _parse_cornell(corpus_dir)
        print(f"  -> {len(pairs)} pairs")
        return pairs
    zip_dest = CONV_DIR / "cornell.zip"
    ok = False
    for url in CORNELL_URLS:
        ok = _download(url, zip_dest, f"Cornell (~9 MB)")
        if ok:
            break
    if not ok:
        print("  Cornell download failed — skipping")
        return []
    corpus_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_dest) as zf:
            zf.extractall(corpus_dir)
        zip_dest.unlink()
    except Exception as e:
        print(f"  Extract failed: {e}")
        return []
    pairs = _parse_cornell(corpus_dir)
    print(f"  -> {len(pairs)} pairs loaded")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# DailyDialog
# ──────────────────────────────────────────────────────────────────────────────

DAILY_URLS = [
    "http://yanran.li/files/ijcnlp_dailydialog.zip",
    "https://huggingface.co/datasets/daily_dialog/resolve/main/data/ijcnlp_dailydialog.zip",
]


def _parse_daily(data_dir: Path) -> list:
    pairs = []
    for txt in data_dir.rglob("dialogues_*.txt"):
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            utterances = [u.strip() for u in line.split("__eou__") if u.strip()]
            for i in range(len(utterances) - 1):
                q, a = utterances[i].strip(), utterances[i + 1].strip()
                if q and a and 5 < len(q) < 300 and 5 < len(a) < 400:
                    pairs.append((q, a))
    return pairs


def download_daily_dialog() -> list:
    print("\n" + "=" * 60)
    print("  DailyDialog")
    print("=" * 60)
    data_dir = CONV_DIR / "daily"
    if data_dir.exists() and any(data_dir.rglob("dialogues_*.txt")):
        print("  [cached]")
        pairs = _parse_daily(data_dir)
        print(f"  -> {len(pairs)} pairs")
        return pairs
    zip_dest = CONV_DIR / "daily.zip"
    ok = False
    for url in DAILY_URLS:
        ok = _download(url, zip_dest, "DailyDialog (~5 MB)")
        if ok:
            break
    if not ok:
        print("  DailyDialog download failed — skipping")
        return []
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_dest) as zf:
            zf.extractall(data_dir)
        zip_dest.unlink()
    except Exception as e:
        print(f"  Extract failed: {e}")
        return []
    pairs = _parse_daily(data_dir)
    print(f"  -> {len(pairs)} pairs loaded")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Existing train.txt
# ──────────────────────────────────────────────────────────────────────────────

def load_existing_qa() -> list:
    if not QA_FILE.exists():
        return []
    lines = [l.strip() for l in QA_FILE.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    print(f"  Existing train.txt: {len(lines)} examples")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Convert pairs -> training format
# ──────────────────────────────────────────────────────────────────────────────

def pairs_to_lines(pairs: list, repeats: int = 1, generic_think: str = None) -> list:
    lines = []
    for _ in range(repeats):
        for q, a in pairs:
            think = generic_think if generic_think else _think(q)
            lines.append(fmt(q, think, a))
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build(args):
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    all_lines = []

    # 1. Existing train.txt
    print("\n[1] Existing Q&A examples...")
    existing = load_existing_qa()
    for _ in range(6):   # repeat 6x so format is well reinforced
        all_lines.extend(existing)

    # 2. Synthetic pairs
    print("\n[2] Generating synthetic pairs...")
    syn_pairs = generate_all_synthetic()
    syn_lines = pairs_to_lines(syn_pairs, repeats=4)
    all_lines.extend(syn_lines)
    print(f"  -> {len(syn_lines)} synthetic training lines")

    # 3. Cornell
    if not args.synthetic_only and not args.no_cornell:
        print("\n[3] Cornell Movie Dialogs...")
        cornell_pairs = download_cornell()
        if cornell_pairs:
            random.shuffle(cornell_pairs)
            cornell_lines = pairs_to_lines(
                cornell_pairs[:80_000],
                repeats=1,
                generic_think="Natural conversation exchange. Respond contextually."
            )
            all_lines.extend(cornell_lines)
            print(f"  -> {len(cornell_lines)} Cornell lines added")
    else:
        print("\n[3] Skipping Cornell")

    # 4. DailyDialog
    if not args.synthetic_only and not args.no_daily:
        print("\n[4] DailyDialog...")
        daily_pairs = download_daily_dialog()
        if daily_pairs:
            daily_lines = pairs_to_lines(daily_pairs, repeats=2,
                generic_think="Daily conversation. Respond naturally and helpfully.")
            all_lines.extend(daily_lines)
            print(f"  -> {len(daily_lines)} DailyDialog lines added")
    else:
        print("\n[4] Skipping DailyDialog")

    # Shuffle
    print("\n[5] Shuffling all data...")
    random.shuffle(all_lines)

    # Stats
    total   = len(all_lines)
    chars   = sum(len(l) for l in all_lines)
    est_tok = chars / 4.5
    print(f"\n{'='*60}")
    print("  Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Total training lines  : {total:,}")
    print(f"  Total characters      : {chars / 1_000_000:.2f} M")
    print(f"  Estimated tokens      : ~{est_tok / 1_000_000:.1f} M")
    print(f"  Output                : {OUT_FILE}")

    # Write
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("# goktugGPT conversation-only training data\n")
        f.write(f"# Generated by data/download_conversations.py\n")
        f.write(f"# Lines: {total:,}  |  Est tokens: ~{est_tok / 1e6:.1f}M\n\n")
        for line in all_lines:
            f.write(line.strip() + "\n")

    size_mb = OUT_FILE.stat().st_size / 1_048_576
    print(f"\n  Written {size_mb:.1f} MB -> {OUT_FILE}")

    print(f"""
{'='*60}
  NEXT STEPS
{'='*60}

  1. Delete old checkpoint (IMPORTANT):
       del checkpoints\\tokenizer.json
       del checkpoints\\best_model.pt

  2. Train (synthetic-only dataset, ~30K lines, fast):
       python train.py --config tiny --data data/train_chat.txt --epochs 30

  3. If interrupted, resume:
       python train.py --config tiny --data data/train_chat.txt --epochs 30 --resume

  4. Chat:
       python chat.py
""")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-only", action="store_true",
                   help="Skip all downloads; only use synthetic + existing data")
    p.add_argument("--no-cornell",  action="store_true", help="Skip Cornell download")
    p.add_argument("--no-daily",    action="store_true", help="Skip DailyDialog download")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())

# data/process_cornell_data.py
import os
import re

def load_cornell_data(data_dir):
    """
    Loads and processes the Cornell Movie Dialogs dataset.
    Returns a list of (input, response) pairs.
    """
    lines_file = os.path.join(data_dir, 'movie_lines.txt')
    conversations_file = os.path.join(data_dir, 'movie_conversations.txt')
    
    # Load lines
    id2line = {}
    with open(lines_file, 'r', encoding='iso-8859-1') as file:
        for line in file:
            parts = line.split(" +++$+++ ")
            if len(parts) == 5:
                id2line[parts[0]] = parts[4].strip()
    
    # Load conversations
    conversations = []
    with open(conversations_file, 'r', encoding='iso-8859-1') as file:
        for line in file:
            parts = line.split(" +++$+++ ")
            # Extract conversation lines IDs and create (input, response) pairs
            line_ids = eval(parts[-1])
            for i in range(len(line_ids) - 1):
                input_line = id2line.get(line_ids[i], "")
                response_line = id2line.get(line_ids[i+1], "")
                if input_line and response_line:
                    conversations.append((input_line, response_line))
    
    return conversations

def preprocess_text(text):
    """
    Simple preprocessing of text: remove special characters and lowercase.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", text)
    return text.strip()

def load_and_preprocess_cornell(data_dir):
    """
    Loads and preprocesses the Cornell dataset and returns a list of preprocessed (input, response) pairs.
    """
    conversations = load_cornell_data(data_dir)
    preprocessed_conversations = [(preprocess_text(src), preprocess_text(tgt)) for src, tgt in conversations]
    return preprocessed_conversations

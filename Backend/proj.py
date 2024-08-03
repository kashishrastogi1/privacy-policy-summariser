# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import requests
# from bs4 import BeautifulSoup
# import re
# from langdetect import detect, LangDetectException
# from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the summarization model and tokenizer
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# def fetch_tnc_and_links(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Check if the request was successful
#         soup = BeautifulSoup(response.text, 'html.parser')

#         # Extract text
#         tnc_text = soup.get_text(separator='\n', strip=True)

#         # Extract hyperlinks and their text
#         links = []
#         for link in soup.find_all('a', href=True):
#             link_text = link.get_text(strip=True)
#             links.append((link_text, link['href']))

#         return tnc_text.strip(), links
#     except requests.exceptions.RequestException as e:
#         return f"Error fetching T&C: {e}", []

# def fetch_link_content(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, 'html.parser')
#         return soup.get_text(separator='\n', strip=True)
#     except requests.exceptions.RequestException as e:
#         return f"Error fetching content from link: {e}"

# def clean_text(text):
#     # Remove unwanted Unicode characters
#     text = text.encode("ascii", "ignore").decode()
#     # Remove specific text patterns using regex
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
#     text = re.sub(r'http\S+', '', text)  # Remove URLs
#     text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
#     text = re.sub(r'\(.*?\)', '', text)  # Remove text in parentheses
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove all non-alphanumeric characters except spaces
#     return text.strip()

# def is_english(text):
#     try:
#         return detect(text) == 'en'
#     except LangDetectException:
#         return False

# def combine_content(tnc_text, link_contents):
#     combined_content = tnc_text
#     for link_text, content in link_contents.items():
#         if not content.startswith("Error fetching content from link:") and is_english(content):  # Exclude invalid and non-English content
#             combined_content += f"\n\n### {link_text}\n{content}"
#     combined_content = combined_content.replace('\n', ' ')
#     return clean_text(combined_content)

# @app.route('/summarize-url', methods=['POST'])
# def summarize_url():
#     data = request.json
#     url = data['url']
    
#     # Fetch and combine content from URL and its links
#     tnc, links = fetch_tnc_and_links(url)
#     link_contents = {link[0]: fetch_link_content(link[1]) for link in links}
#     combined_content = combine_content(tnc, link_contents)

#     # Tokenize and split into chunks
#     inputs = tokenizer(combined_content, return_tensors="pt", max_length=5000, truncation=True)
#     input_ids = inputs["input_ids"][0]

#     chunk_size = 1024  # Define the chunk size

#     # Function to split input_ids into chunks
#     def chunk_input_ids(input_ids, chunk_size):
#         return [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]

#     input_chunks = chunk_input_ids(input_ids, chunk_size)

#     # Summarize each chunk
#     summaries = []
#     for chunk in input_chunks:
#         summary_ids = model.generate(chunk.unsqueeze(0), max_length=500, min_length=90, do_sample=False)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         summaries.append(summary)

#     # Combine the summaries into a single text
#     combined_summary = ' '.join(summaries)

#     return jsonify({"summary": combined_summary})

# if __name__ == '__main__':
#     app.run(debug=True)

    
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import requests
# from bs4 import BeautifulSoup
# import re
# from langdetect import detect, LangDetectException
# import os
# import google.generativeai as genai

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes


# genai.configure(api_key='AIzaSyB722ymFP1U4wZGHR98O-8BE8moHtVOyvI')


# def fetch_tnc_and_links(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Check if the request was successful
#         soup = BeautifulSoup(response.text, 'html.parser')

#         # Extract text
#         tnc_text = soup.get_text(separator='\n', strip=True)

#         # Extract hyperlinks and their text
#         links = []
#         for link in soup.find_all('a', href=True):
#             link_text = link.get_text(strip=True)
#             links.append((link_text, link['href']))

#         return tnc_text.strip(), links
#     except requests.exceptions.RequestException as e:
#         return f"Error fetching T&C: {e}", []

# def fetch_link_content(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, 'html.parser')
#         return soup.get_text(separator='\n', strip=True)
#     except requests.exceptions.RequestException as e:
#         return f"Error fetching content from link: {e}"

# def clean_text(text):
#     # Remove unwanted Unicode characters
#     text = text.encode("ascii", "ignore").decode()
#     # Remove specific text patterns using regex
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
#     text = re.sub(r'http\S+', '', text)  # Remove URLs
#     text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
#     text = re.sub(r'\(.*?\)', '', text)  # Remove text in parentheses
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove all non-alphanumeric characters except spaces
#     return text.strip()

# def is_english(text):
#     try:
#         return detect(text) == 'en'
#     except LangDetectException:
#         return False

# def combine_content(tnc_text, link_contents):
#     combined_content = tnc_text
#     for link_text, content in link_contents.items():
#         if not content.startswith("Error fetching content from link:") and is_english(content):  # Exclude invalid and non-English content
#             combined_content += f"\n\n### {link_text}\n{content}"
#     combined_content = combined_content.replace('\n', ' ')
#     return clean_text(combined_content)

# def summarize_with_gemini(text):
#   model = genai.GenerativeModel('gemini-1.5-flash',generation_config=genai.GenerationConfig(temperature=0.5))
#   response = model.generate_content(f"""Summarize the privacy policy provided in triple quotes. 
#   Summarize it in simple and clear language for a general audience.
#   The summary should focus on the key points that a common man would need to understand, such as:

#   What personal information is collected.
#   How the information is used.
#   How the information is shared.
#   User rights regarding their data.

#   Ensure the summary is concise, easy to read, and avoids technical jargon.

#   privacy policy: '''{text}'''""")

#   return response.text


# # i haven't figured out this part yet i.e decorator and everything
# @app.route('/summarize-url', methods=['POST'])
# def summarize_url():
#     data = request.json
#     url = data['url']
    
#     # Fetch and combine content from URL and its links
#     tnc, links = fetch_tnc_and_links(url)
#     link_contents = {link[0]: fetch_link_content(link[1]) for link in links}
#     combined_content = combine_content(tnc, link_contents)

#     # Summarize the combined content using Gemini
#     combined_summary = summarize_with_gemini(combined_content)
    
#     print("hello")
#     return jsonify({"summary": combined_summary})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re
from langdetect import detect, LangDetectException
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

genai.configure(api_key='')


def fetch_tnc_and_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text
        tnc_text = soup.get_text(separator='\n', strip=True)

        # Extract hyperlinks and their text
        links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            links.append((link_text, link['href']))

        return tnc_text.strip(), links
    except requests.exceptions.RequestException as e:
        return f"Error fetching T&C: {e}", []

def fetch_link_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    except requests.exceptions.RequestException as e:
        return f"Error fetching content from link: {e}"

def clean_text(text):
    # Remove unwanted Unicode characters
    text = text.encode("ascii", "ignore").decode()
    # Remove specific text patterns using regex
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r'\(.*?\)', '', text)  # Remove text in parentheses
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove all non-alphanumeric characters except spaces
    return text.strip()

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def combine_content(tnc_text, link_contents):
    combined_content = tnc_text
    for link_text, content in link_contents.items():
        if not content.startswith("Error fetching content from link:") and is_english(content):  # Exclude invalid and non-English content
            combined_content += f"\n\n### {link_text}\n{content}"
    combined_content = combined_content.replace('\n', ' ')
    return clean_text(combined_content)

def summarize_with_gemini(text):
    model = genai.GenerativeModel('gemini-1.5-flash', generation_config=genai.GenerationConfig(temperature=0.5))
    response = model.generate_content(f"""Summarize the privacy policy provided in triple quotes. 
    Summarize it in simple and clear language for a general audience.
    The summary should focus on the key points that a common man would need to understand, such as:

    What personal information is collected.
    How the information is used.
    How the information is shared.
    User rights regarding their data.

    Ensure the summary is concise, easy to read, and avoids technical jargon.

    privacy policy: '''{text}'''""")

    return response.text

@app.route('/summarize-url', methods=['POST'])
def summarize_url():
    data = request.json
    url = data['url']
    
    # Fetch and combine content from URL and its links
    tnc, links = fetch_tnc_and_links(url)
    link_contents = {link[0]: fetch_link_content(link[1]) for link in links}
    combined_content = combine_content(tnc, link_contents)

    # Summarize the combined content using Gemini
    combined_summary = summarize_with_gemini(combined_content)
    
    print("hello")
    return jsonify({"summary": combined_summary})

if __name__ == '__main__':
    app.run(debug=True)




import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
import re
import csv
import json
import nltk
import base64
import time
import joblib
import pickle
from pinecone import Pinecone
import logging
import requests
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from slack_sdk.web import WebClient
from requests.auth import HTTPBasicAuth
from slack_sdk.socket_mode import SocketModeClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction



#Download required NLTK resources (Download once)
#nltk.download('punkt')
#nltk.download('stopwords')

# File to store version number
VERSION_FILE = "sms_version.json"

# Load API Keys and environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OpenAI Assistant ID
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
#OPENAI THREAD ID
#thread_id = os.getenv("OPENAI_THREAD_ID")

# Set up logging to file
LOG_FILE = "slack_bot.log"

logging.basicConfig(
    level=logging.DEBUG,  # Log everything from DEBUG and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp + Log Level + Message
    handlers=[
        logging.FileHandler(LOG_FILE),  # Save logs to a file
        logging.StreamHandler()  # Also print logs to the console
    ]
)

logger = logging.getLogger(__name__)  # Create logger instance


# Get environment variables
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")  # You'll need this new token
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")

# Confluence details (store securely in environment variables)
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")  # e.g., "https://your-domain.atlassian.net/wiki"
EMAIL = os.getenv("CONFLUENCE_EMAIL")  # Your email used for Confluence login
API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")  # Generate from Atlassian API tokens
PAGE_ID = os.getenv("CONFLUENCE_PAGE_ID")  # Replace with the actual Confluence page ID
SPACE_ID = os.getenv("CONFLUENCE_SPACE_ID") 

#Pinecone details
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")



# Initialize clients
slack_client = WebClient(token=SLACK_BOT_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc_client = Pinecone(api_key=PINECONE_API_KEY)
index = pc_client.Index(PINECONE_INDEX)



# Create Basic Auth string with email+token format
auth = HTTPBasicAuth(EMAIL , API_TOKEN)

# Headers for authentication and content type
HEADERS = {
    "Authorization" : "auth",
    "Accept" : "application/json",
    "Content-Type": "application/json"
}

def get_current_version():
    """Retrieve the current SMS version from the file."""
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as file:
            try:
                data = json.load(file)
                return data.get("version", 0)  # Default to 0 if not found
            except json.JSONDecodeError:
                return 0
    return 0  # Default version if file doesn't exist

def update_version(new_v):
    """Update the SMS version in the file."""
    with open(VERSION_FILE, "w") as file:
        json.dump({"version": new_v}, file)


#Get english stopwords
stop_words = set(stopwords.words('english'))

# Email regex pattern
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

#Clean and pre-process the slack message
def clean_slack_messages(message):
    """
    Cleans Slack chunks by removing unnecessary symbols and formatting
    """
    if not message or message.strip() == "":
        return "", ""
    
    # Convert to string if not already
    message = str(message)

     # Find all email addresses in the text
    extracted_emails = re.findall(email_pattern, message)

    #Remove emails/ links
    message = re.sub(email_pattern, '', message)
    
    # Remove emoji codes (e.g., :outbox_tray:1f4e4📤)
    message = re.sub(r':[a-zA-Z_]+:[\da-fA-F]+[^\s]*', '', message)

    # Identify and extract angled bracket links: <https://example.com>
    message = re.sub(r'<https? ?://([\w./-]+)>', r'https://\1', message)

    # Remove EAE6FF formatted words (assuming specific pattern, adjust as needed)
    message = re.sub(r'\bEAE6FF\b', '', message)

    #Convert Markdown-style links: [Title](URL) → Title URL
    message = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'\1 \2', message)

    # Remove <@channel> and <@here>
    message = re.sub(r'<[@!][^>]+>', '', message)
    message = re.sub(r'<@(?:channel|here)>', '', message)

    # Clean up extra spaces
    message = re.sub(r'\s+', ' ', message).strip()

    # Remove long horizontal lines (at least 5 underscores in a row)
    message = re.sub(r'_+', '', message)
    
    # Remove downward arrow emoji (⬇️)
    message = re.sub(r'⬇️', '', message)
    
    # Remove emoji Unicode characters
    message = re.sub(r'[\U0001F300-\U0001F9FF]', '', message)
    
    # Remove leading/trailing whitespace
    message = message.strip()
    
    # Remove asterisks
    message = re.sub(r'\*+', '', message)
    
    # Remove vertical bars and dashes in tables
    message = re.sub(r'\||\-{3,}', ' ', message)
    
    # Remove brackets and parentheses
    message = re.sub(r'[\[\]\(\)]', '', message)
    
    # Remove special characters
    message = re.sub(r'[#@\|\-\\\:]', ' ', message)
    
    # Remove URLs
    message = re.sub(r'https?://\S+', '', message)
    
    # Remove multiple spaces
    message = re.sub(r'\s+', ' ', message)

    message = message.strip()

    #Tokens the text using NLTK
    tokens = word_tokenize(message)

    #remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    #Reconstruct the cleaned message
    cleaned_text = " ".join(filtered_tokens)

    logger.info(f"Final cleaned slack data: {cleaned_text}")
    return cleaned_text


#start by fetching the slack message
def on_connect(client: SocketModeClient, response):
    """Called when the client connects successfully"""
    logger.info("Connected to Slack Socket Mode!")
    
def on_disconnect(client: SocketModeClient):
    """Called when the client disconnects"""
    logger.info("Disconnected from Slack Socket Mode!")
    
def on_error(client: SocketModeClient, error):
    """Called when the client encounters an error"""
    logger.error(f"Error in Socket Mode: {error}")
"""
# Load CSV Database (Local File)
CSV_FILE = "vectorized_articles.csv"
if not os.path.exists(CSV_FILE):
    print("⚠️ CSV file not found! Please ensure 'vectorized_articles.csv' exists.")
    exit()

#load pandas dataframe
articles_df = pd.read_csv(CSV_FILE)

# Extract text data
article_texts = articles_df["clean_content"].dropna()

THREAD_ID_FILE = "openai_thread.json"

def get_saved_thread_id():
    #Retrieve the saved thread ID from a file if available
    if os.path.exists(THREAD_ID_FILE):
        try:
            with open(THREAD_ID_FILE, "r") as file:
                data = json.load(file)
                return data.get("thread_id")
        except Exception as e:
            logger.error(f"Error reading thread ID file: {e}")
    return None

def save_thread_id(thread_id):
    #Save the thread ID to a file for reuse.
    try:
        with open(THREAD_ID_FILE, "w") as file:
            json.dump({"thread_id": thread_id}, file)
        logger.info(f"✅ Saved thread ID: {thread_id}")
    except Exception as e:
        logger.error(f"Error saving thread ID: {e}")


# Load the saved RFC model (74%) and label encoder
rfc_model = joblib.load("random_forest_model.pkl")
label_encoder = joblib.load("random_forest_label_encoder.pkl")"""

#Load the Support vector machine model and label encoder
svm_model = joblib.load("support_vector_machine_model.pkl")
label_encoder = joblib.load("support_vector_machine_label_encoder.pkl")

# Load the same sentence embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def classify_message(cleaned_text):
    """Classifies the cleaned slack message using Random Forest Classification model"""
    logger.info(f"Classifying the message using RFC:{cleaned_text}")

    #Convert the cleaned slack message into embeddings
    global message_embedding
    message_embedding = embedding_model.encode([cleaned_text]).reshape(1,-1) 

    #Predict category using RFC
    prediction = svm_model.predict(message_embedding)
    predicted_label = label_encoder.inverse_transform(prediction)
    ai_response = predicted_label[0]
    logger.info(f"Predicted message category: {ai_response}")
    return ai_response

def find_related_article(cleaned_text, similarity_threshold=0.6, top_k=5):
    """Find the most relevant Confluence article using Pinecone similarity search."""

    try:
        # ✅ Generate Slack vector
        slack_vector = embedding_model.encode([cleaned_text]).astype(np.float32)
    except Exception as e:
        logger.error(f"⚠️ Error generating Slack vector: {e}")
        return None, None

    try:
        # ✅ Query Pinecone using precomputed embedding
        pinecone_results = index.query(
            vector=slack_vector[0].tolist(),
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        logger.error(f"⚠️ Error querying Pinecone: {e}")
        return None, None

    # ✅ Check if any matches were returned
    if not pinecone_results.matches:
        logger.warning("⚠️ No matches found in Pinecone.")
        return None, None

    # ✅ Convert to similarity using cosine distance
    matches = pinecone_results.matches
    scores = np.array([match.score for match in matches])
    best_match_index = int(np.argmax(scores))
    best_match_score = scores[best_match_index]

    logger.info(f"✅ Best Match Score: {best_match_score:.4f}")

    # ✅ Validate and return metadata
    if best_match_score > similarity_threshold:
        best_match = matches[best_match_index]
        metadata = best_match.metadata

        article_title = metadata.get("title", "Unknown Title")
        article_id = metadata.get("article_id", "Unknown ID")

        logger.info(f"✅ Best Match Found: {article_title} (ID: {article_id})")
        return article_title, article_id
    else:
        logger.warning("⚠️ No relevant article found with high similarity.")
        return None, None



def handle_slack_event(client: SocketModeClient, req: SocketModeRequest):
    """Handle incoming Slack events."""
    logger.info(f"Received event type: {req.type}")
    logger.info(f"Full event payload: {req.payload}")
    logger.info(type(f"Payload type: {req.payload}"))


    if req.type == "events_api":
        # Acknowledge the request
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

        event = req.payload.get("event" , {})
        logger.info(f"Event received: {event}")

        
        # Only process messages from the specified channel
        if event["type"] == "message" and event.get("channel") == CHANNEL_ID:
            logger.info(f"🔹 Received Message: {event.get('text')} from {event.get('user')} in {event.get('channel')}")
            #Initialize variables
            final_response = None
            ai_response = ""
            final_cleaned_slack_text = ""

            # Ignore bot messages and message updates
            if "bot_id" not in event and "subtype" not in event:
                try:
                    #Get the slack message
                    slack_text = event.get("text", "")

                    #Clean the slack message
                    final_cleaned_slack_text = clean_slack_messages(slack_text)
                    thread_ts = event.get("thread_ts")
                    logger.info(f"Fetched thread_ts: {thread_ts}")

                    #Extract the message
                    full_event_payload = None
                    full_event_payload = event

                    logger.info(f"Extracted slack message: {full_event_payload}")

                    try:
                        # ✅ Convert to JSON formatted string with proper indentation & double quotes
                        json_string = json.dumps(full_event_payload, indent=4, ensure_ascii=False)

                        # ✅ Convert formatted JSON string to bytes (UTF-8 encoding)
                        payload_bytes = json_string.encode("utf-8")

                        # ✅ Save the bytes to a file
                        with open("extracted_slack_event.json", "wb") as f:
                            f.write(payload_bytes)

                        logger.info("✅ Slack event payload saved successfully as a properly formatted JSON file in bytes.")

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse JSON: {e}")

                    #Extract the text as it is from the slack message 

                    # ✅ Load JSON from the uploaded file
                    with open("extracted_slack_event.json", "r", encoding="utf-8") as f:
                        json_data = json.load(f)

                    def extract_rich_text_sections(json_data):
                        """Extracts and prints all 'text' and 'link' elements from 'rich_text_section'."""
                        extracted_text = []

                        # ✅ Navigate to event -> blocks
                        blocks = json_data.get("blocks", [])

                        for block in blocks:
                            if block.get("type") == "rich_text":
                                elements = block.get("elements", [])
            
                                for element in elements:
                                    if element.get("type") == "rich_text_section":
                                        for item in element.get("elements", []):
                                            if item["type"] == "text":
                                                extracted_text.append(item["text"])  # ✅ Append normal text
                                            elif item["type"] == "link":
                                                extracted_text.append(f"({item['url']})")  # ✅ Append link with text

                                    # ✅ Extract text from rich_text_list (bullet points)
                                    elif element.get("type") == "rich_text_list":
                                        for list_item in element.get("elements", []):
                                            for list_element in list_item.get("elements", []):
                                                if list_element["type"] == "text":
                                                    extracted_text.append(f"• {list_element['text']}")  # ✅ Add bullet point style
                                                elif list_element["type"] == "link":
                                                    extracted_text.append(f"{list_element['url']}")  # ✅ Add links inside lists

                            # ✅ Join all text into a single string and print
                            final_text = " ".join(extracted_text)
                            print("✅ Extracted Message Content:")
                            

                            # ✅ Save extracted content to a file
                            with open("extracted_text.json", "w", encoding="utf-8") as f:
                                json.dump({"extracted_text": final_text}, f, indent=4, ensure_ascii=False)

                            return final_text

                    # ✅ Run the function and extract text
                    extracted_content = extract_rich_text_sections(json_data)

                    # Get thread messages if it's in a thread
                    thread_messages = []
                    if thread_ts:
                        try:
                            replies = slack_client.conversations_replies(
                                channel=CHANNEL_ID,
                                ts=thread_ts
                            )
                            thread_messages = [msg["text"] for msg in replies.get("messages", [])[1:]]
                            logger.info(f"Fetching messages from thread...")
                        except Exception as e:
                            print(f"Error fetching thread messages: {e}")

                    # Combine thread messages
                    if thread_messages:
                        final_cleaned_slack_text += " " + " ".join(thread_messages)

                    # Process message classification
                    try:
                        #Classify the cleaned slack message
                        final_response = classify_message(final_cleaned_slack_text)
                        logger.info(f"Message category predicted by RFC: {final_response}")
                        print(f"🔍 Message Category predicted by RFC: {final_response}")
                    except Exception as e:
                        print(f"Error classifying message: {e}")
                        final_response = None

                    # Only proceed if we have a category
                    current_v = get_current_version()
                    new_v = current_v + 1
                    update_version(new_v)
                    if final_response:
                        if final_response in ["process update", "product enhancement", "carrier notification"]:
                            # Use hardcoded article title for testing
                            article_title, article_id = find_related_article(final_cleaned_slack_text)
                            
                            ai_response = f"Category by RFC: {final_response}\n✅ Updating Confluence article: {article_title}"
                            
                            # Format and update Confluence
                            formatted_content = f"""
                            <ac:layout><ac:layout-section ac:type="fixed-width" ac:breakout-mode="default"><ac:layout-cell><ac:structured-macro ac:name="children" ac:schema-version="2" data-layout="default"/>
                            <p><h3>Latest Update from Slack channel:</h3></p>
                            <p>Posted on: <em>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
                            <p><strong>Category by SVM:</strong> {final_response}</p>
                            <p>AI chosen article using cosine similarity search:<strong> {article_title}. Id: {article_id}</strong></p>
                            <blockquote>
                                <p>{extracted_content}</p>
                            </blockquote>
                            """
                            
                            # Update Confluence
                            if article_id:
                                success = update_confluence_page(article_id, formatted_content)
                            else:
                                logger.warning(f"No relevant article found. Updating default testing article {PAGE_ID} ")
                                success = update_confluence_page(PAGE_ID, formatted_content)

                            #check if confluence pages are updated successfully.
                            if success:
                                ai_response += "\n✅ Confluence page updated successfully!"
                            else:
                                ai_response += "\n❌ Failed to update Confluence page."
                        else:
                            ai_response = f"Category: {final_response}✅ No Confluence update needed."

                    # Reply in thread if it's a thread message, otherwise send as a new message
                    if ai_response:
                        thread_ts = event.get("thread_ts", event.get("ts"))
                        logger.info(f"Thread timestamp (thread_ts): {thread_ts}")
                        try:
                            slack_client.chat_postMessage(
                                channel=CHANNEL_ID,
                                text=ai_response,
                                thread_ts=thread_ts
                            )
                            logger.info("Message posted successfully in thread")
                        except Exception as e:
                            print(f"Error sending response to Slack: {e}")

                except Exception as e:
                    print(f"Error processing message: {e}")
                    logger.error(f"Error processing message: {e}")
                    try:
                        thread_ts = event.get("thread_ts", event.get("ts"))
                        slack_client.chat_postMessage(
                            channel= CHANNEL_ID,
                            text = f"Error procesing message: {str(e)}",
                            thread_ts=thread_ts
                        )
                    except:
                        pass



def update_confluence_page(PAGE_ID, formatted_content):
    """Update Confluence page using Atlassian's recommended authentication"""
    try:
        # Confluence details
        CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
        EMAIL = os.getenv("CONFLUENCE_EMAIL")
        API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

        # Set up authentication
        auth = HTTPBasicAuth(EMAIL, API_TOKEN)

        # Headers
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        # First get the current version of the page
        get_url = f"{CONFLUENCE_BASE_URL}/api/v2/pages/{PAGE_ID}"

        print(f"Requesting URL: {get_url}")
        
        # Debug print
        print(f"Fetching page content...")
        
        response = requests.get(
            get_url,
            headers=headers,
            auth=auth
        )
        
        if response.status_code == 404:
            print(f"Page not found. Please check:\n"
                  f"1. Page ID: {PAGE_ID} exists\n"
                  f"2. Base URL: {CONFLUENCE_BASE_URL} is correct\n"
                  f"3. You have permission to access this page")
            return False


        if response.status_code != 200:
            print(f"Error fetching page: {response.status_code}, {response.text}")
            return False

        page_data = response.json()
        current_version = page_data['version']['number']
        page_title = page_data['title']

        #Get page content using the content endpoint
        content_url = f"{CONFLUENCE_BASE_URL}/rest/api/content/{PAGE_ID}?expand=body.storage.version"

        content_response = requests.get(
            content_url,
            headers=headers,
            auth=auth
        )

        if content_response.status_code !=200:
            print(f"Error fetching content: {content_response.status_code}, {content_response.text}")
            return False
        
        content_data = content_response.json()
        existing_content = content_data.get('body', {}).get('storage', {}).get('value', '')
        print(f"Current version: {current_version}")
        logger.info(f"Existing content: {existing_content}")

        #Combine existing content with new content
        combined_content = f"{existing_content}\n\n\n {formatted_content}"
        
        # Prepare update data
        new_version = current_version + 1
        update_data = {
            "id": PAGE_ID,
            "status" : "current",
            "type": "page",
            "title": page_data['title'],
            "space": {"key": SPACE_ID},
            "body": {
                "storage": {
                    "value": combined_content,
                    "representation": "storage"
                }
            },
            "version": {"number": new_version}
        }

        # Update the page
        update_url = f"{CONFLUENCE_BASE_URL}/api/v2/pages/{PAGE_ID}"
        
        print(f"Updating page...")
        response = requests.put(
            update_url,
            headers=headers,
            auth=auth,
            json=update_data
        )

        if response.status_code == 200:
            print("✅ Page updated successfully")
            return True
        else:
            print(f"Error updating page: {response.status_code}, {response.text}")
            return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False



def main():
    """Main function to run the Slack event listener."""
    logger.info("=== Starting Slack Bot ===")

    # Initialize Socket Mode client
    logger.info("Initializing Socket Mode client...")
    app = SocketModeClient(
        app_token=SLACK_APP_TOKEN,
        web_client=slack_client
    )

    # Add event handler
    app.socket_mode_request_listeners.append(handle_slack_event)

    # Start the app
    logger.info("⚡️ Connecting to Slack Socket Mode...")
    try:
        app.connect()
        logger.info("Socket Mode connection initiated...")
    except Exception as e:
        logger.error(f"Failed to connect to Socket Mode: {e}")
        return

    # Keep the program running
    import signal

    def signal_handler(signum, frame):
        logger.info("\nStopping the application...")
        app.disconnect()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Bot is running and waiting for events...")

    # Keep the process alive
    while True:
        try:
            signal.pause()
        except AttributeError:
            import time
            time.sleep(1)


if __name__ == "__main__":
    main()

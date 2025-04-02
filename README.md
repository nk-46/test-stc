Overview of the Workflow:

1. Slackbot fetches the new message posted on the #support-important-updates channel

2. Slack message is cleaned and passed through an AI model.
  
3. Pre- trained AI models processes the message using Random Forest Classifier

4. Uses NLP to extract context (e.g., topic, keywords).

5. Convert slack message into embeddings.

6. Matches the topic to a Confluence page.

7. Confluence articles are fetched using APIs, pre-processed, divided into chunks and converted into vectors
   
8. Confluence vectors are stored in ChromaDB

9. A similarity search using cosinesimilarity() returns a suggested Confluence page to update.

10. If the similarity between slack message and the confluence chunk > 60%, the confluence page is returned and updated.

How the Code Flows:

🚀 Start the bot: python main.py

📡 Bot connects to Slack and listens for messages.

📝 Message arrives → Bot cleans the message.

🔍 Message is classified → Finds related article using ChromaDB.

📚 Confluence article is updated dynamically.

💬 Slack thread is updated with success/failure status.




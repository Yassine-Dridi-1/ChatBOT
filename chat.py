from transformers import BertForQuestionAnswering, BertTokenizerFast
import torch
import nltk
import subprocess

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import smtplib
from email.mime.text import MIMEText



def send_email(recipient_email, subject, message):
    sender_email = 'yd54597@gmail.com'
    sender_password = 'yassinedridi123!!'
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    # Create a secure SSL connection
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)

    # Create a MIMEText object to represent the email
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Send the email
    server.sendmail(sender_email, recipient_email, msg.as_string())

    # Close the server connection
    server.quit()

def send_warning_email():
    recipient_email = input("Please enter your email address: ")
    subject = "Warning Alert"
    message = " wanted to bring to your attention some concerns regarding your recent network activity. It appears there have been instances of misuse on the company's network, which is against our policies. Please ensure you adhere to our network usage guidelines to maintain a secure and efficient environment for all employees. Further misuse may result in penalties.\n\nThank you for your understanding and cooperation."

    send_email(recipient_email, subject, message)
    # run nmap
def run_nmap_scan():
    try:
        # Prompt user for address and port
        address = input("Enter the address to scan (e.g., localhost): ")
        port_range = input("Enter the port range to scan (e.g., 1-1024): ")

        # Construct the nmap command
        nmap_command = f"nmap -p {port_range} {address}"

        # Run the nmap scan
        result = subprocess.run(nmap_command, shell=True, capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # Print the output with proper formatting
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")

    except Exception as e:
        print(f"Error occurred during scan: {e}")

# connect to database
connection = mysql.connector.connect(
    host='127.0.0.1',
    user='root',
    password='',
    database='contexts'
)

# Create a cursor object to interact with the database
cursor = connection.cursor()

# Execute a query
query = "SELECT * FROM context"
cursor.execute(query)

# Fetch the results
contexts = cursor.fetchall()

# Close the cursor and connection
cursor.close()
connection.close()


# Load the fine-tuned SEC-BERT model
model_path = './BERT'
model = BertForQuestionAnswering.from_pretrained(model_path)
# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_path)
# Set the device to CPU
device = torch.device('cpu')
model = model.to(device)
def get_prediction(context, question):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0, answer_start:answer_end]))

    return answer

# Your input context and question
while True :
 question =input('give me a question \n') 
 if "send a warning email" in question:
    # Call the function to send the email
    send_warning_email()
 elif "scan system" in question: 
         run_nmap_scan()
         print("System scan completed.")    
 else :

# Preprocess user question
    stop_words = set(stopwords.words('english'))
    user_question_tokens = [word.lower() for word in word_tokenize(question) if word.lower() not in stop_words]

 
# Preprocess contexts
    processed_contexts = [' '.join([word.lower() for word in word_tokenize(context[1]) if word.lower() not in stop_words]) for context in contexts]

# Generate TF-IDF vectors for user question and contexts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([question] + processed_contexts)

# Calculate cosine similarity
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

# Find the context with the highest similarity score
    most_similar_context_index = similarities.argmax()

# Retrieve the most relevant context
    most_similar_context = str(contexts[most_similar_context_index])

    prediction = get_prediction(question, most_similar_context) 
    print(prediction)

 


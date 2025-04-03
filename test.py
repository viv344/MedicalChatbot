import streamlit as st
import pandas as pd
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Disable the warning about using a pre-trained model without specifying the number of labels
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set page configuration
st.set_page_config(page_title="Medical Assistant Chatbot", page_icon="🩺")

# Load and prepare the model and data
@st.cache_resource
def load_saved_model(model_path="best_clinical_bert_model.pt", model_name="emilyalsentzer/Bio_ClinicalBERT"):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model, tokenizer

@st.cache_data
def load_and_prepare_data(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    return df

class MedicalChatbot:
    def __init__(self, model, tokenizer, qa_df):
        self.model = model
        self.tokenizer = tokenizer
        self.qa_df = qa_df
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.qa_df['question'].astype(str))
    
    def find_most_similar_questions(self, query, top_k=5):
        """Find the most similar questions in our dataset"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        
        # Get the indices of the top k most similar questions
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [
            (self.qa_df.iloc[idx]['question'], 
             self.qa_df.iloc[idx]['answer'],
             similarities[idx])
            for idx in top_indices
        ]
    
    def get_best_answer(self, query):
        """Get the best answer for a query using ClinicalBERT and similarity search"""
        similar_qa_pairs = self.find_most_similar_questions(query)
        
        best_score = -float('inf')
        best_answer = "I don't have enough information to answer this medical question."
        
        for question, answer, similarity in similar_qa_pairs:
            # Skip if similarity is too low
            if similarity < 0.3:
                continue
                
            # Use ClinicalBERT to score the relevance of this answer to the query
            inputs = self.tokenizer(
                query,
                answer,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                score = logits[0, 1].item()  # Score for the positive class
            
            # Combine BERT score and similarity
            combined_score = 0.1 * score + 0.9 * similarity
            
            if combined_score > best_score:
                best_score = combined_score
                best_answer = answer
        
        #print(similar_qa_pairs)
        if best_answer == "I don't have enough information to answer this medical question.":
            return self.LLM(query)
        else:
            return best_answer
    
    def LLM(self, question):
        try:
            os.environ["GOOGLE_API_KEY"] = "AIzaSyBDCIWtBZ_Oqsl67LDaoK3CABFwyIIrdJg"
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            # Load the Gemini 1.5 model
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Ask a question
            response = model.generate_content(question)

            return response.text
        except Exception as e:
            st.error(f"Error with Gemini API: {e}")
            return "I'm sorry, but I'm having trouble accessing additional information right now."

def main():
    # Set up the Streamlit app
    st.title("🩺 Medical Assistant Chatbot")
    st.write("Ask me medical questions and I'll help you find answers!")

    # Sidebar for additional information
    st.sidebar.header("About the Chatbot")
    st.sidebar.info(
        "This medical assistant uses a combination of ClinicalBERT and Gemini AI "
        "to provide informative medical responses. Always consult a healthcare "
        "professional for personalized medical advice."
    )

    # Load the model and data
    try:
        model, tokenizer = load_saved_model()
        qa_df = load_and_prepare_data('med_qa.csv')
        
        # Initialize the chatbot
        chatbot = MedicalChatbot(model, tokenizer, qa_df)

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask a medical question"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get chatbot response
            response = chatbot.get_best_answer(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error initializing the chatbot: {e}")
        st.warning("Please check that the model and data files are correctly loaded.")

if __name__ == "__main__":
    main()
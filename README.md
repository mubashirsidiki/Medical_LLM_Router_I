# Medical LLM Router

A specialized medical query system that routes health questions to appropriate specialist agents based on domain classification.

## About

Medical LLM Router analyzes patient medical history documents and answers health-related questions by:
1. Classifying queries into specialized domains (General Health, Neurology, Cardiology, Orthopedics)
2. Retrieving relevant context from medical history
3. Generating personalized responses from domain-specific agents

## Demo

Access the deployed application: [Medical LLM Router on Streamlit](https://medicalllmrouteri-k47vnn8io8qpz8v5bujtqt.streamlit.app/)

## Features

- **Medical Document Processing**: Processes PDF medical history documents
- **Query Classification**: Classifies health queries by medical specialty
- **Specialized Agents**: Routes queries to domain-specific medical agents
- **Context Retrieval**: Finds relevant information from patient medical history
- **Vector Store Caching**: Reuses document embeddings for faster responses

## Setup

### Prerequisites

- Python 3.11
- Git

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/mubashirsidiki/Medical_LLM_Router_I.git
   cd Medical_LLM_Router_I
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your API keys see .env.example for reference

## Running the Application

Start the Streamlit app:
```
streamlit run medical_assistant.py
```

## Usage

1. Upload a medical history document (PDF)
2. Enter a health-related question
3. View the specialized response based on your medical history

## Note

This application provides information only, not medical diagnosis or treatment. Always consult healthcare professionals for medical advice. 

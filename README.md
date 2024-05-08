# GATA

The generative AI teaching assistant (GATA) addresses students' need for fast, context-dependent answers when studying and completing assignments. 

GATA is a Python application built using Streamlit to create a conversational interface powered by OpenAI's language models and Langchain's retrieval libraries. It allows users to interact with the chatbot for educational assistance. The multimodal chatbot provides students with a text answer, specific lecture slides, and video clips from classroom lectures. GATA only retrieves relevant sources that help answer specific questions. Instructors can quickly and easily add to the knowledge base.

## Features

- **Chat Interfacce:** Users can converse with the chatbot in a natural language format
- **Uploading Content:** The application can process PDF documents and lecture videos with transcripts to store text information in retrievable chunks
- **Data Storage:** MongoDB is used for data storage and user management. FAISS vector databases are used to store contextual information
- **Web Application:** The chatbot interface is built using Streamlit, making it accessible via a web browser
- **Authentication:** Users can log in or sign up as students or instructors to access personalized features
- **Source Retrieval:** The application retrieves relevant sources from syllabi, textbooks, slides, and videos for answering user queries
- **Post Forums:** Users can make posts to the class, search posts, and add follow-up discussions to existing posts like in current tools such as Piazza
- **Instructor Tools:** Instructors have access to additional functionalities such as configuring GATA content, viewing user metrics, and managing student rosters
- **Student Tools:** Students have access to metrics and can view the student roster to form class study groups

## Installation

Ensure you have Python installed, then run the following command to install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### Vector Database Setup

To initialize the vector databases for slide, lecture video, syllabus, and textbook context retrieval:

1. Navigate to the `vdb_setup.py` script.
2. Fill in the paths to your syllabus, textbook PDFs, lecture video transcripts, and lecture slides folders
```python
syllabus = "" # Enter the path to the syllabus pdf file
tb = "" # Enter the path to the textbook pdf file
transcript = "" # Enter the path to the lecture video transcript folder
pdf_path = "" # Enter the path to the lecture slides folder
```
3. Run the `vdb_setup.py` script.

## Configuration

Update or create `secrets.toml` within the `.streamlit` subdirectory with MongoDB and OpenAI API keys to connect to the necessary services
```python
[openai]
api_key = ""

[mongo]
host = ""
username = ""
password = ""
dbname = ""
```

## Usage

Launch the application:
```bash
python streamlit run gata_app.py
```

The application will be accessible in your web browser at http://localhost:8501

## Code Structure
- `gata_app.py`: Main Python script containing application logic
- `secrets.toml`: Configuration file containing sensitive data such as database credentials and api secret keys; should be placed within the .streamlit subdiretory

## How it Works: Context Retrieval with HyDE

The application extracts context from syllabi, textbooks, slides, and videos to provide more accurate responses to user queries. Multiple measures are taken to ensure maximum performance. In the current iteration, four vector databases are provided, each specialized for textbooks, lecture videos, slides, or syllabus information. Stored in each vector database is specific metadata that can be extracted to provide sources of information such as images and videos. Hypothetical Document Embedding (HyDE) is used to retrieve this information accurately. Retrieval often struggles to match a query with an answer through cosine similarity and other vector embedding retrieval techniques. HyDE generates hypothetical answers that more closely align with the embedding structure of ideal answers. Lastly, there is a similarity threshold that the retrievers must reach when returning information. This specification forces GATA to provide only information most relevant to individual queries.

## Future Work

GATA is just the beginning. This application showcases the potential of configuring current generative AI tools with the foundational knowledge of a teacher's assistant. Soon, GATA will be able to handle multiple courses and universities simultaneously, revolutionizing how we learn. We're thrilled with the positive feedback we've received and eagerly anticipate the new developments in generative AI that could further enhance GATA's capabilities.

## Demo Video
For in depth examples of usage, please refer to the GATA product demo video:
[GATA Demo Video](https://youtu.be/PGZ1gMrjBu4)










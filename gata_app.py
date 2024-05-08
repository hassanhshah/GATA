# Standard libraries
from datetime import datetime, timedelta
import hashlib
import os
import re
import time
import base64

# Data handling
import pandas as pd

# PDF/OCR
import fitz
from PyPDF2 import PdfReader

# MongoDB connection
import pymongo

# Streamlit for web application
import streamlit as st

# OpenAI API
from openai import OpenAI

# LangChain for embedding and retrieval
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

# Other libraries
from dateutil.parser import parse

# Streamlit configuration
st.set_page_config(
    page_title="GATA",
    initial_sidebar_state="expanded"
)

# Session States
if 'display_page' not in st.session_state:
    st.session_state.display_page = 'GATA_CHAT'

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'registration_key' not in st.session_state:
    st.session_state['registration_key'] = "Fourier Transform" 

# Global folder variables
folders = ['hw1', 'hw2', 'hw3', 'hw4', 'project', 'logistics', 'other', 'quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5', 'exams']
filter = ['All Posts', 'hw1', 'hw2', 'hw3', 'hw4', 'project', 'logistics', 'other', 'quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5', 'exams']

# For displaying logo
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
logo = image_to_base64('logo.png')

# Initialize MongoDB connection
@st.cache_resource
def init_connection():
    try:
        # Construct the full connection string
        mongo_secrets = st.secrets["mongo"]
        connection_string = f"mongodb+srv://{mongo_secrets['username']}:{mongo_secrets['password']}@{mongo_secrets['host']}"
        
        client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        return client

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

client = init_connection()
db = client.ETA  # use your database name here

# User Registration and Sign Up
def register_user(db, first_name, last_name, email, password, instructor):
    if len(password) < 8:
        st.error("Password must be at least 8 characters long.")
        return 
    if db.users.count_documents({"email": email}) > 0:
        st.error("Email already in use. Please choose a different one.")
        return

    next_user_id = db.users.count_documents({}) + 1
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user_data = {
        "user_id": next_user_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "hashed_password": hashed_password,
        "instructor": instructor,
        "created_at": datetime.now().isoformat(),
        "last_logged_in": datetime.now().isoformat()
    }
    db.users.insert_one(user_data)

    st.session_state['logged_in'] = True
    st.session_state['user'] = user_data
    st.session_state['display_page'] = 'GATA_CHAT'
    st.success(f"Welcome {first_name}, you are now registered and logged in!")
    return True

# Checking database for login credentials
def validate_login(db, email, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user = db.users.find_one({"email": email, "hashed_password": hashed_password})
    if user:
        db.users.update_one(
            {"email": email},
            {"$set": {"last_logged_in": datetime.now().isoformat()}}
        )
        return True, user
    return False, "Login failed. Try again or reset password."

# Calculating metrics for the dashboard
def calculate_metrics(_db):
    total_questions = _db.posts.count_documents({"post": "Question"})
    answered_questions = _db.posts.count_documents({
        "post": "Question",
        "gata_response": {"$exists": True},
        "gata_response.content": {"$ne": "Answer not found"}
    })
    response_times = _db.posts.find({
        "post": "Question",
        "gata_response": {"$exists": True},
        "gata_response.content": {"$ne": "Answer not found"}
    }, {"datetime": 1, "gata_response.timestamp": 1})

    total_response_time = 0
    valid_responses = 0
    for post in response_times:
        post_datetime = parse(post['datetime'])
        response_datetime = parse(post['gata_response']['timestamp'])
        response_duration = (response_datetime - post_datetime).total_seconds()
        total_response_time += response_duration
        valid_responses += 1

    average_response_time = total_response_time / valid_responses if valid_responses else 0
    return total_questions, answered_questions, average_response_time

# Displaying initial portal for login and sign up
def display_auth_forms():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center;">
            <img class="header-icon" src="data:image/png;base64,{logo}" width="200"/>
        </div>

        ###

        <h3>Welcome to <b>GATA</b>!</h3>

        <p>This interactive platform enhances your educational experience through a blend of AI and community-driven content. Whether you're a student seeking clarity or an instructor looking to enrich your teaching tools, GATA is designed to cater to your academic needs.</p>

        <h4>Key Features:</h4>
        <ul>
            <li><b>GATA Chat</b>: Engage with our Generative AI Teaching Assistant (GATA) to get instant answers and explanations tailored to your course material.</li>
            <li><b>Discussion Forum</b>: Participate in class discussions, ask questions, and share insights with peers and instructors.</li>
            <li><b>Document and Video Analysis</b>: Access AI-driven insights from course materials, including textbooks, slides, and lecture videos, to enhance your understanding and retention.</li>
            <li><b>Performance Metrics</b>: Instructors can track engagement and question response times, ensuring that every student query is effectively addressed.</li>
        </ul>

        <p><b>Please Note</b>: While GATA aims to provide accurate and helpful information, interactions with GATA should complement your learning and not replace formal educational pursuits.</p>

        <p><b>What's Coming?</b> Look forward to more interactive features and tools to make your learning experience even more comprehensive!</p>

        <p><i>Coming Soon: New AI models for deeper content analysis and personalized learning!</i></p>
        """, unsafe_allow_html=True)
    with st.container(border=True):
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
        with tab_login:
            st.subheader("Login")
            email = st.text_input("Email", key="login_email").lower()
            password = st.text_input("Password", type='password', key="login_password")
            col1, col2 = st.columns([3, 1])
            if col1.button("Login"):
                success, user = validate_login(db, email, password)
                if success:
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = user
                    st.session_state['display_page'] = 'GATA_CHAT'
                    st.success("Successfully logged in.")
                    with st.spinner('Loading... Please wait.'):
                        time.sleep(2)
                        st.rerun()
                else:
                    st.error(user)
            col2.markdown("<small style='color: blue; cursor: pointer;' onclick='alert(\"Please contact support to reset your password.\")'>Forgot your password?</small>", unsafe_allow_html=True)

        with tab_signup:
            user_type = st.radio("Sign up as:", ('Student', 'Instructor'), horizontal=True)
            if user_type == 'Student':
                st.subheader("Student Sign Up")
                registration_key = st.text_input("Enter Registration Key:", key="student_registration_key")
                
                # Check if the correct registration key is entered
                if registration_key:
                    if registration_key == st.session_state['registration_key']:
                        st.success("Registration Key Accepted.")
                        st.info("🎓 You are signing up for: **Machine Learning (DSCI 552, Spring 2024)**")
                        
                        first_name = st.text_input("First Name", key="student_signup_first_name")
                        last_name = st.text_input("Last Name", key="student_signup_last_name")
                        email = st.text_input("Email", key="student_signup_email").lower()
                        password = st.text_input("Password", type='password', key="student_signup_password")
                        confirm_password = st.text_input("Confirm Password", type='password', key="student_confirm_password")
                        if st.button("Register as Student"):
                            if password == confirm_password:
                                success = register_user(db, first_name, last_name, email, password, instructor=False)
                                if success:
                                    with st.spinner('Loading... Please wait.'):
                                        time.sleep(2)
                                        st.rerun()
                                else:
                                    st.error(user)
                            else:
                                st.error("Passwords do not match. Please try again.")
                    else:
                        st.error("Registration Key Incorrect. Please try again or contact your instructor.")

            if user_type == 'Instructor':
                st.subheader("Instructor Sign Up")
                first_name = st.text_input("First Name", key="instructor_signup_first_name")
                last_name = st.text_input("Last Name", key="instructor_signup_last_name")
                email = st.text_input("Email", key="instructor_signup_email").lower()
                password = st.text_input("Password", type='password', key="instructor_signup_password")
                confirm_password = st.text_input("Confirm Password", type='password', key="instructor_confirm_password")
                if st.button("Register as Instructor"):
                    if password == confirm_password:
                        success = register_user(db, first_name, last_name, email, password, instructor=False)
                        if success:
                            with st.spinner('Loading... Please wait.'):
                                time.sleep(2)
                                st.rerun()
                        else:
                            st.error(user)
                    else:
                        st.error("Passwords do not match. Please try again.")

# Fetching posts data from MongoDB
def get_data():
    return list(db.posts.find({}))

client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Extracting text from PDF files
def open_pdf_page(pdf_path, page_number):
    pdf_document = fitz.open(pdf_path)
    if 0 < page_number <= pdf_document.page_count:
        page = pdf_document.load_page(page_number - 1)
        pix = page.get_pixmap()
        image_path = f"page-{page.number}.png"
        pix.save(image_path)
        st.image(image_path, caption=f"Page {page_number}")
        os.remove(image_path)

# Handling timestamps in video files and transcripts
def parse_timestamp(timestamp_str):
    parts = timestamp_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds, milliseconds = map(int, parts[2].split(','))
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return int(total_seconds)

# Helper function to extract text from PDF files
def get_context(doc):
    output_txt = ''
    for d in doc:
         output_txt += d.page_content
    return output_txt

# Displaying and creating the GATA chat interface
def ta_chat(query):
    start_time = time.time()
    with st.spinner('Processing your query...'):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        v_store_pdf= FAISS.load_local("faiss_slides", embeddings,allow_dangerous_deserialization=True)
        v_store_vid = FAISS.load_local("faiss_vids", embeddings,allow_dangerous_deserialization=True)
        v_store_syl= FAISS.load_local("faiss_syllabus", embeddings,allow_dangerous_deserialization=True)
        v_store_txt = FAISS.load_local("faiss_textbook", embeddings,allow_dangerous_deserialization=True)

        retriever_syl = v_store_syl.as_retriever( search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.05})
        retriever_txt = v_store_txt.as_retriever( search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.05})
        retriever_pdf = v_store_pdf.as_retriever( search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2})
        retriever_vid = v_store_vid.as_retriever( search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1})

        system = """Your job is to answer the question to the best of your abilities. If you don't know exactly do your best to make up an answer that could be right in a certain scenario."""

        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{query}"),
        ]
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key = st.secrets["openai"]["api_key"], temperature=0)
        qa_no_context = prompt | llm | StrOutputParser()


        answer = qa_no_context.invoke(
        {
        "query": {query}
        }
        )

        docs_pdf = retriever_pdf.get_relevant_documents(answer)
        docs_vid = retriever_vid.get_relevant_documents(answer)
        docs_syl = retriever_syl.get_relevant_documents(answer)
        docs_txt = retriever_txt.get_relevant_documents(answer)

        syllabus_context = get_context(docs_syl)
        textbook_context = get_context(docs_txt)
        slides_context = get_context(docs_pdf)
        video_context = get_context(docs_vid)

        prompt = f"Use only the following pieces of context to answer the following query.\n--- query start\n{query}\n--- query end\nIf there is not enough context to answer the query please respond: Answer not found\nUse only the following context:\n--- Syllabus start\n{syllabus_context}\n--- Syllabus end\n--- Textbook start\n{textbook_context}\n--- Textbook end\n--- Slides start\n{slides_context}\n--- Slides end\n--- Lecture video start\n{video_context}\n--- Lecture video end\nPlease output in clear and concise sentences if you have found an answer to the query. If not respond: Answer not found\n"    
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an electronic TA. Your job is to answer questions using only context given from the syllabus, textbook, slides, and videos. If there is none then you should output that you do not know the answer."},
                {"role": "user", "content": prompt}
            ]
        )
        response = completion.choices[0].message.content

        sorted_pdf_docs = sorted(docs_pdf, key=lambda d: int(d.page_content.split('\n')[1]))

        with st.container(border=True):
            # Styled answer box
            with st.container():
                st.markdown("### Answer")
                st.write(response)

            with st.expander("Slide Information", expanded=True):
                if docs_pdf:
                    sorted_pdf_docs = sorted(docs_pdf, key=lambda d: int(d.page_content.split('\n')[1]))

                    num_columns = min(len(sorted_pdf_docs), 2)
                    columns = st.columns(num_columns)

                    for index, d in enumerate(sorted_pdf_docs):
                        lines = d.page_content.split('\n')
                        file_path = lines[0]
                        page_num = int(lines[1])
                        file_name = os.path.basename(file_path)
                        # Assuming the file name format is 'Lesson-{number}_{title}.pdf'
                        parts = file_name.split('_')
                        lecture_number = parts[0].replace('Lesson-', '')
                        title = ' '.join(parts[1:]).split('.pdf')[0]

                        col = columns[index % num_columns]
                        with col:
                            col.write(f"Lecture {lecture_number}: {title}, Slide {page_num}")
                            open_pdf_page(file_path, page_num)
                else:
                    st.info("No relevant slide information found.")

            with st.expander("Video Information", expanded=True):
                if docs_vid:
                    num_columns = min(len(docs_vid), 2)
                    columns = st.columns(num_columns)

                    for index, d in enumerate(docs_vid):
                        lines = d.page_content.split('\n')
                        file_n = lines[0].replace('DSCI552_Captions', 'DSCI552_Videos')
                        first_timestamp_match = re.search(r'First Timestamp: (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                        last_timestamp_match = re.search(r'Last Timestamp:\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])

                        if first_timestamp_match and last_timestamp_match:
                            first_timestamp = first_timestamp_match.group(1)
                            last_timestamp = last_timestamp_match.group(1)

                            col = columns[index % num_columns]
                            with col:
                                # Extracting the lecture date from the file name
                                date_match = re.search(r'Lecture Video (\d{2}_\d{2}_\d{4})', file_n)
                                lecture_date = date_match.group(1).replace('_', '/')

                                col.write(f"Lecture on {lecture_date}")
                                col.write(f"From {first_timestamp.replace(',', '.')} to {last_timestamp.replace(',', '.')}")
                                
                                start = parse_timestamp(first_timestamp)
                                col.video(file_n, format="video/mp4", start_time=start)
                else:
                    st.info("No relevant video information found.")

            with st.expander("Textbook Information", expanded=True):
                if docs_txt:
                    num_columns = min(len(docs_txt), 2)
                    columns = st.columns(num_columns)

                    for index, d in enumerate(docs_txt):
                        lines = d.page_content.split('\n')
                        file_n = "DSCI552Textbook.pdf"
                        page_num = int(lines[0])

                        col = columns[index % num_columns]
                        with col:
                            col.write(f"Page: {page_num}")
                            open_pdf_page(file_n, page_num)
                else:
                    st.info("No relevant textbook information found.")

            with st.expander("Syllabus Information", expanded=True):
                if docs_syl:
                    for d in docs_syl:
                        st.write(d.page_content)
                else:
                    st.info("No relevant syllabus information found.")

            end_time = time.time()
            processing_time = end_time - start_time
            st.caption(f"GATA response time: {processing_time:.2f} seconds")

# Automatic GATA responses to user questions to the whole class
def gata_response(post_id, query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    v_store_pdf= FAISS.load_local("faiss_slides", embeddings,allow_dangerous_deserialization=True)
    v_store_vid = FAISS.load_local("faiss_vids", embeddings,allow_dangerous_deserialization=True)
    v_store_syl= FAISS.load_local("faiss_syllabus", embeddings,allow_dangerous_deserialization=True)
    v_store_txt = FAISS.load_local("faiss_textbook", embeddings,allow_dangerous_deserialization=True)

    retriever_syl = v_store_syl.as_retriever( search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.05})
    retriever_txt = v_store_txt.as_retriever( search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.05})
    retriever_pdf = v_store_pdf.as_retriever( search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2})
    retriever_vid = v_store_vid.as_retriever( search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1})

    docs_pdf = retriever_pdf.get_relevant_documents(query)
    docs_vid = retriever_vid.get_relevant_documents(query)
    docs_syl = retriever_syl.get_relevant_documents(query)
    docs_txt = retriever_txt.get_relevant_documents(query)

    syllabus_context = get_context(docs_syl)
    textbook_context = get_context(docs_txt)
    slides_context = get_context(docs_pdf)
    video_context = get_context(docs_vid)

    prompt = f"Use only the following pieces of context to answer the following query.\n--- query start\n{query}\n--- query end\nIf there is not enough context to answer the query please respond: Answer not found\nUse only the following context:\n--- Syllabus start\n{syllabus_context}\n--- Syllabus end\n--- Textbook start\n{textbook_context}\n--- Textbook end\n--- Slides start\n{slides_context}\n--- Slides end\n--- Lecture video start\n{video_context}\n--- Lecture video end\nPlease output in clear and concise sentences if you have found an answer to the query. If not respond: Answer not found\n"    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an electronic TA. Your job is to answer questions using only context given from the syllabus, textbook, slides, and videos. If there is none then you should output that you do not know the answer."},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion.choices[0].message.content
    
    response_data = {
        'content': response,
        'timestamp': datetime.now().isoformat()
    }

    db.posts.update_one(
        {"post_id": post_id},
        {"$set": {"gata_response": response_data}}
    )

# Get the start and end of the current week
def get_week_range(date):
    start_of_week = date - timedelta(days=date.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week

# Categorize posts by week
def categorize_posts(posts):
    today = datetime.now().date()
    this_week_start, this_week_end = get_week_range(today)
    categorized_posts = {}

    # Create dictionary for weeks
    for post in posts:
        post_date = parse(post['datetime']).date()
        start_of_post_week, end_of_post_week = get_week_range(post_date)

        week_key = f"Week {start_of_post_week.strftime('%m/%d')} - {end_of_post_week.strftime('%m/%d')}"
        if week_key not in categorized_posts:
            categorized_posts[week_key] = []
        categorized_posts[week_key].append(post)

    # Sort posts within each week by datetime in descending order
    for week, week_posts in categorized_posts.items():
        week_posts.sort(key=lambda post: parse(post['datetime']), reverse=True)

    # Sort weeks so the current week is on top, handle "Week " prefix in sorting key
    sorted_week_keys = sorted(categorized_posts.keys(), key=lambda x: datetime.strptime(x[5:10], "%m/%d"), reverse=True)
    ordered_posts = {week: categorized_posts[week] for week in sorted_week_keys}

    return ordered_posts

# Fetch data from MongoDB and store the _id
posts_data = get_data()
categorized_posts = categorize_posts(posts_data)

# Convert MongoDB datetime objects to Python datetime for display
def convert_datetime(posts):
    for post in posts:
        if 'datetime' in post:
            post['datetime'] = parse(post['datetime']).strftime('%Y-%m-%d %H:%M:%S')
        if 'responses' in post:
            for response in post['responses']:
                if 'datetime' in response:
                    response['datetime'] = parse(response['datetime']).strftime('%Y-%m-%d %H:%M:%S')
    return posts

posts_data = convert_datetime(posts_data)

# Search posts with search bar
def search_posts(categorized_posts, query):
    query_words = query.lower().strip().split()
    seen_post_ids = set()  # Keep track of posts that have already been added to the results
    filtered_categorized_posts = {}

    for category, posts in categorized_posts.items():
        filtered_category_posts = []
        for post in posts:
            if post['post_id'] in seen_post_ids:
                continue  # Skip this post if it has already been processed

            post_content = post['content'].lower()
            if any(word in post_content for word in query_words):
                filtered_category_posts.append(post)
                seen_post_ids.add(post['post_id'])  # Mark this post as processed
            else:
                for response in post.get('responses', []):
                    response_content = response['content'].lower()
                    if any(word in response_content for word in query_words):
                        filtered_category_posts.append(post)
                        seen_post_ids.add(post['post_id'])  # Mark this post as processed
                        break

        if filtered_category_posts:
            filtered_categorized_posts[category] = filtered_category_posts

    return filtered_categorized_posts

# Filtering posts with folder specific tags
def filter_posts_by_selected_tag(posts, selected_tag):
    if 'All Posts' in selected_tag or not selected_tag:
        return posts
    else:
        return [post for post in posts if set(selected_tag) & set(post.get('tags', []))]

# Creating new posts
def new_post(subject, content, tags, post_to, post_type, display_name):
    next_post_number = db.posts.count_documents({}) + 1

    new_post = {
        "user_id": user['user_id'],
        "display_name": display_name,
        "post_id": next_post_number,
        "subject": subject,
        "datetime": datetime.now().isoformat(),
        "content": content,
        "instructor": user['instructor'],
        "pinned": False,
        "drafts": False,
        "post": post_type,
        "tags": tags,
        "visibility": "everyone" if post_to == 'Entire Class' else "private",
        "responses": []
    }
    insert_result = db.posts.insert_one(new_post)
    if insert_result.inserted_id:
        with st.spinner('Please wait, your post is being submitted...'):
            if post_type.lower() == 'question' and not user['instructor']:
                gata_response(next_post_number, content)
            st.session_state['update_sidebar'] = True
        st.success("Post submitted successfully.")
    else:
        st.error("There was an issue submitting the post.")
    time.sleep(2)
    st.rerun()

# Function to add a response to a post
def add_followup_to_post(followup_content, post_id, display_name, target_response_id=None):
    if not followup_content.strip():
        st.error("Follow-up content cannot be empty.")
        return

    post = db.posts.find_one({"post_id": int(post_id)})
    if not post:
        st.error("Post not found.")
        return

    if target_response_id:
        # Adding a follow-up to an existing response
        paths = target_response_id.split('-')
        new_response_id = target_response_id
        nested_path = "responses"
        update_path = nested_path

        # Traverse the nested structure to the correct response to append the new follow-up
        for idx in range(1, len(paths)):
            index = int(paths[idx]) - 1
            if idx < len(paths) - 1:
                nested_path += f".{index}.responses"
                update_path = nested_path
            else:
                # Get the count of responses at this level to create a new response ID
                nested_doc = db.posts.find_one({"post_id": int(post_id)}, {nested_path: 1})
                count = len(nested_doc[nested_path][index]['responses']) if nested_doc else 0
                new_response_id = f"{target_response_id}-{count + 1}"
                update_path += f".{index}.responses"
    else:
        # Adding a new response to the main post
        count = len(post.get('responses', []))
        new_response_id = f"{post_id}-{count + 1}"
        update_path = "responses"

    new_followup = {
        "user_id": user['user_id'],
        "display_name": display_name,
        "response_id": new_response_id,
        "content": followup_content,
        "datetime": datetime.now().isoformat(),
        "instructor": user['instructor'],
        "responses": []
    }

    update_result = db.posts.update_one(
        {"_id": post['_id']},
        {"$push": {update_path: new_followup}}
    )

    if update_result.modified_count == 1:
        st.success("Follow-up added successfully.")
    else:
        st.error("Failed to add the follow-up. Please try again.")

# Displaying posts in the main screen from the sidebar
def display_post_details(post_or_response_id, level=0, path=[]):
    def find_post_and_response(posts, response_id):
        for post in posts:
            if post.get('post_id') == response_id or post.get('response_id') == response_id:
                return post, None
            for response in post.get('responses', []):
                if response.get('response_id') == response_id:
                    return post, response
                found_post, found_response = find_post_and_response([response], response_id)
                if found_response:
                    return found_post, found_response
        return None, None

    # Display the post or response
    if '-' in str(post_or_response_id):
        parent_post, post = find_post_and_response(get_data(), post_or_response_id)
        if not parent_post or not post:
            st.error(f"Parent post or response with ID {post_or_response_id} not found.")
            return
        display_id = post_or_response_id
    else:
        post_id = int(post_or_response_id)
        post = db.posts.find_one({"post_id": post_id})
        if not post:
            st.error(f"Post with ID {post_id} not found.")
            return
        parent_post = post
        display_id = post_id

    with st.container():
        post_id_str = f"{'Response ID' if '-' in str(display_id) else 'Post ID'}: {display_id}"
        post_type_icon = "https://img.icons8.com/fluency/48/question-mark--v1.png" if post.get('post') == 'Question' else "https://img.icons8.com/fluency/48/note--v1.png"
        user_icon = "https://img.icons8.com/fluency/48/teacher.png" if post.get('instructor') else "https://img.icons8.com/fluency/48/students.png"

        post_author = post.get('display_name')
        post_date = parse(post['datetime']).strftime('%b %-d, %Y at %H:%M:%S')
        gata_response_data = post.get('gata_response')
        color = "#ee011e" if post.get('instructor') else "#05abe9"

        tag_html = ''
        if post.get('tags'):
            tag_html = ''.join(f"<span class='tag'>{tag}</span>" for tag in post['tags'])

        left_padding = 40 * level

        is_instructor_response = post.get('instructor', False)
        border_color = "#ee011e" if is_instructor_response else "transparent"
        container_class = f"response-container-{'instructor' if is_instructor_response else 'student'}-{level}"
        # Main post container
        if level == 0:
            st.markdown(f"""
            <style>
                .details-container {{
                    background-color: #ffffff;
                    border-left: 10px solid {color};
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                    margin-bottom: 20px;
                }}
                .post-id-display {{
                    font-size: 12px;
                    color: #888;
                    margin-bottom: -15px;
                }}
                .details-title {{
                    font-weight: 700;
                    font-size: 24px;
                    color: {color};
                    margin-bottom: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }}
                .details-content {{
                    font-size: 18px;
                    color: #4a4a4a;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    hyphens: auto;
                }}
                .tag {{
                    display: inline-block;
                    padding: 5px 10px;
                    margin-right: 5px;
                    background-color: #e1ecf4;
                    border-radius: 10px;
                    font-size: 12px;
                    color: {color};
                    margin-bottom: 5px; /* Added space below tags */
                    margin-top: 15px; /* Added space below tags */
                }}
                .details-footer {{
                    font-size: 14px;
                    color: #888;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .ta-response-container {{
                    background-color: #ffffff;
                    border-left: 10px solid #323232;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                    margin-bottom: 20px;
                }}
            </style>
            <div class='details-container'>
                <div class='details-title'>
                    <span class='details-title-text'>{post['subject']}</span>
                    <img src="{post_type_icon}" width="40" />
                </div>
                <div class='details-content'>{post['content']}</div>
                <div class='tag-container'>{tag_html}</div> <!-- Container for the tags -->
                <div class='details-footer'>
                    {post_date} | {post_author}
                    <img src="{user_icon}" width="40" style="float:right;" />
                </div>
                <div class='post-id-display'>{post_id_str}</div>
            </div>
            """, unsafe_allow_html=True)
            if (gata_response_data and not post.get('instructor') and 
                post.get('post') == "Question" and 
                gata_response_data.get('content') != "Answer not found"):

                post_datetime = parse(post['datetime'])
                gata_response_time = parse(gata_response_data['timestamp'])
                response_duration = (gata_response_time - post_datetime).total_seconds()

                st.markdown(f"""
                <style>
                    .ta-response-container {{
                        background-color: #ffffff;
                        border-left: 10px solid #323232;
                        padding: 20px;
                        border-radius: 5px;
                        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                        margin-bottom: 20px;
                    }}
                    .ta-response-title {{
                        font-weight: 700;
                        font-size: 24px;
                        color: #323232;
                        margin-bottom: 10px;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                    }}
                    .ta-response-content {{
                        font-size: 18px;
                        color: #4a4a4a;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                        hyphens: auto;
                        margin-bottom: 5px;
                    }}
                    .ta-response-footer {{
                        font-size: 14px;
                        font-style: italic;
                        color: #888;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }}
                </style>
                <div class='ta-response-container'>
                    <div class='ta-response-title'>
                        <span class='ta-response-title-text'>GATA (Automated Response)</span>
                        <img class="header-icon" src="data:image/png;base64,{logo}" width="50"/>
                    </div>
                    <div class='ta-response-content'>{gata_response_data['content']}</div>
                    <div class='ta-response-footer'>
                        <span>GATA automated response time: {response_duration:.2f} seconds</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <style>
                .{container_class} {{
                    background-color: #ffffff;
                    padding: 10px 20px;
                    border-radius: 5px;
                    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                    margin-left: {left_padding}px;
                    margin-bottom: 10px;
                    border-left: 10px solid {border_color};
                }}
                .response-content {{
                    font-size: 16px;
                    color: #4a4a4a;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    hyphens: auto;
                }}
                .response-footer {{
                    font-size: 14px;
                    color: #888;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
            </style>
            <div class='{container_class}'>
                <div class='response-content'>{post['content']}</div>
                <div class='response-footer'>
                    {post_date} | {post_author}
                    <img src="{user_icon}" width="30" style="float:right;" />
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Place the follow-up form at the bottom of the main post only
    if 'responses' in post:
        for idx, response in enumerate(post['responses']):
            if 'response_id' in response:
                display_post_details(response['response_id'], level + 1, path + [idx])
            else:
                st.warning("Response post ID not found.")

    # Place the follow-up form at the bottom of the main post and first-level responses only
    if level == 0:
        st.markdown("###")  # Spacing
        with st.container(border=True):
            main_post_details = f"Initial Post ({parse(parent_post['datetime']).strftime('%b %-d, %Y at %H:%M:%S')} | {parent_post.get('display_name', 'Unknown')})"
            response_details = [
                f"Response {i+1} ({parse(response['datetime']).strftime('%b %-d, %Y at %H:%M:%S')} | {response.get('display_name', 'Unknown')})"
                for i, response in enumerate(parent_post.get('responses', []))
            ]
            followup_target = [main_post_details] + response_details
            selected_target = st.selectbox("Respond to:", followup_target, key=f"target_{display_id}")
            followup_content = st.text_area("Write your follow-up here:", key=f"followup_{display_id}")

            if 'user' in st.session_state:
                logged_in_user_name = f"{st.session_state.user['first_name']} {st.session_state.user['last_name']}"
                if user['instructor']:
                    display_name = logged_in_user_name
                else:
                    display_name = st.selectbox('Show my name as:', [logged_in_user_name, 'Anonymous'])

            if st.button("Submit Follow-up", key=f"submit_{display_id}"):
                if followup_content:
                    followup_index = followup_target.index(selected_target) - 1
                    if followup_index < 0:
                        # Follow-up to the main post
                        add_followup_to_post(followup_content, parent_post['post_id'], display_name)
                    else:
                        # Follow-up to a direct response, construct the correct response_id
                        response_id = f"{parent_post['post_id']}-{followup_index + 1}"
                        add_followup_to_post(followup_content, parent_post['post_id'], display_name, response_id)
                    time.sleep(2)   
                    st.rerun()

# List of instructors
def get_instructors():
    instructors = list(db.users.find({"instructor": True}, {"_id": 0, "first_name": 1, "last_name": 1, "email": 1}).sort("user_id", pymongo.ASCENDING))
    return pd.DataFrame(instructors)

# List of students
def get_students():
    students = db.users.aggregate([
        {"$match": {"instructor": False}},
        {"$lookup": {
            "from": "posts", 
            "localField": "user_id", 
            "foreignField": "user_id", 
            "as": "posts"
        }},
        {"$project": {
            "_id": 0,  # Exclude the _id field
            "first_name": 1, 
            "last_name": 1, 
            "email": 1, 
            "number_of_posts": {"$size": "$posts"}
        }},
        {"$sort": {"number_of_posts": -1}}
    ])
    return pd.DataFrame(list(students))

# Extracting text from PDF files
def read_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        #st.write(pdf_path)
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        text = ''
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += pdf_path + '\n' + str(page_num+1) + '\n' + page.extract_text() + '###'
    return text

# Extracting timestamps from video captions
def extract_timestamps(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    chunks = [lines[i:i+4] for i in range(0, len(lines), 4)]
    grouped_chunks = [chunks[i:i+15] for i in range(0, len(chunks), 15)]
    final_text = ''
    for group in grouped_chunks:
        first_timestamp = group[0][1].split(' --> ')[0]
        last_timestamp = group[-1][1].split(' --> ')[1]
        text = ''.join(chunk[2] for chunk in group)
        final_text += '\n' + filename.replace('.txt','') + '\n' + "First Timestamp: " + first_timestamp + " Last Timestamp:" + last_timestamp + '\n' + "Text:" + text.strip() + '\n' +"###"
        
    return final_text

# Getting the context from the relevant documents
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="###", chunk_size=200, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks
# Getting special context chunks from the syllabus
def get_text_chunks_syl(text):
    text_splitter = CharacterTextSplitter(separator = '\n', chunk_size=400, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Adding to the respective vector store
def add_vstore(file_path, doc_type):
    """Adds documents to the vector store based on the document type."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if doc_type == 'vids':
        transcript = extract_timestamps(file_path)
        chunks = get_text_chunks(transcript)
    else:
        text = read_pdf(file_path)
        if doc_type == 'syllabus':
            chunks = get_text_chunks(text, separator='\n', chunk_size=400, chunk_overlap=100)
        else:
            chunks = get_text_chunks(text)

    try:
        v_store = FAISS.load_local("faiss_" + doc_type, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        v_store = FAISS(embeddings)

    v_store.add_texts(chunks)
    v_store.save_local("faiss_" + doc_type)

# Displaying instructor tools
def display_instructor_tools():
    st.subheader("Configure GATA")
    st.markdown("GATA currently supports the following data types:")
    st.markdown("* **Slides**: Visual content from lectures.")
    st.markdown("* **Videos**: Recorded video lectures. (Captions required)")
    st.markdown("* **Textbook**: Content from the assigned textbook.")
    st.markdown("* **Syllabus**: Course outline and policies.")

    st.markdown("###")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload your document")
        doc_type = st.selectbox("Document Type", ["slides", "vids", "textbook", "syllabus"])
        if st.button("Process and Add to GATA"):
            if uploaded_file is not None and doc_type:
                with st.spinner('Processing...'):
                    temp_dir = "temp"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)  # Create 'temp' directory if it does not exist
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    add_vstore(temp_file_path, doc_type)
                    st.success("Document processed and added to GATA.")

    st.markdown("###")
    st.subheader("GATA Metrics")
    total_questions, answered_questions, average_response_time = calculate_metrics(db)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Questions Submitted", total_questions)
        col2.metric("Questions Answered by GATA", answered_questions)
        col3.metric("Average Response Time (sec)", f"{average_response_time:.2f}")

    st.subheader("Course Registration Key")
    with st.container(border=True):
        new_key = st.text_input("Set New Registration Key:", value=st.session_state['registration_key'])
        if st.button("Update Key"):
            st.session_state['registration_key'] = new_key
            st.success("Registration key updated successfully.")

    st.subheader("Instructor Roster")
    instructors_df = get_instructors()
    st.dataframe(instructors_df)

    st.subheader("Student Roster")
    students_df = get_students()
    st.dataframe(students_df)

# Displaying student tools
def display_student_tools():
    st.markdown("###")
    st.subheader("GATA Metrics")
    total_questions, answered_questions, average_response_time = calculate_metrics(db)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Questions Submitted", total_questions)
        col2.metric("Questions Answered by GATA", answered_questions)
        col3.metric("Average Response Time (sec)", f"{average_response_time:.2f}")

    st.subheader("Student Roster")
    students_df = get_students()
    st.dataframe(students_df)

# Clearing search bar session state
def clear_search():
    st.session_state["search_query"] = ""

# Main streamlit application logic

if not st.session_state.logged_in:
    display_auth_forms()
else:
    user = st.session_state.user
    user_name = f"{user['first_name']} {user['last_name']}"

    with st.sidebar:
        if st.session_state.get('update_sidebar', False):
            posts_data = get_data()  # Refetch the posts data to include the new post
            categorized_posts = categorize_posts(posts_data)
            st.session_state['update_sidebar'] = False  # Reset the flag after updating
        
        button_col1, button_col2, button_col3 = st.columns([1, 1, 1], gap="small")
        with button_col1:
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="https://img.icons8.com/external-kiranshastry-lineal-color-kiranshastry/64/external-user-interface-kiranshastry-lineal-color-kiranshastry.png", width="60">
                <br>
                <span> {user_name} </span>
            </div>
            """, unsafe_allow_html=True)

        with button_col2:
            if user['instructor']:
                if st.button("Instructor Tools :gear:"):
                    st.session_state.display_page = 'TOOLS'
            else:
                if st.button("Student Tools :books:"):
                    st.session_state.display_page = 'TOOLS'


            if st.button("Logout :wave:"):
                st.session_state.logged_in = None
                st.rerun()

        with button_col3:
            if st.button('Chat with GATA :smiley_cat:'):
                st.session_state.display_page = 'GATA_CHAT'

            if st.button('New Post :pencil:'):
                st.session_state.display_page = 'NEW_POST'
        
        st.divider()

        col1, col2, col3 = st.columns([3, 6, 2], gap="small")
        with col1:
            class_selected = st.selectbox(
                "Select Class",
                ["DSCI 552", "DSCI 560"],
                key="class_select",
                label_visibility="collapsed")
        with col2:
            search_query = st.text_input('Search', placeholder="Search for a post...", label_visibility="collapsed", key="search_query").strip()
            if search_query:
                categorized_posts = search_posts(categorized_posts, search_query)
        with col3:
            if st.button("Reset", key="reset_button", on_click=clear_search):
                categorized_posts = categorize_posts(posts_data)
        
        selected_tag = st.multiselect('Filter Posts', filter, default=['All Posts'])

        for category, posts in categorized_posts.items():
            filtered_posts = filter_posts_by_selected_tag(posts, selected_tag)
            if filtered_posts:
                with st.expander(category, expanded=True):
                    for post in filtered_posts:
                        col1, col2 = st.columns([0.85, 0.15])
                        with col1:
                            gata_icon = f'<img src="data:image/png;base64,{logo}" width="25" height="25" style="margin-left: 5px;" />' if 'gata_response' in post and post['gata_response'].get('content', '') != "Answer not found" else ''
                            instructor_response = any(resp.get('instructor', False) for resp in post.get('responses', [])) and not post.get('instructor', False)
                            instructor_icon = '<img src="https://img.icons8.com/fluency/48/teacher.png" width="25" height="25" style="margin-left: 5px;" />' if instructor_response else ''
                            
                            instructor_tag = '<img src="https://img.icons8.com/fluency/48/teacher.png" width="30" /> ' if post.get('instructor', False) else ''
                            post_type = post.get('post', 'Note')
                            date_str = parse(post['datetime']).strftime('%-m/%-d/%y') if 'datetime' in post and post['datetime'] else ''
                            post_icon = "https://img.icons8.com/fluency/48/question-mark--v1.png" if post_type == 'Question' else "https://img.icons8.com/fluency/48/note--v1.png"

                            post_author = post.get('display_name')

                            border_color_class = "post-container-instructor" if post.get('instructor', False) else "post-container-student"
                            title_color_class = "post-title-instructor" if post.get('instructor', False) else "post-title-student"

                            post_style = f"""
                            <style>
                                .{border_color_class} {{
                                    background-color: #ffffff;
                                    border-left: 10px solid {("#ee011e" if post.get('instructor', False) else "#05abe9")};
                                    padding: 10px 15px;
                                    border-radius: 5px;
                                    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                                    transition: 0.3s;
                                    margin-bottom: 10px;
                                    line-height: 1.6;
                                }}
                                .{border_color_class}:hover {{
                                    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
                                }}
                                .{title_color_class} {{
                                    font-weight: 700;
                                    font-size: 14px;
                                    color: {("#ee011e" if post.get('instructor', False) else "#05abe9")};
                                    display: flex;
                                    justify-content: space-between;
                                    align-items: center;

                                }}
                                .post-title-text {{
                                    display: -webkit-box;
                                    -webkit-line-clamp: 1;
                                    -webkit-box-orient: vertical;
                                    overflow: hidden;
                                    text-overflow: ellipsis;
                                }}
                                .post-content {{
                                    font-size: 14px;
                                    color: #4a4a4a;
                                    display: -webkit-box;
                                    -webkit-line-clamp: 2;
                                    -webkit-box-orient: vertical;
                                    overflow: hidden;
                                    text-overflow: ellipsis;
                                }}
                                .post-footer {{
                                    display: flex;
                                    justify-content: space-between;
                                    font-size: 11px;
                                    color: #888;
                                }}
                            </style>
                            <div class='{border_color_class}'>
                                <div class='{title_color_class}'>
                                    <div class='post-title-text'>{instructor_tag}{post['subject']} </div>
                                    <img src="{post_icon}" width="30" />
                                </div>
                                <div class='post-content'>{post['content']}</div>
                                <div class='post-footer'>
                                    <span>{post_type} from {post_author} {gata_icon} {instructor_icon}</span>
                                    <span>{date_str}</span>
                                </div>
                            </div>
                            """
                            st.markdown(post_style, unsafe_allow_html=True)
                        with col2:
                            # Button to show the post in the main area
                            btn_key = f"btn_{post.get('post_id')}"
                            if st.button("📂", key=btn_key):
                                st.session_state.display_page = 'NONE'
                                st.session_state.active_post = post
    
    if st.session_state.display_page == 'NEW_POST':
        if user['instructor']:
            post_type = 'Note'
        else:
            post_type = st.radio("Post Type:", ('Question', 'Note'), horizontal=True)
        post_to = st.radio("Post To:", ('Entire Class', 'Instructor(s)'), horizontal=True)
        tags = st.multiselect("Select Folder(s):", folders)
        subject = st.text_input('Summary:', max_chars=100, placeholder="Enter a one line summary, 100 characters or less")
        content = st.text_area('Enter your post details:', height=200)
        if user['instructor']:
            display_name = user_name
        else:
            display_name = st.selectbox('Show my name as:', [f'{user_name}', 'Anonymous'])
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            if st.button(f'Post My {post_type} to {class_selected}!'):
                if subject and content and tags:
                    new_post(subject, content, tags, post_to, post_type, display_name)
                else:
                    st.error("Please fill in all fields and select at least one folder.")

    elif st.session_state.display_page == 'TOOLS':
        if user['instructor']:
            display_instructor_tools()
        else:
            display_student_tools()

    elif st.session_state.display_page == 'GATA_CHAT':
        description = f"GATA is the Generative AI Teaching Assistant that can answer your questions about {class_selected}!"
        header_style = f"""
        <style>
            .header-container {{
                display: flex;
                align-items: center;
                justify-content: center;
                color: #2b2e4a;
                text-align: center;
            }}
            .header-icon {{
                margin-right: 0px;
            }}
            .subhead-text {{
                font-size: 18px;
                color: #2b2e4a;
                margin-top: 0px;
                text-align: center;
                font-weight: 500;
            }}
        </style>
        <div class="header-container">
            <img class="header-icon" src="data:image/png;base64,{logo}" width="75"/>
            <h1>Chat with GATA</h1>
        </div>
        <div class="subhead-text">
            {description}
        </div>
        """
        st.markdown(header_style, unsafe_allow_html=True)
        st.markdown("###")

        query = st.text_area("Enter your query:" )
        col1, col2 = st.columns([4, 1])
        if col2.button("Submit"):
            if query:
                ta_chat(query)
            else:
                st.warning("Please enter a query before submitting.")

    else:
        if st.session_state.active_post:
            with st.spinner('Post loading...'):
                display_post_details(st.session_state.active_post['post_id'])

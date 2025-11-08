import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="AI Skill Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .match-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ecc71;
    }
    .job-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    .skill-match {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

class SkillMatcher:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_skills(self, text):
        """Extract skills from text using keyword matching"""
        common_skills = {
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular',
            'vue', 'node', 'express', 'django', 'flask', 'fastapi', 'mongodb',
            'mysql', 'postgresql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'git', 'jenkins', 'machine learning', 'deep learning', 'nlp',
            'computer vision', 'data analysis', 'pandas', 'numpy', 'tensorflow',
            'pytorch', 'scikit-learn', 'tableau', 'powerbi', 'excel'
        }
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def calculate_similarity(self, job_descriptions, resume_text):
        """Calculate similarity between job descriptions and resume"""
        # Preprocess all texts
        processed_jobs = [self.preprocess_text(job) for job in job_descriptions]
        processed_resume = self.preprocess_text(resume_text)
        
        # Combine for vectorization
        all_texts = processed_jobs + [processed_resume]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity
        job_vectors = tfidf_matrix[:-1]
        resume_vector = tfidf_matrix[-1]
        
        similarities = cosine_similarity(resume_vector, job_vectors)
        
        return similarities[0]
    
    def get_match_analysis(self, job_description, resume_text):
        """Get detailed match analysis"""
        job_skills = self.extract_skills(job_description)
        resume_skills = self.extract_skills(resume_text)
        
        matching_skills = set(job_skills) & set(resume_skills)
        missing_skills = set(job_skills) - set(resume_skills)
        
        skill_match_rate = len(matching_skills) / len(job_skills) if job_skills else 0
        
        return {
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'skill_match_rate': skill_match_rate,
            'total_job_skills': len(job_skills),
            'matching_skills_count': len(matching_skills)
        }

def load_sample_data():
    """Load sample job and resume data"""
    # Sample job data
    jobs_data = {
        'Job Title': [
            'Data Scientist',
            'Frontend Developer',
            'Backend Developer',
            'ML Engineer',
            'Full Stack Developer',
            'DevOps Engineer'
        ],
        'Company': [
            'Tech Corp', 'Web Solutions', 'API Masters', 
            'AI Innovations', 'Digital Creations', 'Cloud Experts'
        ],
        'Description': [
            'We are looking for a Data Scientist with strong Python skills, experience in machine learning, pandas, numpy, and SQL. Knowledge of TensorFlow or PyTorch is a plus.',
            'Seeking Frontend Developer proficient in JavaScript, React, HTML, CSS. Experience with modern frameworks and responsive design required.',
            'Backend Developer needed with expertise in Python, Django, Flask, SQL, APIs. Knowledge of microservices architecture preferred.',
            'Machine Learning Engineer required with deep learning, NLP, computer vision experience. Proficiency in PyTorch, TensorFlow, and Python essential.',
            'Full Stack Developer with React, Node.js, Python, MongoDB, and AWS experience. Must have full project lifecycle experience.',
            'DevOps Engineer with Docker, Kubernetes, AWS, CI/CD, Jenkins, and infrastructure as code experience. Strong scripting skills required.'
        ],
        'Required Skills': [
            'python,machine learning,pandas,numpy,sql,tensorflow',
            'javascript,react,html,css,responsive design',
            'python,django,flask,sql,apis,microservices',
            'python,machine learning,deep learning,nlp,computer vision,pytorch',
            'react,node.js,python,mongodb,aws,full stack',
            'docker,kubernetes,aws,ci/cd,jenkins,scripting'
        ]
    }
    
    # Sample resumes
    resumes_data = {
        'Resume Text': [
            """Experienced Data Scientist with 5 years in Python, machine learning, pandas, numpy, SQL. 
            Strong background in TensorFlow and data analysis. Skilled in statistical modeling and data visualization.""",
            
            """Frontend Developer specializing in JavaScript, React, HTML5, CSS3. 
            Built responsive web applications and mobile-first designs. Experience with Redux and TypeScript.""",
            
            """Backend Developer with Django, Flask, Python, and SQL expertise. 
            Developed REST APIs and microservices. Knowledge of Docker and AWS services.""",
            
            """Machine Learning Engineer focused on deep learning and computer vision. 
            Proficient in PyTorch, TensorFlow, and Python. Experience with NLP projects."""
        ]
    }
    
    jobs_df = pd.DataFrame(jobs_data)
    resumes_df = pd.DataFrame(resumes_data)
    
    return jobs_df, resumes_df

def main():
    load_css()
    
    st.markdown('<h1 class="main-header">🔍 AI-Powered Skill Matcher</h1>', unsafe_allow_html=True)
    st.markdown("### Bridge the Gap Between Job Seekers and Roles")
    
    # Initialize session state
    if 'matcher' not in st.session_state:
        st.session_state.matcher = SkillMatcher()
    if 'jobs_df' not in st.session_state:
        st.session_state.jobs_df, st.session_state.resumes_df = load_sample_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["Skill Matching", "Job Search", "Resume Analysis", "About"])
    
    if app_mode == "Skill Matching":
        show_skill_matching()
    elif app_mode == "Job Search":
        show_job_search()
    elif app_mode == "Resume Analysis":
        show_resume_analysis()
    else:
        show_about()

def show_skill_matching():
    st.header("🎯 Skill Matching")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Your Resume")
        resume_input = st.text_area(
            "Paste your resume text here:",
            height=300,
            placeholder="""Enter your resume text here. Include your skills, experience, education, and projects.

Example:
Experienced software developer with 5 years in web development. 
Skills: Python, JavaScript, React, Node.js, SQL, MongoDB, AWS.
Education: Bachelor's in Computer Science.
Projects: Built e-commerce platform using MERN stack."""
        )
        
        # Or use sample resume
        if st.button("Use Sample Resume"):
            sample_resume = st.session_state.resumes_df.iloc[0]['Resume Text']
            st.session_state.sample_resume = sample_resume
    
    with col2:
        st.subheader("📊 Matching Results")
        
        if resume_input or 'sample_resume' in st.session_state:
            resume_text = resume_input if resume_input else st.session_state.sample_resume
            
            # Calculate similarities
            similarities = st.session_state.matcher.calculate_similarity(
                st.session_state.jobs_df['Description'].tolist(),
                resume_text
            )
            
            # Add similarity scores to jobs dataframe
            results_df = st.session_state.jobs_df.copy()
            results_df['Match Score'] = (similarities * 100).round(2)
            results_df = results_df.sort_values('Match Score', ascending=False)
            
            # Display results
            for idx, row in results_df.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="job-card">
                        <h3>{row['Job Title']} - {row['Company']}</h3>
                        <p class="match-score">Match Score: {row['Match Score']}%</p>
                        <p><strong>Description:</strong> {row['Description'][:150]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show detailed analysis
                    with st.expander("View Detailed Analysis"):
                        analysis = st.session_state.matcher.get_match_analysis(
                            row['Description'], resume_text
                        )
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.success("✅ Matching Skills")
                            for skill in analysis['matching_skills']:
                                st.markdown(f'<div class="skill-match">{skill}</div>', unsafe_allow_html=True)
                        
                        with col_b:
                            st.error("❌ Missing Skills")
                            for skill in analysis['missing_skills']:
                                st.markdown(f'<div class="skill-match" style="background-color: #f8d7da;">{skill}</div>', unsafe_allow_html=True)
                        
                        st.info(f"Skill Match Rate: {analysis['skill_match_rate']*100:.1f}%")
            
            # Visualization
            st.subheader("📈 Match Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=results_df, x='Match Score', y='Job Title', palette='viridis')
            ax.set_xlabel('Match Score (%)')
            ax.set_ylabel('Job Titles')
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.info("Please enter your resume text to see matching jobs.")

def show_job_search():
    st.header("🔍 Job Search")
    
    search_term = st.text_input("Search for jobs by title or skills:")
    
    filtered_jobs = st.session_state.jobs_df
    if search_term:
        filtered_jobs = filtered_jobs[
            filtered_jobs['Job Title'].str.contains(search_term, case=False) |
            filtered_jobs['Description'].str.contains(search_term, case=False)
        ]
    
    for idx, row in filtered_jobs.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="job-card">
                <h3>{row['Job Title']} - {row['Company']}</h3>
                <p><strong>Description:</strong> {row['Description']}</p>
                <p><strong>Required Skills:</strong> {row['Required Skills']}</p>
            </div>
            """, unsafe_allow_html=True)

def show_resume_analysis():
    st.header("📊 Resume Analysis")
    
    resume_text = st.text_area("Paste your resume for analysis:", height=300)
    
    if resume_text:
        matcher = st.session_state.matcher
        
        # Extract skills
        skills = matcher.extract_skills(resume_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛠️ Detected Skills")
            for skill in skills:
                st.markdown(f'<div class="skill-match">{skill}</div>', unsafe_allow_html=True)
            
            st.metric("Total Skills Detected", len(skills))
        
        with col2:
            st.subheader("📊 Skill Cloud")
            if skills:
                skill_text = ' '.join(skills)
                wordcloud = WordCloud(width=400, height=200, background_color='white').generate(skill_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        # Resume suggestions
        st.subheader("💡 Improvement Suggestions")
        if len(skills) < 10:
            st.warning("Consider adding more technical skills to your resume")
        else:
            st.success("Good variety of skills detected!")
        
        if 'experience' not in resume_text.lower():
            st.info("Add more details about your work experience")

def show_about():
    st.header("ℹ️ About AI Skill Matcher")
    
    st.markdown("""
    ### 🚀 How It Works
    
    This AI-powered skill matcher uses advanced Natural Language Processing (NLP) techniques to:
    
    1. **Text Processing**: Cleans and preprocesses resume and job description text
    2. **Skill Extraction**: Identifies key technical skills using keyword matching
    3. **Similarity Analysis**: Uses TF-IDF vectorization and cosine similarity
    4. **Match Scoring**: Calculates compatibility scores between resumes and jobs
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python
    - **ML Libraries**: Scikit-learn, NLTK
    - **NLP**: TF-IDF, Cosine Similarity
    - **Data Processing**: Pandas, NumPy
    
    ### 📈 Features
    
    - Real-time skill matching
    - Detailed match analysis
    - Skill gap identification
    - Interactive visualizations
    - Sample data for testing
    
    ### 🎯 Use Cases
    
    - Job seekers finding compatible roles
    - Recruiters identifying suitable candidates
    - Career guidance and skill development
    - Resume optimization and improvement
    """)

if __name__ == "__main__":
    main()
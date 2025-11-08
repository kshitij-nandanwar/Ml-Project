import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import ssl

# Fix SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

# Download NLTK data
download_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="AI Skill Matcher - India",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #FF9933 0%, #138808 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .skill-match {
        background-color: #d4edda;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .missing-skill {
        background-color: #f8d7da;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .indian-flag {
        color: #FF9933;
        font-size: 1.2em;
    }
    .stProgress > div > div > div > div {
        background-color: #138808;
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
            'pytorch', 'scikit-learn', 'tableau', 'powerbi', 'excel', 'typescript',
            'rest api', 'graphql', 'microservices', 'linux', 'bash', 'shell scripting',
            'agile', 'scrum', 'ci/cd', 'terraform', 'ansible', 'prometheus', 'grafana',
            'spring boot', 'hibernate', 'android', 'kotlin', 'swift', 'ios',
            'php', 'laravel', 'wordpress', 'shopify', 'mern stack', 'mean stack'
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

@st.cache_data
def load_sample_data():
    """Load sample job and resume data with Indian context"""
    # Sample job data for Indian market
    jobs_data = {
        'Job Title': [
            'Data Scientist',
            'Frontend Developer',
            'Backend Developer',
            'ML Engineer',
            'Full Stack Developer',
            'DevOps Engineer',
            'Android Developer',
            'Java Developer',
            'Python Developer',
            'React Native Developer'
        ],
        'Company': [
            'Tata Consultancy Services', 'Infosys', 'Wipro', 
            'HCL Technologies', 'Tech Mahindra', 'Accenture India',
            'Flipkart', 'Paytm', 'Zomato', 'Ola Cabs'
        ],
        'Description': [
            'Looking for Data Scientist with strong Python skills, machine learning experience, pandas, numpy, SQL. Knowledge of TensorFlow or PyTorch required. Experience with Indian market data preferred.',
            'Seeking Frontend Developer proficient in JavaScript, React, HTML, CSS. Experience with responsive design and modern frameworks. Knowledge of Indian UI/UX trends.',
            'Backend Developer needed with expertise in Java, Spring Boot, Microservices, SQL. Experience with high-traffic systems. Knowledge of Indian digital payment systems a plus.',
            'Machine Learning Engineer required with deep learning, NLP, computer vision experience. Proficiency in Python, PyTorch, TensorFlow. Experience with Indian languages NLP preferred.',
            'Full Stack Developer with React, Node.js, MongoDB, and AWS experience. Must have experience in startup environment. Knowledge of Indian e-commerce domain.',
            'DevOps Engineer with Docker, Kubernetes, AWS, CI/CD, Jenkins experience. Strong scripting skills. Experience with scalable infrastructure for Indian user base.',
            'Android Developer with Kotlin/Java experience. Knowledge of Android SDK, Material Design. Experience with apps for Indian users.',
            'Java Developer with Spring Framework, Hibernate, REST APIs. Experience with enterprise applications. Knowledge of Indian banking systems preferred.',
            'Python Developer with Django/Flask experience. Strong in algorithms and data structures. Experience with Indian startup ecosystem.',
            'React Native Developer for cross-platform mobile apps. Experience with Redux, TypeScript. Knowledge of Indian mobile market trends.'
        ],
        'Required Skills': [
            'python,machine learning,pandas,numpy,sql,tensorflow',
            'javascript,react,html,css,responsive design',
            'java,spring boot,microservices,sql,hibernate',
            'python,machine learning,deep learning,nlp,computer vision,pytorch',
            'react,node.js,mongodb,aws,full stack',
            'docker,kubernetes,aws,ci/cd,jenkins,scripting',
            'android,kotlin,java,mobile development',
            'java,spring framework,hibernate,rest apis',
            'python,django,flask,algorithms,data structures',
            'react native,javascript,redux,typescript'
        ],
        'Location': [
            'Bangalore', 'Hyderabad', 'Pune', 'Delhi NCR', 'Chennai',
            'Mumbai', 'Gurgaon', 'Noida', 'Kolkata', 'Ahmedabad'
        ],
        'Salary': [
            '₹12-18 LPA', '₹8-12 LPA', '₹10-15 LPA', '₹15-22 LPA', '₹9-14 LPA',
            '₹11-16 LPA', '₹10-14 LPA', '₹8-13 LPA', '₹7-11 LPA', '₹9-13 LPA'
        ],
        'Experience': [
            '3-6 years', '2-4 years', '3-5 years', '4-7 years', '2-5 years',
            '3-6 years', '2-4 years', '2-5 years', '1-4 years', '2-4 years'
        ]
    }
    
    # Sample resumes with Indian context
    resumes_data = {
        'Resume Text': [
            """Experienced Data Scientist with 4 years experience in Bengaluru. Strong in Python, machine learning, pandas, numpy, SQL. 
            Worked on Indian e-commerce data analysis. Skilled in TensorFlow and statistical modeling. Education: B.Tech from IIT Delhi.""",
            
            """Frontend Developer from Pune with 3 years experience. Specializing in JavaScript, React, HTML5, CSS3. 
            Built responsive web applications for Indian startups. Experience with Redux and modern frontend tools.""",
            
            """Backend Developer with Java expertise. 4 years experience in Hyderabad. Strong in Spring Boot, Microservices, Hibernate. 
            Worked on banking applications for Indian banks. Knowledge of SQL and system design.""",
            
            """Machine Learning Engineer focused on computer vision. 3 years experience in Delhi. 
            Proficient in PyTorch, TensorFlow, and Python. Experience with Indian language text recognition projects.""",
            
            """Full Stack Developer from Mumbai with MERN stack experience. Built e-commerce platforms for Indian market. 
            Strong in React, Node.js, MongoDB, and AWS deployment."""
        ],
        'Title': [
            'Senior Data Scientist',
            'Frontend Developer',
            'Java Backend Developer', 
            'ML Engineer',
            'Full Stack Developer'
        ]
    }
    
    jobs_df = pd.DataFrame(jobs_data)
    resumes_df = pd.DataFrame(resumes_data)
    
    return jobs_df, resumes_df

def show_skill_matching():
    st.header("🎯 Skill Matching for Indian Job Market")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Your Resume")
        resume_input = st.text_area(
            "Paste your resume text here:",
            height=300,
            placeholder="""Enter your resume text here. Include your skills, experience, education, and projects.

Example:
Software Developer with 3 years experience in Bangalore.
Skills: Java, Spring Boot, Microservices, SQL, AWS.
Education: B.Tech in Computer Science from VTU.
Projects: Built payment gateway integration for Indian market."""
        )
        
        # Sample resume selector
        st.subheader("🎲 Use Indian Sample Resume")
        sample_resumes = st.session_state.resumes_df['Resume Text'].tolist()
        sample_titles = st.session_state.resumes_df['Title'].tolist()
        
        sample_options = [f"{title}" for title in sample_titles]
        selected_sample = st.selectbox("Choose a sample resume:", ["Select..."] + sample_options)
        
        if selected_sample != "Select...":
            sample_index = sample_options.index(selected_sample)
            resume_input = sample_resumes[sample_index]
            st.text_area("Sample Resume Preview:", value=resume_input, height=150, key="sample_preview")
    
    with col2:
        st.subheader("📊 Matching Results")
        
        if resume_input:
            with st.spinner('Analyzing your skills for Indian job market...'):
                # Calculate similarities
                similarities = st.session_state.matcher.calculate_similarity(
                    st.session_state.jobs_df['Description'].tolist(),
                    resume_input
                )
                
                # Add similarity scores to jobs dataframe
                results_df = st.session_state.jobs_df.copy()
                results_df['Match Score'] = (similarities * 100).round(2)
                results_df = results_df.sort_values('Match Score', ascending=False)
                
                # Display progress
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                
            # Display results
            st.success(f"Found {len(results_df)} potential job matches in India!")
            
            for idx, row in results_df.iterrows():
                # Determine match color
                match_color = "#2ecc71" if row['Match Score'] > 70 else "#f39c12" if row['Match Score'] > 50 else "#e74c3c"
                
                with st.container():
                    st.markdown(f"""
                    <div class="job-card">
                        <h3>{row['Job Title']} - {row['Company']}</h3>
                        <p style="color: {match_color}; font-size: 1.2rem; font-weight: bold;">
                            Match Score: {row['Match Score']}%
                        </p>
                        <p><strong>📍 Location:</strong> {row['Location']} | <strong>💰 Salary:</strong> {row['Salary']}</p>
                        <p><strong>📅 Experience:</strong> {row['Experience']}</p>
                        <p><strong>Description:</strong> {row['Description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show detailed analysis
                    with st.expander("View Detailed Skill Analysis"):
                        analysis = st.session_state.matcher.get_match_analysis(
                            row['Description'], resume_input
                        )
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.success(f"✅ Matching Skills ({analysis['matching_skills_count']} found)")
                            if analysis['matching_skills']:
                                for skill in analysis['matching_skills']:
                                    st.markdown(f'<div class="skill-match">{skill}</div>', unsafe_allow_html=True)
                            else:
                                st.info("No matching skills found")
                        
                        with col_b:
                            st.error(f"❌ Missing Skills ({len(analysis['missing_skills'])} needed)")
                            if analysis['missing_skills']:
                                for skill in analysis['missing_skills']:
                                    st.markdown(f'<div class="missing-skill">{skill}</div>', unsafe_allow_html=True)
                            else:
                                st.success("All required skills matched!")
                        
                        st.info(f"**Skill Match Rate:** {analysis['skill_match_rate']*100:.1f}%")
            
            # Visualization
            st.subheader("📈 Match Score Distribution - Indian Companies")
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['#2ecc71' if x > 70 else '#f39c12' if x > 50 else '#e74c3c' for x in results_df['Match Score']]
            bars = ax.barh(results_df['Job Title'] + ' - ' + results_df['Company'], results_df['Match Score'], color=colors)
            ax.set_xlabel('Match Score (%)')
            ax.set_title('Job Match Scores for Indian Companies')
            ax.bar_label(bars, fmt='%.1f%%')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.info("👆 Please enter your resume text or select a sample resume to see matching jobs in India.")

def show_job_search():
    st.header("🔍 Job Search in India")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("Search jobs by title, skills, or company:")
    
    with col2:
        location_filter = st.selectbox("Filter by city:", ["All India"] + st.session_state.jobs_df['Location'].unique().tolist())
    
    with col3:
        experience_filter = st.selectbox("Experience level:", ["All Levels"] + st.session_state.jobs_df['Experience'].unique().tolist())
    
    filtered_jobs = st.session_state.jobs_df.copy()
    
    if search_term:
        filtered_jobs = filtered_jobs[
            filtered_jobs['Job Title'].str.contains(search_term, case=False) |
            filtered_jobs['Description'].str.contains(search_term, case=False) |
            filtered_jobs['Company'].str.contains(search_term, case=False) |
            filtered_jobs['Required Skills'].str.contains(search_term, case=False)
        ]
    
    if location_filter != "All India":
        filtered_jobs = filtered_jobs[filtered_jobs['Location'] == location_filter]
    
    if experience_filter != "All Levels":
        filtered_jobs = filtered_jobs[filtered_jobs['Experience'] == experience_filter]
    
    if len(filtered_jobs) == 0:
        st.warning("No jobs found matching your criteria. Try broadening your search.")
    else:
        st.success(f"Found {len(filtered_jobs)} jobs matching your criteria in India.")
        
        for idx, row in filtered_jobs.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="job-card">
                    <h3>{row['Job Title']} - {row['Company']}</h3>
                    <p><strong>📍 Location:</strong> {row['Location']} | <strong>💰 Salary:</strong> {row['Salary']}</p>
                    <p><strong>📅 Experience:</strong> {row['Experience']}</p>
                    <p><strong>Description:</strong> {row['Description']}</p>
                    <p><strong>Required Skills:</strong> {row['Required Skills']}</p>
                </div>
                """, unsafe_allow_html=True)

def show_resume_analysis():
    st.header("📊 Resume Analysis for Indian Market")
    
    resume_text = st.text_area("Paste your resume for analysis:", height=300,
                              placeholder="Paste your resume text here to analyze your skills and get improvement suggestions for Indian job market...")
    
    if resume_text:
        matcher = st.session_state.matcher
        
        with st.spinner('Analyzing your resume for Indian job market...'):
            # Extract skills
            skills = matcher.extract_skills(resume_text)
            
            # Skill categories relevant to Indian market
            skill_categories = {
                'Programming Languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'kotlin', 'swift'],
                'Web Development': ['html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask'],
                'Mobile Development': ['android', 'react native', 'flutter', 'ios', 'mobile development'],
                'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
                'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ci/cd'],
                'Data Science & ML': ['python', 'machine learning', 'deep learning', 'nlp', 'computer vision', 
                                    'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn'],
                'Java Ecosystem': ['java', 'spring boot', 'hibernate', 'microservices', 'rest apis'],
                'Tools & Other': ['git', 'linux', 'bash', 'agile', 'scrum', 'tableau', 'powerbi', 'excel']
            }
            
            categorized_skills = {}
            for category, category_skills in skill_categories.items():
                found_skills = [skill for skill in skills if skill in category_skills]
                if found_skills:
                    categorized_skills[category] = found_skills
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛠️ Detected Skills")
            st.metric("Total Skills Detected", len(skills))
            
            for category, category_skills in categorized_skills.items():
                with st.expander(f"{category} ({len(category_skills)} skills)"):
                    for skill in category_skills:
                        st.markdown(f'<div class="skill-match">{skill}</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Skill Distribution")
            if skills:
                # Create skill count by category
                category_counts = {category: len(skills_list) for category, skills_list in categorized_skills.items()}
                
                fig, ax = plt.subplots(figsize=(8, 6))
                if category_counts:
                    colors = ['#FF9933', '#138808', '#000080', '#FF69B4', '#8B4513', '#2E8B57', '#DC143C', '#696969']
                    ax.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', 
                          startangle=90, colors=colors[:len(category_counts)])
                    ax.axis('equal')
                    ax.set_title('Skill Distribution for Indian Market')
                    st.pyplot(fig)
            else:
                st.info("No skills detected. Make sure to include technical skills in your resume.")
        
        # Resume suggestions for Indian market
        st.subheader("💡 Improvement Suggestions for Indian Job Market")
        
        suggestion_count = 0
        
        if len(skills) < 5:
            st.warning("🔸 Consider adding more technical skills - Indian companies value diverse skill sets")
            suggestion_count += 1
        
        if 'java' not in skills and 'python' not in skills:
            st.warning("🔸 Java and Python are highly demanded in Indian IT market - consider learning them")
            suggestion_count += 1
        
        if 'experience' not in resume_text.lower() and 'work' not in resume_text.lower():
            st.warning("🔸 Add more details about your work experience - Indian recruiters value relevant experience")
            suggestion_count += 1
        
        if 'education' not in resume_text.lower() and 'degree' not in resume_text.lower():
            st.warning("🔸 Include your educational background - Indian companies often consider educational qualifications")
            suggestion_count += 1
        
        if 'project' not in resume_text.lower():
            st.warning("🔸 Add details about your projects - Indian startups especially value project experience")
            suggestion_count += 1
        
        # Check for Indian market specific skills
        indian_demanded_skills = ['java', 'spring boot', 'python', 'react', 'aws', 'sql']
        missing_demanded_skills = [skill for skill in indian_demanded_skills if skill not in skills]
        if missing_demanded_skills:
            st.info(f"🔸 High-demand skills in India: Consider adding {', '.join(missing_demanded_skills)}")
        
        if suggestion_count == 0:
            st.success("🎉 Your resume looks good for Indian job market! It includes key sections and in-demand technical skills.")

def show_about():
    st.header("ℹ️ About AI Skill Matcher - India Edition")
    
    st.markdown("""
    ### 🇮🇳 Made for Indian Job Market
    
    This AI-powered skill matcher is specifically designed for the Indian job market, helping bridge the gap between Indian job seekers and opportunities.
    
    ### 🚀 How It Works
    
    This AI-powered skill matcher uses advanced Natural Language Processing (NLP) techniques to:
    
    1. **Text Processing**: Cleans and preprocesses resume and job description text
    2. **Skill Extraction**: Identifies key technical skills using keyword matching
    3. **Similarity Analysis**: Uses TF-IDF vectorization and cosine similarity
    4. **Match Scoring**: Calculates compatibility scores between resumes and Indian jobs
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python
    - **ML Libraries**: Scikit-learn, NLTK
    - **NLP**: TF-IDF, Cosine Similarity
    - **Data Processing**: Pandas, NumPy
    
    ### 📈 Features for Indian Users
    
    - Real-time skill matching with Indian companies
    - Salary ranges in INR (LPA)
    - Indian city-based job locations
    - Experience level filtering
    - Skills in demand in Indian market
    
    ### 🎯 Use Cases for Indian Job Seekers
    
    - IT professionals finding opportunities in Indian companies
    - Freshers matching skills with entry-level positions
    - Experienced professionals exploring Indian job market
    - Career guidance and skill development for Indian tech industry
    
    ### 🔧 How to Use
    
    1. Go to **Skill Matching** tab
    2. Enter your resume text or use Indian sample resume
    3. View matching jobs with Indian companies and salaries
    4. Analyze skill gaps for Indian market demands
    5. Use insights to improve your resume for Indian recruiters
    """)

def main():
    load_css()
    
    st.markdown('<h1 class="main-header">🇮🇳 AI-Powered Skill Matcher - India</h1>', unsafe_allow_html=True)
    st.markdown("### Bridge the Gap Between Indian Job Seekers and Opportunities")
    
    # Initialize session state
    if 'matcher' not in st.session_state:
        st.session_state.matcher = SkillMatcher()
    if 'jobs_df' not in st.session_state:
        st.session_state.jobs_df, st.session_state.resumes_df = load_sample_data()
    
    # Sidebar
    st.sidebar.title("🇮🇳 Navigation")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["Skill Matching", "Job Search", "Resume Analysis", "About"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Quick Start for Indian Job Seekers:**
    1. Go to **Skill Matching**
    2. Enter your resume
    3. View matches with Indian companies
    4. Analyze skill gaps for Indian market
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.success("""
    **Popular Indian Skills:**
    - Java & Spring Boot
    - Python & Django/Flask
    - React & Node.js
    - AWS & DevOps
    - Data Science & ML
    """)
    
    if app_mode == "Skill Matching":
        show_skill_matching()
    elif app_mode == "Job Search":
        show_job_search()
    elif app_mode == "Resume Analysis":
        show_resume_analysis()
    else:
        show_about()

if __name__ == "__main__":
    main()
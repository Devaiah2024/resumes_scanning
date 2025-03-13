

import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber  # Extract text from PDF resumes

# Load NLP model (Download first if not installed: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Sample Job Descriptions Dataset
# Ensure each Job Title has a matching Job Description
job_titles = [
    "Web Developer", "Frontend Developer", "Backend Developer", "Full Stack Developer",
    "Software Engineer", "Mobile App Developer", "Game Developer", 
    "DevOps Engineer", "Site Reliability Engineer (SRE)", "Cloud Engineer",
    "Cybersecurity Analyst", "Penetration Tester", "Ethical Hacker",
    "Data Scientist", "Machine Learning Engineer", "Deep Learning Engineer", "AI Research Scientist",
    "NLP Engineer", "Big Data Engineer", "Blockchain Developer", "IoT Engineer",
    "Embedded Systems Engineer", "Database Administrator", "Network Administrator",
    "System Administrator", "IT Support Specialist", "IT Project Manager",
    "Business Analyst", "ERP Consultant", "IT Auditor", "Software Tester",
    "Quality Assurance Engineer", "Product Manager", "UI/UX Designer",
    "E-Commerce Manager", "SEO Specialist", "Digital Marketing Specialist",
    "Mechanical Engineer", "Electrical Engineer", "Civil Engineer",
    "Doctor", "Pharmacist", "Nurse",
    "Financial Analyst", "Investment Banker", "Accountant",
    "Marketing Manager", "Sales Executive", "HR Manager",
    "Lawyer", "Teacher", "Graphic Designer",
    "Hotel Manager", "Chef", "Tour Guide",
]

job_descriptions = [
    "Develop and maintain websites using HTML, CSS, JavaScript, and frameworks like React or Angular.",
    "Build interactive user interfaces using JavaScript frameworks and optimize frontend performance.",
    "Develop server-side logic, databases, and APIs using Node.js, Django, Flask, or .NET.",
    "Handle both frontend and backend development to create full-fledged web applications.",
    "Develop scalable software applications using Python, Java, C++, or C#.",
    "Build iOS and Android applications using Swift, Kotlin, or Flutter.",
    "Create interactive and immersive video games using Unity, Unreal Engine, and C#.",
    "Automate infrastructure deployment and CI/CD workflows using Docker, Kubernetes, and Jenkins.",
    "Optimize cloud architecture on AWS, Azure, and Google Cloud for scalability and security.",
    "Perform security audits, risk assessments, and ensure IT infrastructure security.",
    "Conduct penetration testing and identify vulnerabilities in security systems.",
    "Assess and enhance cybersecurity measures to protect businesses from threats.",
    "Develop predictive models, analyze data trends, and build AI solutions.",
    "Implement and optimize machine learning models using TensorFlow and PyTorch.",
    "Build deep learning applications for image, speech, and NLP processing.",
    "Conduct AI research and innovate in machine learning techniques.",
    "Develop NLP models for text analysis, chatbots, and language processing.",
    "Process and analyze big data using Hadoop, Spark, and NoSQL databases.",
    "Develop blockchain applications and smart contracts using Solidity.",
    "Design and develop IoT devices, integrating sensors and cloud platforms.",
    "Work on embedded software and firmware development for hardware devices.",
    "Manage and maintain large-scale databases for business applications.",
    "Maintain and troubleshoot enterprise networks and IT infrastructure.",
    "Manage IT systems, servers, and network configurations.",
    "Provide technical support for hardware, software, and network issues.",
    "Lead IT projects, ensure timely delivery, and align with business goals.",
    "Analyze business processes and optimize IT solutions for companies.",
    "Manage ERP implementations and business process automation.",
    "Conduct IT audits to ensure compliance with security and regulatory standards.",
    "Test software applications, identify bugs, and ensure software quality.",
    "Ensure software meets industry quality standards through rigorous testing.",
    "Manage product development lifecycles and oversee design processes.",
    "Design user-friendly interfaces and improve user experience (UX).",
    "Manage online retail businesses and optimize e-commerce platforms.",
    "Optimize website rankings through SEO techniques and keyword research.",
    "Plan and execute digital marketing campaigns and social media strategies.",
    "Design and analyze mechanical components and manufacturing processes.",
    "Develop electrical systems, circuits, and power management solutions.",
    "Design and oversee civil construction projects and infrastructure development.",
    "Diagnose and treat illnesses, conduct medical research, and provide healthcare guidance.",
    "Dispense medications, counsel patients on drug use, and monitor prescriptions.",
    "Provide patient care, administer treatments, and assist doctors.",
    "Analyze financial data, create reports, and guide investment decisions.",
    "Assess financial risks, monitor credit scores, and advise clients on investments.",
    "Manage company accounts, prepare financial statements, and ensure tax compliance.",
    "Plan and execute marketing strategies, advertising campaigns, and brand positioning.",
    "Drive sales growth, negotiate contracts, and manage client relationships.",
    "Recruit, hire, and train employees while ensuring company policies are followed.",
    "Advise businesses on legal issues, contracts, and regulatory compliance.",
    "Teach students, develop curriculums, and conduct academic research.",
    "Design branding materials, logos, and digital content using graphic tools.",
    "Manage hotel operations, guest services, and staff to ensure high customer satisfaction.",
    "Prepare gourmet dishes, manage kitchen staff, and maintain food quality.",
    "Assist customers with travel planning, ticket booking, and itinerary management."
]

# Creating DataFrame
# Ensure both lists have the same length
min_length = min(len(job_titles), len(job_descriptions))

# Trim lists to the shortest length
job_titles = job_titles[:min_length]
job_descriptions = job_descriptions[:min_length]

# Create DataFrame
jobs_df = pd.DataFrame({
    "Job_Title": job_titles,
    "Job_Description": job_descriptions
})

print("DataFrame created successfully!")



# Function to extract text from a PDF resume
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

# Function to match resume text with job roles
def match_resume_to_jobs(resume_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text] + jobs_df["Job_Description"].tolist())

    resume_vector = tfidf_matrix[0]  # Resume text
    job_vectors = tfidf_matrix[1:]   # Job descriptions

    similarity_scores = cosine_similarity(resume_vector, job_vectors)
    top_idx = similarity_scores.argmax()

    return jobs_df.iloc[top_idx]["Job_Title"]

# Streamlit UI
st.title("AI Resume Screening & Job Matching")

# Choose input method: Paste text or Upload PDF
option = st.radio("Choose input method:", ["Paste Resume Text", "Upload PDF Resume"])

resume_text = ""

if option == "Paste Resume Text":
    resume_text = st.text_area("Paste your resume here:")
elif option == "Upload PDF Resume":
    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Resume Text:", resume_text, height=200)

# Run job matching when the button is clicked
if st.button("Find Matching Jobs"):
    if resume_text:
        best_job = match_resume_to_jobs(resume_text)
        st.success(f"✅ Best Job Match: **{best_job}**")
    else:
        st.warning("⚠️ Please provide resume text or upload a resume!")
        
        
print("run above code using this code:",streamlit run app.py)


# AI-Powered Semantic PDF Assistant for Sustainable Education

## 📌 Project Overview
This project is an AI-powered Semantic PDF Question Answering system built using **Retrieval-Augmented Generation (RAG)**. It enables users to upload large PDF documents (100+ pages) and ask natural-language questions to receive concise, context-aware answers.

The solution addresses challenges in accessing and understanding long educational documents and contributes to **SDG 4: Quality Education** by promoting inclusive, efficient, and sustainable learning.

---

## 🎯 Sustainable Development Goal (SDG)
**Primary SDG:**  
- SDG 4 – Quality Education  

**Secondary SDGs:**  
- SDG 9 – Industry, Innovation & Infrastructure  
- SDG 16 – Peace, Justice & Strong Institutions (Access to Information)

---

## ❓ Problem Statement
Students, educators, and researchers often struggle to navigate lengthy PDFs such as textbooks, research papers, and policy documents. Traditional keyword-based search fails to capture semantic meaning, making information retrieval inefficient and time-consuming. This limits accessibility to knowledge and negatively impacts inclusive education.

---

## 💡 Solution Description
The system uses **Retrieval-Augmented Generation (RAG)** to combine semantic search with a transformer-based language model. Uploaded PDFs are split into chunks, embedded into vectors, and stored in a FAISS vector database. When a user asks a question, the system retrieves the most relevant document sections and generates a clear, accurate answer strictly based on the uploaded content.

---

## 🧠 AI Technologies Used
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Sentence Transformer Embeddings
- FAISS Vector Database
- Transformer-based Language Model (FLAN-T5)
- Prompt Engineering
- LangChain
- Streamlit

---

## 👥 Target Users
- College and university students  
- Self-learners and lifelong learners  
- Teachers and academic researchers  
- Educational and training institutions  

---

## 🌍 Expected Impact
- Improves learning efficiency and comprehension  
- Enables inclusive and self-paced education  
- Reduces time spent reading long documents  
- Promotes sustainable digital learning practices  
- Democratizes access to knowledge  

---

## ⚖️ Responsible AI Considerations
- **Fairness:** No demographic or personal data used  
- **Transparency:** Answers generated only from uploaded documents  
- **Ethics:** Prevents hallucinations by restricting context  
- **Privacy:** No data storage beyond session usage  

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt

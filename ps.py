#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import re
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter
import openai
import os
import time
import uuid
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# RAG & AI Enhancement Imports (Experimental Feature)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import PyPDF2
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ RAG dependencies not installed. RAG features disabled.")

def force_scroll_to_bottom():
    """Auto-scroll to bottom of page for better chat experience"""
    js_code = """
    <script>
    function scrollDown() {
        try {
            // Try scrolling to bottom
            window.parent.scrollTo({
                top: window.parent.document.body.scrollHeight,
                behavior: 'smooth'
            });
            
            // Alternative scroll method
            window.parent.document.documentElement.scrollTop = window.parent.document.documentElement.scrollHeight;
            
            // Find chat section
            const chatSection = window.parent.document.getElementById('chat-section');
            if (chatSection) {
                chatSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            
            // Find chat input box
            const chatInput = window.parent.document.querySelector('[data-testid="stChatInput"]');
            if (chatInput) {
                chatInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            // Find last message
            const chatMessages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            if (chatMessages.length > 0) {
                const lastMessage = chatMessages[chatMessages.length - 1];
                lastMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            // Force scroll with delay
            setTimeout(() => {
                window.parent.scrollTo({
                    top: window.parent.document.body.scrollHeight + 1000,
                    behavior: 'smooth'
                });
            }, 100);
            
        } catch (error) {
            console.log("Scroll error:", error);
            // Simple fallback scroll
            window.parent.scrollTo(0, 999999);
        }
    }
    
    // Run scroll function multiple times
    scrollDown();
    setTimeout(scrollDown, 50);
    setTimeout(scrollDown, 150);
    setTimeout(scrollDown, 300);
    setTimeout(scrollDown, 500);
    setTimeout(scrollDown, 800);
    setTimeout(scrollDown, 1200);
    setTimeout(scrollDown, 2000);
    </script>
    """
    components.html(js_code, height=0)


class ResumeData:
    """Contains all personal and professional information"""
    
    def __init__(self):
        self.data = {
            "personal_info": {
                "name": "Petros Venieris",
                "title": "Data Scientist & Machine Learning Engineer",
                "email": "pgvenieris@outlook.com",
                "linkedin": "https://linkedin.com/in/petrosvenieris",
                "github": "https://github.com/venie1",
                "location": "Atlanta, GA"
            },
            
            "summary": """Elite Data Scientist and Machine Learning Engineer with a Master's degree from Georgia Institute of Technology (#9 globally ranked Data Science program). Proven expertise in predictive modeling, advanced analytics, and AI system development with exceptional academic performance (3.91/4.0 GPA). Distinguished by industry partnerships with Sandia National Laboratories and quantifiable achievements including 90% accuracy in live sports prediction systems and exceptional performance in algorithmic trading applications.""",
            
            "education": [
                {
                    "school": "Georgia Institute of Technology (Georgia Tech)",
                    "degree": "Master of Science in Analytics",
                    "period": "August 2023 - August 2025 (Expected)",
                    "gpa": "3.91/4.0",
                    "ranking": "🏆 #9 Globally Ranked Data Science Program | #4 in United States",
                    "location": "Atlanta, GA",
                    "program_rigor": """
**Program Rigor & Selectivity:**
• Highly selective program with ~15% acceptance rate
• 36 credit hours of intensive coursework in advanced analytics
• Requires strong mathematical background and programming proficiency
• Industry practicum requirement with top-tier organizations
• Curriculum designed by leading industry experts and researchers

**Academic Excellence Indicators:**
• 3.91/4.0 GPA demonstrates mastery of challenging technical content
• Consistent top-tier performance across all core subjects
• Selected for prestigious Sandia National Laboratories practicum
• Academic standing places student in top 10% of cohort
                    """,
                    "coursework": [
                        "Advanced Machine Learning & Deep Learning",
                        "Time Series Analysis & Forecasting", 
                        "Statistical Methods & Hypothesis Testing",
                        "Optimization & Operations Research",
                        "AI-Driven Analytics & Reinforcement Learning",
                        "Data Mining & Pattern Recognition",
                        "High-Performance Computing for Analytics",
                        "Business Analytics & Strategy"
                    ]
                },
                {
                    "school": "University of Piraeus",
                    "degree": "Bachelor of Science in Digital Systems",
                    "period": "September 2018 - July 2022",
                    "gpa": "7.4/10.0",
                    "location": "Piraeus, Greece",
                    "coursework": ["Software Engineering", "Database Systems", "Computer Networks", "Data Structures", "Algorithms"]
                }
            ],
            
            "experience": [
                {
                    "role": "Data Science Research Assistant",
                    "company": "Sandia National Laboratories",
                    "period": "May 2025 – Aug 2025",
                    "location": "Albuquerque, NM (Remote)",
                    "highlights": [
                        "Developed advanced predictive maintenance algorithms for critical infrastructure systems using 1.1+ million data points",
                        "Achieved 0.732 ROC-AUC score in failure prediction models, significantly outperforming baseline approaches",
                        "Implemented ensemble methods combining XGBoost, Random Forest, and Neural Networks for optimal performance",
                        "Created comprehensive data preprocessing pipelines handling multiple sensor data streams and temporal patterns",
                        "Pioneered behavioral vs. technical failure prediction methodology with potential for research publication"
                    ],
                    "metrics": [
                        "📊 Processed 1.1+ million sensor data points with 94% data quality retention",
                        "🎯 Achieved 0.732 ROC-AUC in predictive maintenance models (15% improvement over baseline)",
                        "⚡ Reduced false positive rates by 23% through advanced feature engineering",
                        "🔧 Deployed models processing real-time data streams with <100ms latency"
                    ],
                    "tech_stack": ["Python", "Scikit-learn", "XGBoost", "TensorFlow", "Pandas", "NumPy", "SQL", "Git"]
                },
                {
                    "role": "Sports Data Scientist, Self employed with Partner",
                    "company": "BetMax",
                    "period": "Jan 2019 – Dec 2022",
                    "location": "Athens, Greece",
                    "highlights": [
                        "Developed machine learning models achieving 90% accuracy in live sports prediction across 1,000+ matches",
                        "Built comprehensive ETL pipelines processing real-time sports data from multiple API sources",
                        "Implemented advanced feature engineering incorporating team performance, player statistics, and contextual factors",
                        "Created automated model retraining systems maintaining prediction accuracy throughout seasons",
                        "Delivered actionable insights through interactive dashboards and real-time alerting systems"
                    ],
                    "metrics": [
                        "🎯 90% prediction accuracy across 1,000+ live sports matches",
                        "📈 35% improvement in betting strategy performance through ML insights",
                        "⚡ Real-time predictions delivered within 3-second SLA",
                        "💰 Generated measurable ROI improvement for analytics-driven decisions"
                    ],
                    "tech_stack": ["Python", "Scikit-learn", "Pandas", "SQL", "API Integration", "Power BI", "Git"]
                }
            ],
            
            "projects": [
                {
                    "name": "AI Resume Agent (AI Engineering/Open Source)",
                    "period": "2025 - Current (Personal Project, Open Source)",
                    "description": "Interactive AI-powered resume assistant using OpenAI GPT-4 API integration. Features conversational AI, advanced prompt engineering, session management, and real-time chat interface. Built with Streamlit for deployment and demonstrates practical LLM application development.",
                    "achievements": [
                        "OpenAI GPT-4 API integration with custom prompt engineering",
                        "Conversational AI interface with intelligent response system",
                        "Advanced session state management and real-time chat functionality",
                        "Professional UI/UX with interactive navigation and visualizations",
                        "Dynamic data visualization integration with Plotly charts",
                        "Streamlit deployment with responsive design and scroll automation"
                    ],
                    "tech_stack": ["Python", "OpenAI API", "Streamlit", "Plotly", "Session Management", "LLM Integration"],
                    "github": "https://github.com/venie1/ResumeAgent",
                    "project_type": "Personal/AI Engineering (open source)"
                },
                {
                    "name": "Sandia Labs Predictive Maintenance (Work/Research)",
                    "period": "May 2025 – Aug 2025 (Sandia National Laboratories, Georgia Tech)",
                    "description": "Behavioral vs. technical analysis of truck failures for predictive maintenance. Modeled both driver behaviors and mechanical indicators to forecast component failures. Large-scale, real-world telematics data (1.1M time-steps, 23,550 vehicles).",
                    "achievements": [
                        "Merged 1.1M time-steps from 23,550 vehicles; engineered 533 features",
                        "Partitioned behavioral (24) and technical (81) features for hybrid modeling",
                        "Used Random Forest, Gradient Boosting, XGBoost; AUC up to 0.732, 20% cost reduction",
                        "Business impact: targeted driver training, reduced safety incidents, cost savings",
                        "Report and summary available; code proprietary (NDA)"
                    ],
                    "tech_stack": ["Python", "pandas", "NumPy", "scikit-learn", "XGBoost", "matplotlib", "seaborn"],
                    "github": "https://github.com/venie1/Predictive-Maintenance-Behavioral-vs.-Technical-Analysis-of-Truck-Failures",
                    "project_type": "Work/Research (no code, report only)"
                },
                {
                    "name": "RealEstateAdvisor (School/Open Source)",
                    "period": "July 2025 (Georgia Tech, Open Source)",
                    "description": "End-to-end real estate forecasting platform for city-level price prediction. Used macroeconomic, socio-economic, and market features. Competitive MAPE for 1-/3-month (6%) and 6-/12-month (7–10%) horizons.",
                    "achievements": [
                        "Merged Redfin, FRED, Census data; engineered lagged/rolling features",
                        "Trained RidgeCV, Random Forest, XGBoost, Prophet",
                        "Automated 80% of preprocessing; Power BI dashboard with dynamic maps",
                        "Simulated 12–15% portfolio return uplift in backtests",
                        "Open source, code and results available"
                    ],
                    "tech_stack": ["Python", "Prophet", "Scikit-learn", "Streamlit", "GeoPandas", "SQL", "APIs"],
                    "github": "https://github.com/venie1/RealEstateAdvisor",
                    "project_type": "School/Group (open source)"
                },
                {
                    "name": "Sports Prediction System (BetMax, Self-Employed)",
                    "period": "Jan 2019 – Dec 2022 (Self-Employed, Athens, Greece)",
                    "description": "Co-founded and built an automated live football match prediction system for second-half goal occurrence (over 0.5 goals). 90% accuracy for high-confidence picks (vs 75–80% base rate). Real-time and historical data scraping (Selenium), automated notifications, and open source deployment.",
                    "achievements": [
                        "Developed and maintained with a partner; processed 1,000+ matches",
                        "Automated data scraping (Selenium) and real-time ETL pipelines",
                        "Automated high-confidence email alerts for stakeholders",
                        "Benchmarked against public league statistics (SoccerSTATS, AFootballReport)",
                        "Open source, code and methodology available"
                    ],
                    "tech_stack": ["Python", "Selenium", "Scikit-learn", "Pandas", "SQL", "API Integration", "Power BI", "Git"],
                    "github": "https://github.com/venie1/BetPredictor",
                    "project_type": "Self-Employed/Group (open source)"
                },
                {
                    "name": "COVID-19 Forecasting & Insights Application (Bachelor Thesis)",
                    "period": "2022 (University of Piraeus, Bachelor Thesis, Grade: 10/10)",
                    "description": "Bachelor thesis: Designed and deployed a Flask app for COVID-19 case analysis and forecasting. Automated data scraping, ETL, and notifications. Benchmarked seven time-series models for 14-day forecasts. Reduced RMSE by 15% over naive baseline (Prophet RMSE: 88.8).",
                    "achievements": [
                        "Automated data scraping and ETL from ECDC using Selenium",
                        "Engineered time-series features and uncertainty quantification",
                        "Benchmarked Polynomial, SVM, Holt's, ARIMA/SARIMA, Prophet",
                        "Automated notifications and interactive dashboards (Plotly, Flask)",
                        "Grade: 10/10, delivered actionable insights to public health stakeholders"
                    ],
                    "tech_stack": ["Python", "Selenium", "Flask", "Prophet", "Plotly", "Streamlit", "Pandas", "APIs", "Statistical Modeling"],
                    "github": "https://github.com/venie1/Covid-prediction-application",
                    "project_type": "Bachelor Thesis (open source)"
                }
            ],
            
            "skills": {
                "Core Technical": [
                    "Python", "R", "SQL", "Scikit-learn", "TensorFlow", "PyTorch", "XGBoost", "Prophet", "Selenium", "Flask", "Docker", "Git", "API Integration", "ETL Pipelines", "Data Visualization"
                ],
                "Data Science & Analytics": [
                    "Time-Series Forecasting", "Predictive Modeling", "A/B Testing", "Hypothesis Testing", "Hyperparameter Tuning", "Feature Engineering", "Statistical Analysis", "Machine Learning", "Deep Learning"
                ],
                "Visualization & BI": [
                    "Power BI", "Plotly", "Matplotlib", "Tableau", "Streamlit", "Jupyter Notebooks"
                ],
                "Programming & Development": [
                    "Python", "R", "SQL", "JavaScript", "Git", "Linux/Unix", "API Development"
                ],
                "Database & Cloud": [
                    "PostgreSQL", "MySQL", "MongoDB", "AWS", "Data Warehousing"
                ],
                "Specialized Domains": [
                    "Financial Analytics", "Sports Analytics", "Predictive Maintenance", "Real Estate Forecasting"
                ]
            },
            
            "technical_highlights": [
                "🤖 **AI Engineering:** Built production AI Resume Agent with OpenAI GPT-4 API integration, conversational AI, and advanced prompt engineering",
                "🧠 **RAG & Neural Retrieval:** Implemented state-of-the-art Retrieval-Augmented Generation with vector embeddings, semantic search, and multi-modal AI architecture",
                "🎯 **90% Prediction Accuracy:** Achieved exceptional performance in live sports prediction systems across 1,000+ matches",
                "🏭 **Sandia Labs Research:** Developing cutting-edge predictive maintenance algorithms with national laboratory scientists",
                "💬 **LLM Applications:** Practical experience with large language model integration, intelligent response systems, and conversational AI development",
                "🔍 **Vector Embeddings & FAISS:** Advanced semantic similarity search using sentence-transformers and efficient indexing for neural information retrieval",
                "🎓 **Georgia Tech Excellence:** 3.91/4.0 GPA in world's #9 ranked Data Science program with advanced ML specialization",
                "🔬 **Research Potential:** In discussions for co-authored publication with Sandia National Laboratories and Georgia Tech faculty",
                "⚡ **Real-time Systems:** Built production-grade ML systems processing millions of data points with sub-second latency",
                "🌐 **Full-Stack AI:** End-to-end AI capabilities from data processing to deployed conversational AI applications with knowledge base augmentation",
                "📊 **Advanced Analytics:** Expert in ensemble methods, time-series forecasting, and statistical modeling techniques"
            ]
        }

class ChatbotConfig:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 300
        self.temperature = 0.5
        self.use_fallback = True
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]
        self.enable_openai = True
        self.enable_rag = False  # RAG experimental feature

class RAGEnhancedSystem:
    """🧠 Advanced RAG (Retrieval-Augmented Generation) System with Vector Embeddings & Semantic Search
    
    This experimental system leverages state-of-the-art neural information retrieval to enhance 
    conversational AI responses with contextually relevant resume data through multi-modal AI architecture.
    """
    
    def __init__(self, resume_data, pdf_path="CV_Petros_Venieris.pdf"):
        self.resume_data = resume_data
        self.pdf_path = pdf_path
        self.embedding_model = None
        self.vector_store = None
        self.text_chunks = []
        self.chunk_metadata = []
        
        if RAG_AVAILABLE:
            self._initialize_rag_system()
        else:
            st.warning("⚠️ RAG system unavailable - install dependencies to enable neural semantic search")
    
    def _initialize_rag_system(self):
        """Initialize the Neural Semantic Search & Vector Embedding System"""
        try:
            # Load sentence transformer model for semantic embeddings
            with st.spinner("🧠 Loading Neural Embedding Model..."):
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Process and chunk resume data + PDF
            self._create_knowledge_base()
            
            # Build vector store with FAISS for efficient similarity search
            self._build_vector_store()
            
        except Exception as e:
            st.error(f"RAG System initialization failed: {str(e)}")
    
    def _create_knowledge_base(self):
        """Create comprehensive knowledge base from resume data and PDF"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Process structured resume data
        for section, content in self.resume_data.data.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, str):
                        chunks = text_splitter.split_text(f"{key}: {value}")
                        for chunk in chunks:
                            self.text_chunks.append(chunk)
                            self.chunk_metadata.append({"source": f"resume_{section}_{key}", "type": "structured"})
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                chunks = text_splitter.split_text(f"{key}: {item}")
                            elif isinstance(item, dict):
                                item_text = " ".join([f"{k}: {v}" for k, v in item.items() if isinstance(v, (str, int, float))])
                                chunks = text_splitter.split_text(item_text)
                            
                            for chunk in chunks:
                                self.text_chunks.append(chunk)
                                self.chunk_metadata.append({"source": f"resume_{section}_{key}", "type": "structured"})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        chunks = text_splitter.split_text(item)
                    elif isinstance(item, dict):
                        item_text = " ".join([f"{k}: {v}" for k, v in item.items() if isinstance(v, (str, int, float))])
                        chunks = text_splitter.split_text(item_text)
                    
                    for chunk in chunks:
                        self.text_chunks.append(chunk)
                        self.chunk_metadata.append({"source": f"resume_{section}", "type": "structured"})
        
        # Process PDF if available
        if os.path.exists(self.pdf_path):
            try:
                with open(self.pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                    
                    pdf_chunks = text_splitter.split_text(pdf_text)
                    for chunk in pdf_chunks:
                        self.text_chunks.append(chunk)
                        self.chunk_metadata.append({"source": "resume_pdf", "type": "pdf"})
            except Exception as e:
                st.warning(f"Could not process PDF: {str(e)}")
    
    def _build_vector_store(self):
        """Build FAISS vector store for efficient semantic similarity search"""
        if not self.text_chunks:
            return
        
        with st.spinner("🔍 Building Vector Embeddings & Semantic Index..."):
            # Generate embeddings for all text chunks
            embeddings = self.embedding_model.encode(self.text_chunks)
            
            # Initialize FAISS index for cosine similarity search
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.vector_store.add(embeddings.astype('float32'))
    
    def retrieve_context(self, query, top_k=3):
        """🎯 Neural Semantic Retrieval - Find most relevant context using vector similarity"""
        if not self.vector_store or not self.embedding_model:
            return []
        
        try:
            # Encode query using the same embedding model
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Perform semantic similarity search
            scores, indices = self.vector_store.search(query_embedding.astype('float32'), top_k)
            
            # Retrieve relevant text chunks with metadata
            retrieved_contexts = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.text_chunks) and score > 0.3:  # Similarity threshold
                    retrieved_contexts.append({
                        'text': self.text_chunks[idx],
                        'metadata': self.chunk_metadata[idx],
                        'similarity_score': float(score),
                        'rank': i + 1
                    })
            
            return retrieved_contexts
        except Exception as e:
            st.error(f"RAG retrieval error: {str(e)}")
            return []
    
    def generate_rag_enhanced_context(self, query, user_input):
        """🌐 Generate enhanced context using RAG for improved AI responses"""
        if not RAG_AVAILABLE:
            return ""
        
        retrieved_contexts = self.retrieve_context(query, top_k=3)
        
        if not retrieved_contexts:
            return ""
        
        # Build enhanced context prompt
        context_prompt = "\n🧠 **NEURAL SEMANTIC CONTEXT** (Retrieved via RAG):\n"
        for ctx in retrieved_contexts:
            context_prompt += f"📊 Source: {ctx['metadata']['source']} (Similarity: {ctx['similarity_score']:.3f})\n"
            context_prompt += f"🎯 Context: {ctx['text']}\n\n"
        
        context_prompt += "⚡ Use this contextually relevant information to provide enhanced, accurate responses about Petros Venieris.\n"
        context_prompt += "🎯 Maintain all original agent guidelines while leveraging this semantic context.\n\n"
        
        return context_prompt

class IntelligentResponseSystem:
    """Handles chat responses and keyword matching"""
    
    def __init__(self, resume_data, config):
        self.resume_data = resume_data
        self.config = config
        self.keywords = self.build_keyword_index()
        self.data_science_keywords = self.build_data_science_keywords()
        
        # Initialize RAG system if enabled
        self.rag_system = None
        if config.enable_rag and RAG_AVAILABLE:
            self.rag_system = RAGEnhancedSystem(resume_data)
        
        if config.enable_openai:
            openai.api_key = config.openai_api_key
        
    def build_keyword_index(self):
        """Create keyword dictionary for matching user questions to topics"""
        keywords = {
            'experience': ['experience', 'work', 'job', 'employment', 'career', 'professional', 'role', 'position', 'sandia', 'betmax', 'practicum', 'data scientist', 'machine learning engineer'],
            'education': ['education', 'degree', 'school', 'university', 'study', 'academic', 'learning', 'qualification', 'georgia tech', 'piraeus', 'analytics', 'masters', 'gpa'],
            'projects': ['project', 'portfolio', 'work', 'development', 'built', 'created', 'application', 'system', 'real estate', 'covid', 'forecasting', 'prediction', 'trading', 'sports', 'ai resume agent', 'resume agent', 'chatbot', 'ai assistant', 'conversational ai', 'openai', 'gpt', 'llm', 'rag', 'vector embeddings', 'semantic search', 'neural retrieval'],
            'skills': ['skills', 'technology', 'programming', 'language', 'tool', 'framework', 'expertise', 'proficiency', 'python', 'sql', 'machine learning', 'ml', 'ai', 'deep learning', 'openai api', 'gpt integration', 'llm', 'conversational ai', 'prompt engineering', 'streamlit', 'rag', 'retrieval augmented generation', 'vector embeddings', 'semantic search', 'faiss', 'sentence transformers', 'langchain', 'neural information retrieval'],
            'achievements': ['achievement', 'accuracy', 'performance', 'results', 'improvement', 'success', 'metrics', 'roi', 'impact'],
            'contact': ['contact', 'email', 'linkedin', 'github', 'reach', 'connect', 'information', 'phone'],
            'summary': ['about', 'summary', 'overview', 'background', 'profile', 'introduction', 'candidate']
        }
        return keywords
    
    def build_data_science_keywords(self):
        """Keywords to check if question is about data science roles"""
        return {
            'ml_keywords': ['machine learning', 'ml', 'data science', 'ai', 'artificial intelligence', 
                           'predictive', 'analytics', 'modeling', 'statistics', 'python', 'sql', 
                           'tensorflow', 'scikit', 'data scientist', 'data analyst', 'quantitative'],
            'related_fields': ['software engineer', 'backend', 'frontend', 'full stack', 'devops', 
                              'research', 'academic', 'analyst', 'consultant', 'tech'],
            'unrelated_fields': ['marketing', 'sales', 'hr', 'human resources', 'accounting', 
                                'finance manager', 'operations manager', 'customer service', 
                                'retail', 'hospitality', 'construction', 'healthcare provider']
        }
    
    def find_best_match(self, user_input):
        """Match user question to relevant resume section"""
        user_input_lower = user_input.lower()
        scores = {}
        
        for section, keywords in self.keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                scores[section] = score
        
        if scores:
            return max(scores, key=scores.get)
        return None
    
    def is_relevant_field(self, user_input):
        """Check if question is about tech/data science roles"""
        user_input_lower = user_input.lower()
        
        # Check for ML/DS keywords (highly relevant)
        if any(keyword in user_input_lower for keyword in self.data_science_keywords['ml_keywords']):
            return 'highly_relevant'
        
        # Check for related tech fields
        if any(keyword in user_input_lower for keyword in self.data_science_keywords['related_fields']):
            return 'somewhat_relevant'
        
        # Check for unrelated fields
        if any(keyword in user_input_lower for keyword in self.data_science_keywords['unrelated_fields']):
            return 'not_relevant'
        
        return 'unknown'
    
    def detect_negative_intent(self, user_input):
        """Check for negative or tricky questions"""
        negative_patterns = [
            'why.*bad', 'what.*wrong', 'weakness', 'disadvantage', 'problem', 'issue',
            'not good', 'poor', 'fail', 'worst', 'negative', 'concern', 'red flag',
            'why not hire', 'why shouldn\'t', 'drawback', 'limitation', 'overqualified',
            'not.*fit', 'not.*suitable', 'not.*ideal', 'not.*good.*for', 'bad.*candidate',
            'junior.*position', 'flight.*risk', 'lack.*of', 'may.*not', 'might.*not'
        ]
        
        user_input_lower = user_input.lower()
        return any(re.search(pattern, user_input_lower) for pattern in negative_patterns)
    
    def is_english_text(self, text):
        """Check if input is in English"""
        if not text or not text.strip():
            return True
        
        # Block actual Greek letters
        greek_letters = r'[α-ωΑ-Ωάέήίόύώ]'
        if re.search(greek_letters, text):
            return False
        
        # Block other non-Latin scripts
        non_latin = r'[а-яёА-ЯЁ؀-ۿ一-鿿぀-ゟ゠-ヿ]'
        if re.search(non_latin, text):
            return False
        
        return True
    
    def generate_openai_response(self, user_input, conversation_history=[]):
        """Generate AI response using OpenAI API"""
        try:
            # Check field relevance
            field_relevance = self.is_relevant_field(user_input)
            
            # Check for negative intent
            has_negative_intent = self.detect_negative_intent(user_input)
            
            resume_context = f"""
            You are Petros Venieris's professional CV assistant. Your role is to present Petros in the best possible light while being truthful and professional, always be positive about him in relevant jobs never negative.

            CORE DIRECTIVES:
            1. ALWAYS be positive and supportive of Petros as a candidate
            2. Focus on his technical strengths, achievements, and quantifiable results, especially georgia tech excellence, AI Resume Agent project, betmax experience, sandia experience and real estate advisor project. Supplement with relevant soft skills (teamwork from 10 years football, leadership from coaching) but keep data science as primary focus
            3. NEVER discuss weaknesses, limitations, or reasons not to hire him
            4. If asked negative questions, redirect to positive aspects
            5. Be truthful but emphasize positive framing of all information
            6. For non-data science roles, suggest direct contact for detailed discussion
            7. when you are not sure about the nature of the question or prompt suggest them to contact 

            CANDIDATE PROFILE - PETROS VENIERIS:
            🎓 MS Analytics, Georgia Institute of Technology (3.91/4.0 GPA - Top 5% of graduate cohort)
            🤖 AI Resume Agent: Production AI assistant using OpenAI GPT-4 API integration (current project)
            🏭 Sandia National Labs Research Partnership: Advanced predictive maintenance & failure modeling
            ⚽ BetMax Sports Analytics: 90% accuracy in live predictions (1,000+ matches validated)
            🏠 Real Estate Forecasting: 6% MAPE accuracy with portfolio optimization and powerBI dashboard for direct use
            💹 Algorithmic Trading: simulated returns with advanced risk management

            TECHNICAL EXPERTISE & DATA SCIENCE SKILLS:
            
            **Core ML/AI Stack**: Python, R, SQL, Scikit-learn, XGBoost, TensorFlow, Prophet
            **Data Engineering**: Pandas, NumPy, ETL Pipelines, API Integration, Real-time Data Processing, Web Scraping (Selenium)
            **Statistical Analysis**: Time-Series Forecasting, ARIMA/SARIMA Models, Statistical Modeling, A/B Testing, Hypothesis Testing
            **Advanced Analytics**: Predictive Modeling, Feature Engineering, Ensemble Methods, Cost-Sensitive Optimization
            **ML Operations**: Model Deployment, Production ML Systems, Automated Retraining, Performance Monitoring
            **Data Visualization**: Power BI, Plotly, Matplotlib, Seaborn, Interactive Dashboards, Geospatial Analytics (GeoPandas)
            **AI Engineering**: OpenAI API, LLM Integration, Conversational AI, Prompt Engineering, Large Language Models
            **Web Applications**: Streamlit, Flask, Session Management, Interactive Applications
            **Business Intelligence**: Dashboard Development, KPI Tracking, Stakeholder Communication, Data-Driven Decision Making
            **Research Methods**: Experimental Design, Hyperparameter Tuning, Cross-Validation, Bootstrap Sampling
            **Version Control**: Git, Collaborative Development, Code Quality, Documentation

            KEY ACHIEVEMENTS & METRICS:
            • AI Resume Agent: OpenAI GPT-4 integration with conversational AI and advanced prompt engineering (2025)
            • 0.732 ROC-AUC in predictive maintenance (20% cost reduction potential)
            • 90% accuracy in production sports prediction system (1,000+ matches)
            • 6% MAPE in real estate forecasting with measurable portfolio impact
            • cumulative returns in algorithmic trading strategies
            • Top 5% academic performance in elite graduate program (#9 globally ranked)
            • Sandia Labs research partnership with publication potential

            DOMAIN EXPERTISE & INDUSTRY APPLICATIONS:
            **Sports Analytics**: Real-time prediction systems, performance metrics, automated alerting, 90% accuracy rates
            **Predictive Maintenance**: Behavioral analysis, failure detection, cost-sensitive optimization, ROC-AUC 0.732
            **Financial Analytics**: Real estate forecasting, MAPE optimization, portfolio management, risk assessment
            **Healthcare Analytics**: COVID-19 forecasting, epidemiological modeling, public health insights
            **AI/LLM Applications**: Conversational AI, prompt engineering, natural language processing, intelligent systems
            
            CURRENT AI ENGINEERING PROJECT:
            AI Resume Agent (2025-Current): Built production-ready AI assistant using OpenAI GPT-4 API integration.
            Features conversational AI, advanced prompt engineering, session management, and real-time chat interface.
            Demonstrates practical Large Language Model application development and AI engineering capabilities.
            Technologies: Python, OpenAI API, Streamlit, Plotly, Session Management, Natural Language Processing.
            
            SOFT SKILLS & LEADERSHIP EXPERIENCE:
            **Team Collaboration**: 10+ years of football (team sport requiring coordination, communication, strategy)
            **Leadership**: Coached young players, demonstrated mentoring and guidance capabilities
            **Project Management**: Co-founded and maintained BetMax for 4 years, showing sustained project leadership
            **Cross-functional Communication**: Worked with diverse stakeholders (Georgia Tech faculty, Sandia Labs scientists)
            **Problem-Solving**: Innovative approaches to complex analytical challenges across multiple domains
            **Adaptability**: Successfully transitioned between different technical domains (sports, healthcare, real estate, AI)
            **Work Ethic**: Maintained part-time work while excelling academically (3.91 GPA), demonstrating time management
            **Mentorship**: Experience coaching football players translates to team development and knowledge transfer
            
            RECRUITER-RELEVANT KEYWORDS (100% truthful to experience):
            Machine Learning Engineer, Data Scientist, AI Engineer, MLOps, Production ML Systems, Real-time Analytics,
            Time-Series Forecasting, Predictive Analytics, Statistical Modeling, Feature Engineering, Model Deployment,
            Data Visualization, Business Intelligence, ETL Pipelines, API Development, Conversational AI, 
            Large Language Models, Prompt Engineering, Research & Development, Academic Excellence (Top 5%),
            Team Leadership, Cross-functional Collaboration, Project Management, Mentorship, Problem-Solving. 

            RESPONSE GUIDELINES:
            - Always highlight Petros's exceptional qualifications and achievements
            - Use specific metrics and quantifiable results
            - Emphasize his academic excellence and research partnerships
            - Focus on technical depth and proven business impact
            - Redirect negative questions to positive accomplishments
            - For unrelated fields, be supportive but suggest direct contact
            - Maintain professional, confident, and enthusiastic tone
            - Always use pgvenieris@outlook.com for contact information
            """
            
            # Handle different scenarios
            if has_negative_intent:
                system_addition = """
                IMPORTANT: The user is asking about negative aspects or weaknesses. You must:
                1. Politely redirect the conversation to Petros's strengths
                2. Never provide negative information or reasons not to hire
                3. Transform any negative framing into positive achievements
                4. Focus on his exceptional qualifications and proven track record
                5. in case of further confusion suggest to contact petros directly at pgvenieris@outlook.com
                """
                resume_context += system_addition
            
            if field_relevance == 'not_relevant':
                system_addition = """
                IMPORTANT: This query is about a field outside of Petros's core expertise in Data Science/ML.
                Respond supportively but suggest direct contact: "While Petros's primary expertise is in Data Science and Machine Learning, I'd recommend contacting him directly at pgvenieris@outlook.com to discuss how his analytical skills and technical background might apply to this role."
                """
                resume_context += system_addition
            
            # 🧠 RAG ENHANCEMENT: Add neural semantic context if enabled
            if self.config.enable_rag and self.rag_system:
                rag_context = self.rag_system.generate_rag_enhanced_context(user_input, user_input)
                if rag_context:
                    resume_context += rag_context
            
            client = openai.OpenAI(api_key=self.config.openai_api_key)
            
            messages = [{"role": "system", "content": resume_context}]
            
            # Add conversation history (limited to avoid token overload)
            for msg in conversation_history[-3:]:
                if isinstance(msg, dict) and 'user' in msg and 'assistant' in msg:
                    messages.append({"role": "user", "content": msg["user"]})
                    messages.append({"role": "assistant", "content": msg["assistant"]})
            
            messages.append({"role": "user", "content": user_input})
            
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=0.6,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self.get_fallback_response(user_input)
    
    def get_fallback_response(self, user_input):
        """Fallback response when OpenAI API fails"""
        # Check for negative intent in fallback
        if self.detect_negative_intent(user_input):
            return """
## 🌟 Petros's Exceptional Qualifications

I'm here to highlight Petros's outstanding achievements and qualifications! Here's why he's an exceptional Data Science candidate:

### 🏆 **Academic Excellence**
- **3.91/4.0 GPA** at Georgia Institute of Technology (Top 5% of graduate cohort)
- **#9 globally ranked** Data Science program with rigorous curriculum

### 🎯 **Proven Track Record**
- **90% accuracy** in production ML systems (validated across 1,000+ predictions)
- **0.732 ROC-AUC** in predictive maintenance with measurable business impact
- **Sandia Labs partnership** - elite research collaboration

### 💼 **Industry-Ready Skills**
- Advanced predictive modeling and time-series forecasting
- Production ML system deployment and optimization
- Full-stack data science capabilities from research to deployment

**What specific achievements or technical skills would you like to know more about?**

**Contact**: pgvenieris@outlook.com
            """
        
        # Check field relevance
        field_relevance = self.is_relevant_field(user_input)
        if field_relevance == 'not_relevant':
            return """
## 🤝 Direct Contact Recommended

While Petros's primary expertise is in **Data Science and Machine Learning**, his strong analytical background and technical skills may be valuable in other contexts. 

For roles outside his core specialization, I'd recommend contacting him directly:
- **Email**: pgvenieris@outlook.com
- **LinkedIn**: https://linkedin.com/in/petrosvenieris

This will allow for a detailed discussion about how his **quantitative analysis skills**, **research experience**, and **technical problem-solving abilities** might apply to your specific needs.

**For Data Science, ML Engineering, or Analytics roles, I'm here to provide detailed information about his exceptional qualifications!**
            """
        
        # Default positive response
        section = self.find_best_match(user_input)
        if section:
            return self.get_section_response(section)
        return self.get_default_response()
    
    def generate_intelligent_response(self, user_input, conversation_history=[]):
        """Main function to generate responses"""
        # Pre-check input
        if not self.is_english_text(user_input):
            return self.get_language_error_response()
            
        if self.detect_negative_intent(user_input):
            return self.get_positive_redirect_response()
        
        if self.config.enable_openai and len(user_input.split()) > 2:
            return self.generate_openai_response(user_input, conversation_history)
        
        section = self.find_best_match(user_input)
        
        if section:
            return self.get_section_response(section)
        
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'start', 'begin']):
            return self.get_welcome_response()
        
        if any(word in user_input_lower for word in ['help', 'what', 'how', 'can', 'ask']):
            return self.get_help_response()
        
        return self.get_default_response()
    
    def get_language_error_response(self):
        """Show error when user inputs non-English text"""
        return """
## 🌍 Language Notice

**This assistant only accepts and responds to English language queries.**

Please rephrase your question in English to receive detailed information about Petros Venieris's:
- Technical skills and expertise
- Project achievements and results  
- Academic excellence and qualifications
- Professional experience and impact

**Example English queries:**
- "Tell me about his machine learning experience"
- "What are his technical skills?"
- "Show me his project achievements"
- "What makes him a strong candidate?"

Thank you for your understanding!
        """

    def get_positive_redirect_response(self):
        """Handle negative questions by showing positive info"""
        return """
## 🎯 Highlighting Exceptional Qualifications

I'm here to showcase Petros's outstanding strengths and achievements. For any detailed discussions beyond his documented qualifications, please contact him directly at **pgvenieris@outlook.com**.

### 🏆 **Why Petros is an Exceptional Candidate:**

**🎓 Academic Excellence:**
- 3.91/4.0 GPA at Georgia Tech (#9 globally ranked program)
- Top 5% performance in elite graduate cohort
- Advanced ML and Data Science specialization

**🔬 Research & Industry Impact:**
- Sandia National Labs research partnership
- 90% accuracy in production ML systems
- 0.732 ROC-AUC with 20% cost reduction potential
- Publication potential with national laboratory

**💼 Proven Technical Leadership:**
- Full-stack data science capabilities
- Real-time ML system deployment
- Quantifiable business impact across all projects

**What specific achievements or technical competencies would you like to explore?**

**Contact**: pgvenieris@outlook.com
        """
    
    def get_welcome_response(self):
        data = self.resume_data.data
        return f"""
## 👋 Welcome to Petros Venieris's Data Science Portfolio

**🎓 MS Analytics Candidate | Georgia Institute of Technology**  
*3.91/4.0 GPA (Top 5% of Graduate Cohort) | Graduating August 2025*

### 🏆 Core Technical Competencies
**Machine Learning & Predictive Analytics Specialist** with proven expertise in:
- **Time-Series Forecasting**: Prophet, ARIMA/SARIMA with 6% MAPE performance
- **Predictive Modeling**: XGBoost, Random Forest, ensemble methods (0.732 ROC-AUC)
- **Production ML Systems**: 90% accuracy in real-time environments (1,000+ predictions)
- **Feature Engineering**: Advanced behavioral and technical pattern analysis
- **Model Deployment**: End-to-end MLOps with <100ms latency requirements

### 🔬 Elite Research & Industry Experience
**Sandia National Laboratories Research Partnership** *(Current)*
- Developing predictive maintenance algorithms for critical infrastructure
- Behavioral vs. technical failure analysis with 1.1M+ data points
- Potential research publication with national laboratory scientists

### 📊 Quantifiable Business Impact
- **0.732 ROC-AUC** - Predictive maintenance models (20% cost reduction potential)
- **90% Accuracy** - Production sports prediction system (1,000+ matches validated)
- **6% MAPE** - Real estate forecasting with 12-15% portfolio optimization
- **Top 5%** - Academic performance in elite graduate program

**What specific aspect of my Data Science and ML expertise would you like to explore?**

**Contact**: pgvenieris@outlook.com
        """

    def get_help_response(self):
        return """
## 🎯 Explore Petros's Data Science & ML Expertise

### **🤖 Machine Learning & AI**
- "Tell me about his predictive modeling achievements"
- "What advanced ML algorithms does he specialize in?"
- "Show me his time-series forecasting performance"
- "How does he approach ensemble methods and feature engineering?"

### **📊 High-Impact Data Science Projects**
- "Walk me through his Sandia Labs research partnership"
- "Explain his real estate forecasting methodology and results"
- "What's his sports analytics system architecture?"
- "Show me his quantifiable business impact metrics"

### **🛠️ Technical Skills & Tools**
- "What's his Python/R/SQL proficiency level?"
- "How experienced is he with TensorFlow and XGBoost?"
- "Tell me about his ETL pipeline and data engineering skills"
- "What production ML deployment experience does he have?"

### **🎓 Academic Excellence & Research**
- "How does his 3.91 GPA compare in Georgia Tech's program?"
- "What advanced coursework has he completed?"
- "Tell me about his research publication potential"

### **💼 Industry Readiness**
- "How does he compare to other data science candidates?"
- "What makes him stand out for ML engineering roles?"
- "Show me his technical assessment readiness"

**Ask me anything about his technical skills, quantifiable achievements, or project experience!**

**Contact**: pgvenieris@outlook.com
        """

    def get_default_response(self):
        return """
## 🎯 Petros Venieris - Data Scientist & ML Engineer

### 🎓 **Elite Academic Foundation**
**Georgia Institute of Technology** - MS Analytics (3.91/4.0 GPA - Top 5% of Cohort)
- Advanced Machine Learning, Deep Learning, Statistical Methods
- Time-Series Analysis, Optimization, AI-driven Analytics
- Specialized in predictive modeling and production ML systems

### 💼 **High-Impact Professional Experience**
**Sandia National Laboratories** - Research Partnership *(Current)*
- Predictive maintenance modeling with behavioral analysis
- 0.732 ROC-AUC performance with 20% cost reduction potential
- 1.1M+ data points, 533 engineered features

**BetMax Sports Analytics** - Co-Founder & Lead Data Scientist
- 90% accuracy in live football match predictions
- Real-time ML systems with <3s latency (1,000+ matches)
- Automated ETL pipelines and notification systems

### 🚀 **Core Technical Expertise**
**Programming & ML**: Python, R, SQL, Scikit-learn, XGBoost, TensorFlow, Prophet  
**Specializations**: Time-series forecasting, Predictive modeling, Feature engineering, Model deployment, Statistical analysis, ETL pipelines

### 📈 **Quantifiable Achievements**
- **0.732 ROC-AUC** in predictive maintenance with industrial impact
- **90% accuracy** in production ML systems (validated across 1,000+ predictions)
- **6% MAPE** in real estate forecasting with portfolio optimization
- **Top 5% academic performance** in elite graduate program

**Ready to discuss how my proven Data Science and ML expertise can drive measurable results for your organization.**

**Contact**: pgvenieris@outlook.com
        """

    def get_section_response(self, section):
        """Format response for specific resume sections"""
        data = self.resume_data.data
        
        if section == "summary":
            return f"""
## 🎯 Professional Summary

{data['summary']}

### 📍 Contact Information
- **Location**: {data['personal_info']['location']}
- **Email**: pgvenieris@outlook.com
- **LinkedIn**: [Profile]({data['personal_info']['linkedin']})
- **GitHub**: [Portfolio]({data['personal_info']['github']})

### 🎯 Core Value Proposition
Elite Data Scientist combining rigorous academic training (Top 5% at Georgia Tech) with hands-on industry experience in predictive analytics, time-series forecasting, and production ML systems. Proven ability to deliver quantifiable business impact through advanced machine learning methodologies and statistical analysis.
            """
        
        elif section == "contact":
            info = data['personal_info']
            return f"""
## 📞 Contact Information

**{info['name']}**  
*{info['title']}*

📧 **Email**: pgvenieris@outlook.com  
🔗 **LinkedIn**: {info['linkedin']}  
💻 **GitHub**: {info['github']}  
📍 **Location**: {info['location']}

### 🚀 Ready to Connect
Available for data science and machine learning opportunities. Open to discussing how my predictive modeling expertise, research background, and Georgia Tech training can contribute to your data-driven initiatives and business objectives.
            """
        
        elif section == "experience":
            response = "## 💼 Professional Experience\n\n"
            for exp in data['experience']:
                response += f"### **{exp['role']}**\n"
                response += f"**{exp['company']}** | *{exp['period']}*\n\n"
                
                for highlight in exp['highlights']:
                    response += f"• {highlight}\n"
                
                if 'metrics' in exp:
                    response += "\n**Key Performance Metrics:**\n"
                    for metric in exp['metrics']:
                        response += f"📈 {metric}\n"
                
                response += f"\n**Technical Stack**: {', '.join(exp['tech_stack'])}\n\n"
                response += "---\n\n"
            
            response += "**Contact**: pgvenieris@outlook.com"
            return response
        
        elif section == "projects":
            response = "## 🚀 Data Science & ML Projects\n\n"
            for i, project in enumerate(data['projects'], 1):
                response += f"### **{i}. {project['name']}**\n"
                response += f"*{project['period']}*\n"
                if 'github' in project:
                    response += f"🔗 [View Project]({project['github']})\n"
                response += f"\n{project['description']}\n\n"
                
                response += "**Key Technical Achievements:**\n"
                for achievement in project['achievements']:
                    response += f"• {achievement}\n"
                
                response += f"\n**Technology Stack**: {', '.join(project['tech_stack'])}\n\n"
                response += "---\n\n"
            
            response += "**Contact**: pgvenieris@outlook.com"
            return response
        
        elif section == "skills":
            response = "## 🛠️ Technical Skills & Expertise\n\n"
            for category, skills in data['skills'].items():
                response += f"### **{category}**\n"
                response += f"{', '.join(skills)}\n\n"
            
            response += "**Contact**: pgvenieris@outlook.com"
            return response
        
        elif section == "achievements":
            response = "## 🏆 Key Achievements & Technical Highlights\n\n"
            for achievement in data['technical_highlights']:
                response += f"• {achievement}\n\n"
            
            response += "**Contact**: pgvenieris@outlook.com"
            return response
        
        elif section == "education":
            response = "## 🎓 Education & Academic Excellence\n\n"
            for edu in data['education']:
                response += f"### **{edu['degree']}**\n"
                response += f"**{edu['school']}** | *{edu['period']}*\n"
                response += f"📍 {edu['location']} | **GPA: {edu['gpa']}**\n\n"
                
                if 'ranking' in edu:
                    response += f"{edu['ranking']}\n\n"
                
                if 'program_rigor' in edu:
                    response += f"**Program Excellence:**\n{edu['program_rigor']}\n\n"
                
                if 'coursework' in edu:
                    response += "**Advanced Coursework:**\n"
                    for course in edu['coursework']:
                        response += f"• {course}\n"
                
                response += "\n---\n\n"
            
            response += "**Contact**: pgvenieris@outlook.com"
            return response
        
        return self.get_default_response()

    def analyze_skill_match(self, job_description: str) -> Dict:
        """Compare job requirements with candidate skills"""
        candidate_skills = set()
        for category, skills in self.resume_data.data['skills'].items():
            candidate_skills.update([s.lower() for s in skills])
        required_skills = re.findall(r'\b(\w+)\b', job_description.lower())
        matched_skills = [skill for skill in required_skills if skill in candidate_skills]
        match_percent = min(95, int(len(matched_skills) / max(1, len(required_skills)) * 100))
        return {
            "match_percent": match_percent,
            "matched_skills": matched_skills[:10],
            "missing_skills": []
        }


class PreMadeResponses:
    """Pre-written responses for sidebar navigation"""
    
    @staticmethod
    def get_professional_summary():
        response = """## 🎯 Professional Summary - Data Scientist & AI Engineer

**MS Analytics student at Georgia Tech (GPA: 3.91/4.0)** with proven expertise in **data science, machine learning, and AI engineering**. Specialized in predictive modeling, real-time data processing, and deploying production ML systems with recent focus on LLM integration.

### 🏆 **Key Highlights**
- **Academic Excellence**: Top 5% at Georgia Tech's #9 globally ranked Data Science program
- **Production AI Systems**: 90% accuracy BetMax sports analytics, 98% housing price prediction, GPT-4 powered resume agent
- **AI Engineering**: OpenAI API integration, conversational AI development, intelligent response systems
- **Research Impact**: Sandia Labs predictive maintenance with 500+ sensor features

### 🚀 **Technical Focus Areas**
- **Data Science**: Time-series forecasting, predictive analytics, ETL pipelines
- **Machine Learning**: XGBoost, Prophet, scikit-learn, hyperparameter optimization  
- **AI Engineering**: OpenAI API integration, conversational AI, intelligent agents, LLM applications
- **Data Analysis**: Statistical modeling, A/B testing, Power BI dashboards

**Contact**: pgvenieris@outlook.com | [LinkedIn](https://linkedin.com/in/petrosvenieris) | [GitHub](https://github.com/venie1)
"""
        
        # Add supporting visualization
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        
        st.markdown(response)
        
        # Key achievements chart
        achievements = pd.DataFrame({
            'Achievement': ['Georgia Tech GPA', 'BetMax Accuracy', 'Housing Model', 'AI Agent'],
            'Score': [3.91, 90, 98, 95],
            'Category': ['Academic', 'Sports Analytics', 'Real Estate', 'AI Engineering']
        })
        
        fig = px.bar(achievements, x='Achievement', y='Score', color='Category',
                    title='Key Performance Metrics',
                    color_discrete_map={'Academic': '#2196f3', 'Sports Analytics': '#4caf50', 
                                      'Real Estate': '#ff9800', 'AI Engineering': '#9c27b0'})
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        return response

    @staticmethod
    def get_academic_excellence():
        return """
## 🎓 Academic Excellence & Educational Achievements

### 🏆 **Georgia Institute of Technology - Master of Science in Analytics**
**GPA: 3.91/4.0 | Top 5% of Graduate Cohort | Expected Graduation: August 2025**

#### **🌟 Program Distinction & Global Recognition**
- **#9 Globally Ranked** Data Science Program
- **#4 in United States** for Analytics Education
- **Highly Selective**: ~15% acceptance rate from global applicant pool
- **Elite Curriculum**: 36 credit hours of intensive, industry-aligned coursework

#### **📚 Advanced Technical Coursework**
- **Machine Learning & Deep Learning**: Advanced algorithms, neural networks, ensemble methods
- **Time-Series Analysis & Forecasting**: ARIMA/SARIMA, Prophet, econometric modeling
- **Statistical Methods & Hypothesis Testing**: Bayesian inference, experimental design
- **Optimization & Operations Research**: Linear/nonlinear programming, constraint optimization
- **AI-Driven Analytics**: Reinforcement learning, natural language processing
- **High-Performance Computing**: Distributed computing, parallel processing for analytics

#### **🎯 Academic Performance Context**
- **3.91/4.0 GPA** represents mastery of highly challenging technical content
- **Top 5% ranking** among exceptionally qualified international student body
- **Consistent Excellence**: Superior performance across all core technical subjects
- **Research Recognition**: Selected for prestigious Sandia National Laboratories practicum

#### **🔬 Industry Integration**
- **Sandia Labs Partnership**: Real-world research on critical infrastructure AI
- **Faculty Excellence**: World-renowned professors from top tech companies and research institutions
- **Peer Network**: Classmates from FAANG companies and premier global universities
- **Industry Relevance**: Curriculum designed with input from leading data science practitioners

### 🎓 **University of Piraeus - Bachelor of Science in Digital Systems**
**GPA: 7.4/10.0 | September 2018 - July 2022**

#### **🚀 Technical Foundation & Capstone Excellence**
- **Strong Programming Base**: Software engineering, data structures, algorithms
- **Systems Knowledge**: Database systems, computer networks, distributed systems
- **Thesis Achievement**: COVID-19 forecasting application - **Grade: 10/10**
- **Full-Stack Development**: End-to-end application development and deployment

### 📈 **Academic Impact & Recognition**
- **Research Potential**: Publication discussions with Sandia scientists and Georgia Tech faculty
- **Technical Leadership**: Consistent demonstration of advanced problem-solving capabilities
- **International Excellence**: Recognition across two premier educational systems
- **Continuous Learning**: Commitment to staying at forefront of ML/AI developments

**This academic foundation provides the theoretical depth and practical skills essential for leading data science initiatives in competitive, technology-driven environments.**

**Contact**: pgvenieris@outlook.com
        """

    @staticmethod
    def get_work_experience():
        response = """## 💼 Professional Experience

### 🏭 **Sandia National Laboratories - Data Scientist Practicum**
**May-Aug 2025 | Research Partnership**
- **Project**: Predictive maintenance with 500+ sensor features
- **Innovation**: Behavioral + technical failure analysis 
- **Performance**: Cost-sensitive optimization, operational resilience
- **Scale**: Multi-strategy data cleaning framework

### ⚽ **BetMax - Sports Data Scientist & Co-Founder**
**Jan 2019 - Dec 2022 | Self-Employed Partnership**
- **Achievement**: 90% accuracy across 1,000+ football matches
- **Technology**: Real-time data processing every 15 minutes
- **System**: Automated prediction pipeline with email alerts
- **Impact**: Consistent ROI through analytics-driven strategies

### 🏪 **Intersport - Sales Associate**
**Jan 2021 - Jan 2023 | Customer Service**
- **Skills**: Customer service, product recommendations
- **Balance**: Part-time work with full-time university coursework
- **Development**: Strong communication and interpersonal skills

### 🎯 **AI Engineering Experience**
- **AI Resume Agent**: OpenAI integration, conversational AI development
- **Technical Stack**: Python, Streamlit, API integration, session management
- **Focus**: Intelligent automation, natural language processing
"""
        
        # Add supporting visualization
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        
        st.markdown(response)
        
        # Experience timeline chart
        experience_data = pd.DataFrame({
            'Role': ['AI Projects', 'Sandia Labs', 'BetMax Analytics', 'Intersport'],
            'Duration_Months': [12, 3, 48, 24],
            'Type': ['AI Engineering', 'Research', 'Data Science', 'Customer Service']
        })
        
        fig = px.bar(experience_data, x='Role', y='Duration_Months', color='Type',
                    title='Professional Experience Timeline (Months)',
                    color_discrete_map={'AI Engineering': '#9c27b0', 'Research': '#2196f3',
                                      'Data Science': '#4caf50', 'Customer Service': '#ff9800'})
        fig.update_layout(height=300, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        return response

    @staticmethod
    def get_data_science_projects():
        response = """## 🚀 Data Science & AI Project Portfolio

### 🤖 **AI Resume Agent** (Latest - LLM Integration Project)
**Interactive Conversational AI System | 2025 - Current**
- **LLM Engineering**: OpenAI GPT-4 API integration with advanced prompt engineering
- **AI Architecture**: Conversational AI interface, intelligent response system, session management
- **Technology Stack**: Python, Streamlit, OpenAI API, Plotly, advanced state management
- **Features**: Real-time chat interface, dynamic visualizations, professional UI/UX design
- **GitHub**: [ResumeAgent Repository](https://github.com/venie1/ResumeAgent)
- **Impact**: Demonstrates practical LLM application development and AI engineering skills

### 🏭 **Sandia National Laboratories - Predictive Maintenance**
**Data Scientist Research Practicum | May-Aug 2025**
- **Scale**: 500+ sensor features, 1.1M+ data points from 23,550 vehicles
- **Performance**: 0.732 ROC-AUC, 20% cost reduction potential
- **Innovation**: Behavioral + technical failure analysis, production ML systems

### ⚽ **BetMax Sports Analytics**
**Lead Data Scientist | Jan 2019 - Dec 2022**
- **Performance**: 90% accuracy across 1,000+ football matches
- **Real-time**: 15-minute data processing, automated prediction system
- **GitHub**: [BetPredictor](https://github.com/venie1/BetPredictor)

### 🏠 **Real Estate Investment Advisor**
**ML Engineer | July 2025**
- **Performance**: 98% explained variance, 12-15% portfolio return uplift
- **Technology**: Prophet forecasting, Power BI dashboards, ETL pipelines
- **GitHub**: [RealEstateAdvisor](https://github.com/venie1/RealEstateAdvisor)

### 🦠 **COVID-19 Forecasting Application**
**Bachelor Thesis | 2022 - Grade: 10/10**
- **Achievement**: 15% RMSE reduction, Flask web application
- **Models**: Prophet, ARIMA/SARIMA, SVM benchmarking
"""
        
        # Add supporting visualization
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        
        st.markdown(response)
        
        # Projects performance chart
        projects_data = pd.DataFrame({
            'Project': ['AI Resume Agent', 'Sandia Labs', 'BetMax Sports', 'Real Estate', 'COVID Forecasting'],
            'Performance': [95, 92, 90, 98, 85],
            'Category': ['AI Engineering', 'Research', 'Sports Analytics', 'Real Estate', 'Healthcare']
        })
        
        fig = px.bar(projects_data, x='Project', y='Performance', color='Category',
                    title='Project Performance Metrics (%)',
                    color_discrete_map={'AI Engineering': '#9c27b0', 'Research': '#2196f3',
                                      'Sports Analytics': '#4caf50', 'Real Estate': '#ff9800', 'Healthcare': '#f44336'})
        fig.update_layout(height=300, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        return response

    @staticmethod
    def get_technical_skills():
        response = """## 🛠️ Technical Skills Portfolio

### 🚀 **Data Science & AI Engineering** (Primary Focus)
- **Python**: pandas, NumPy, scikit-learn, XGBoost, Prophet - *production ML systems*
- **Machine Learning**: Time-series forecasting, predictive modeling, ensemble methods
- **AI Engineering**: OpenAI API integration, prompt engineering, conversational AI
- **LLM Integration**: GPT-4 API, intelligent response systems, session management
- **Data Analysis**: Statistical modeling, A/B testing, hypothesis testing

### 💻 **Core Technologies**
- **Programming**: Python, R, SQL, Flask - *4+ years experience*
- **ML Frameworks**: scikit-learn, XGBoost, Prophet, OpenAI API
- **Visualization**: Power BI, Plotly, matplotlib - *interactive dashboards*
- **Development**: Git, Docker, ETL pipelines, API integration, Streamlit

### 🎯 **Applied Projects**
- **AI Resume Agent**: OpenAI GPT integration, Streamlit deployment, conversational AI
- **BetMax Analytics**: 90% accuracy sports prediction, real-time data processing
- **Real Estate AI**: 98% explained variance, Prophet forecasting, Power BI dashboards
- **Sandia Research**: 500+ sensor features, predictive maintenance, cost optimization
"""
        
        # Add supporting visualization
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        
        st.markdown(response)
        
        # Skills experience chart (years and projects)
        skills_data = pd.DataFrame({
            'Skill': ['Python/ML', 'Deep Learning', 'LLM/NLP', 'Time-Series', 'AI Engineering'],
            'Experience_Years': [4, 2, 1, 3, 1],
            'Projects_Count': [8, 3, 2, 4, 2],
            'Focus_Area': ['Core', 'AI', 'AI', 'Analysis', 'AI']
        })
        
        fig = px.scatter(skills_data, x='Experience_Years', y='Projects_Count', 
                        size='Projects_Count', color='Focus_Area', hover_name='Skill',
                        title='Technical Skills: Experience & Project Application',
                        labels={'Experience_Years': 'Years of Experience', 'Projects_Count': 'Number of Projects'},
                        color_discrete_map={'Core': '#2196f3', 'AI': '#9c27b0', 'Analysis': '#4caf50'})
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        return response

    @staticmethod
    def get_key_achievements():
        response = """## 🏆 Key Achievements & Performance Excellence

### 🎓 **Academic Excellence**
- **Georgia Tech GPA**: 3.91/4.0 (Top 5% in #9 globally ranked Data Science program)
- **Research Partnership**: Selected for prestigious Sandia National Laboratories practicum
- **Bachelor Thesis**: Perfect 10/10 grade for COVID-19 forecasting application

### 🔬 **Technical Achievements**
- **AI Resume Agent**: OpenAI GPT integration, conversational AI system development
- **Sandia Research**: 0.732 ROC-AUC, 20% cost reduction potential, 500+ sensor features
- **BetMax Analytics**: 90% accuracy across 1,000+ football matches, real-time processing
- **Real Estate ML**: 98% explained variance, 12-15% portfolio return uplift

### 🚀 **Innovation & Impact**
- **AI Engineering**: Advanced prompt engineering, session management, intelligent automation
- **Production Systems**: 4+ years maintaining high-accuracy ML systems
- **Research Excellence**: Behavioral + technical failure analysis innovation
- **Open Source**: Multiple repositories with educational and practical value

### 💼 **Professional Recognition**
- **Elite Partnership**: National laboratory collaboration opportunity
- **Technical Leadership**: Co-founded successful analytics venture
- **Academic Performance**: Consistent excellence across rigorous coursework
- **Industry Impact**: Quantifiable ROI and cost reduction across projects
"""
        
        # Add supporting visualization
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        
        st.markdown(response)
        
        # Achievements metrics chart
        achievements_data = pd.DataFrame({
            'Achievement': ['Academic GPA', 'AI Engineering', 'Sandia Research', 'BetMax System', 'Real Estate ML'],
            'Impact_Score': [95, 93, 92, 90, 98],
            'Category': ['Academic', 'AI Engineering', 'Research', 'Sports Analytics', 'Real Estate']
        })
        
        fig = px.bar(achievements_data, x='Achievement', y='Impact_Score', color='Category',
                    title='Key Achievement Impact Scores',
                    color_discrete_map={'Academic': '#2196f3', 'AI Engineering': '#9c27b0',
                                      'Research': '#4caf50', 'Sports Analytics': '#ff9800', 'Real Estate': '#f44336'})
        fig.update_layout(height=300, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        return response

    @staticmethod
    def get_contact_information():
        response = """## 📞 Contact Information

### 👤 **Petros Venieris**
*Data Scientist & AI Engineer | MS Analytics @ Georgia Tech*

### 📧 **Direct Contact**
- **Email**: pgvenieris@outlook.com
- **Location**: Athens, Greece / Atlanta, GA
- **Availability**: Immediate for opportunities post-August 2025
- **Response Time**: Within 24 hours for professional inquiries

### 🔗 **Professional Profiles**
- **LinkedIn**: [linkedin.com/in/petrosvenieris](https://linkedin.com/in/petrosvenieris)
- **GitHub**: [github.com/venie1](https://github.com/venie1)
- **Portfolio**: Interactive AI Resume Agent (this application)

### 🎯 **Open to Opportunities**
- **Data Science** positions with ML/AI focus
- **AI Engineering** roles involving OpenAI/LLM integration
- **Research** collaborations in predictive analytics
- **Full-time** opportunities starting September 2025
"""
        
        # Add supporting visualization
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        
        st.markdown(response)
        
        # Availability timeline chart
        availability_data = pd.DataFrame({
            'Period': ['Jan-Apr 2025', 'May-Aug 2025', 'Sep 2025+'],
            'Status': ['Coursework', 'Sandia Research', 'Available'],
            'Availability': [20, 50, 100]
        })
        
        fig = px.bar(availability_data, x='Period', y='Availability', color='Status',
                    title='Professional Availability Timeline (%)',
                    color_discrete_map={'Coursework': '#ff9800', 'Sandia Research': '#4caf50', 'Available': '#2196f3'})
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        return response

class GeorgaTechShowcase:
    """Showcase Georgia Tech program rigor and reputation"""
    
    @staticmethod
    def create_program_highlights():
        """Create Georgia Tech program showcase"""
        return {
            "ranking_info": {
                "global_ranking": "#9 Globally Ranked Data Science Program",
                "us_ranking": "#4 in United States for Analytics",
                "acceptance_rate": "Highly Selective (~15% acceptance rate)",
                "industry_reputation": "Elite industry partnerships with Google, Microsoft, Meta, Amazon"
            },
            "program_rigor": {
                "credit_hours": "36 Credit Hours of Advanced Coursework",
                "core_requirements": "Machine Learning, Deep Learning, Statistical Methods, Optimization",
                "capstone": "Industry Practicum (Sandia National Laboratories Partnership)",
                "gpa_significance": "3.91/4.0 GPA represents Top 5% performance in rigorous curriculum"
            },
            "faculty_excellence": [
                "World-renowned faculty from top tech companies and research institutions",
                "Active researchers in cutting-edge ML, AI, and analytics domains",
                "Industry practitioners bringing real-world data science expertise"
            ],
            "peer_quality": [
                "Classmates from FAANG companies (Google, Amazon, Microsoft, Meta)",
                "International students from premier global universities",
                "Working professionals with 5+ years industry experience in data science"
            ]
        }
    
    @staticmethod
    def display_program_showcase():
        """Display Georgia Tech program information for recruiters"""
        program_info = GeorgaTechShowcase.create_program_highlights()
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #003057 0%, #B3A369 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        ">
            <h2>🏆 Georgia Institute of Technology</h2>
            <h3>Master of Science in Analytics</h3>
            <h4>#9 Globally Ranked Data Science Program</h4>
            <p style="font-size: 1.1rem; margin-top: 1rem;">3.91/4.0 GPA - Top 5% of Graduate Cohort</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Program Rankings & Elite Status")
            for key, value in program_info["ranking_info"].items():
                st.markdown(f"• **{key.replace('_', ' ').title()}**: {value}")
            
            st.markdown("### 🎓 Academic Rigor & Excellence")
            for key, value in program_info["program_rigor"].items():
                st.markdown(f"• **{key.replace('_', ' ').title()}**: {value}")
        
        with col2:
            st.markdown("### 👨‍🏫 Faculty Excellence")
            for point in program_info["faculty_excellence"]:
                st.markdown(f"• {point}")
            
            st.markdown("### 👥 Peer Quality & Networking")
            for point in program_info["peer_quality"]:
                st.markdown(f"• {point}")

class ResumeAssistantChatbot:
    """Professional Resume Assistant focused on ML/DS expertise"""
    
    def __init__(self):
        self.config = ChatbotConfig()
        self.resume_data = ResumeData()
        self.intelligent_response = IntelligentResponseSystem(self.resume_data, self.config)
        self.conversation_history = []
        
        self.main_menu = {
            "👤 Professional Summary": "summary",
            "🎓 Education & Academic Excellence": "education",
            "💼 Professional Experience": "experience", 
            "🚀 Data Science Projects": "projects",
            "🛠️ Technical Skills": "skills",
            "🏆 Key Achievements": "achievements",
            "📞 Contact Information": "contact"
        }
    
    def get_response(self, user_input: str) -> str:
        """Get intelligent response with skill matching"""
        if "fit for" in user_input.lower() or "match for" in user_input.lower():
            job_desc = user_input.split(":", 1)[-1].strip()
            analysis = self.intelligent_response.analyze_skill_match(job_desc)
            response = f"""
## 🎯 Skill Match Analysis
**Match Score: {analysis['match_percent']}%**  
Petros is exceptionally well-qualified for this data science role!

### ✅ Matching Technical Skills:
{' • '.join(analysis['matched_skills'])}

### 💡 Why Petros is an outstanding candidate:
1. **Elite Academic Foundation**: Georgia Tech MS Analytics (Top 5% with 3.91 GPA)
2. **Proven Production Experience**: 90% accuracy in live ML systems
3. **Research Excellence**: Sandia Labs partnership with publication potential
4. **Quantifiable Impact**: 0.732 ROC-AUC, 20% cost reduction
5. **Full-Stack Capabilities**: Research, development, deployment, and optimization

**Recommendation: Priority candidate for immediate interview scheduling**

**Contact**: pgvenieris@outlook.com
            """
            return response
        
        try:
            response = self.intelligent_response.generate_intelligent_response(
                user_input, 
                self.conversation_history
            )
            
            self.conversation_history.append({
                "user": user_input,
                "assistant": response
            })
            
            return response
        except Exception as e:
            print(f"Response error: {e}")
            return f"I apologize for the technical issue. Here's a summary of Petros's expertise: {self.resume_data.data['summary']}\n\n**Contact**: pgvenieris@outlook.com"

    def get_section_response(self, section: str) -> str:
        """Get formatted response for a specific section"""
        return self.intelligent_response.get_section_response(section)

def main():
    """Main Streamlit application function"""
    
    st.set_page_config(
        page_title="Petros Venieris - Data Scientist & ML Engineer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS styling for the app
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Root Variables for Consistent Theme */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-gradient: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --text-primary: #ffffff;
        --text-secondary: #e0e6ed;
        --text-accent: #a0aec0;
        --shadow-glass: 0 8px 32px rgba(0, 0, 0, 0.3);
        --shadow-hover: 0 20px 40px rgba(0, 0, 0, 0.4);
        --sidebar-gradient: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    }
    /* Profile photo styling */
    .stImage img {
        border-radius: 50%;
        border: 3px solid rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }

    .stImage img:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
    }
    /* Animated Background */
    body, .main, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background: var(--dark-gradient);
        color: var(--text-primary);
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }
    
    /* Animated Background Overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
        animation: backgroundFlow 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes backgroundFlow {
        0%, 100% {
            transform: translateX(0%) translateY(0%) rotate(0deg);
            opacity: 0.3;
        }
        25% {
            transform: translateX(5%) translateY(-10%) rotate(90deg);
            opacity: 0.4;
        }
        50% {
            transform: translateX(-5%) translateY(-5%) rotate(180deg);
            opacity: 0.3;
        }
        75% {
            transform: translateX(-10%) translateY(10%) rotate(270deg);
            opacity: 0.4;
        }
    }
    
    /* Enhanced Typography with Better Contrast */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 700;
        color: var(--text-primary);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        line-height: 1.2;
    }
    
    p, span, div, li {
        color: var(--text-secondary);
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Glassmorphism Base */
    .glass-container {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        box-shadow: var(--shadow-glass);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-container:hover {
        background: rgba(255, 255, 255, 0.15);
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    /* Auto-scroll and chat positioning */
    #chat-section {
        scroll-margin-top: 100px;
    }
    
    .main .block-container {
        padding: 2rem 1rem 5rem 1rem;
        max-width: 1400px;
    }
    
    /* Enhanced Chat Container with Glassmorphism */
    .chat-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 2px solid transparent;
        background-clip: padding-box;
        border-radius: 25px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        animation: chatPulse 3s ease-in-out infinite;
        overflow: hidden;
    }
    
    .chat-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--primary-gradient);
        opacity: 0.1;
        border-radius: 25px;
    }
    
    @keyframes chatPulse {
        0%, 100% { 
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1),
                0 0 0 0 rgba(102, 126, 234, 0.4);
        }
        50% { 
            box-shadow: 
                0 12px 40px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2),
                0 0 0 8px rgba(102, 126, 234, 0.1);
        }
    }
    
    /* Enhanced Chat Header */
    .chat-header {
        background: var(--primary-gradient);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 
            0 8px 25px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        animation: headerGlow 4s ease-in-out infinite alternate;
        position: relative;
        overflow: hidden;
    }
    
    .chat-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: headerShimmer 8s linear infinite;
    }
    
    @keyframes headerGlow {
        from { 
            box-shadow: 
                0 8px 25px rgba(102, 126, 234, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        to { 
            box-shadow: 
                0 12px 35px rgba(102, 126, 234, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
    }
    
    @keyframes headerShimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Project Cards with Glassmorphism */
    .project-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        padding: 2.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: cardFadeIn 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .project-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .project-card:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            0 0 0 1px rgba(102, 126, 234, 0.3);
        transform: translateY(-8px) scale(1.02);
    }
    
    .project-card:hover::before {
        left: 100%;
    }
    
    @keyframes cardFadeIn {
        from { 
            opacity: 0; 
            transform: translateY(30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    /* Profile photo styling */
    .stImage > img {
        border-radius: 50% !important;
        border: 4px solid rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        object-fit: cover !important;
    }

    .stImage > img:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
    }

    /* Mobile responsiveness for hero section */
    @media (max-width: 768px) {
        .hero-section h1 {
            font-size: 2rem !important;
        }
        .hero-section h2 {
            font-size: 1.3rem !important;
        }
        .hero-section h3 {
            font-size: 1.1rem !important;
        }
    }
    /* Enhanced Skill Badges */
    .skill-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: var(--text-primary);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 0.5em 1em;
        margin: 0.3em 0.4em 0.3em 0;
        font-size: 0.9em;
        font-weight: 500;
        vertical-align: middle;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .skill-badge:hover {
        background: var(--primary-gradient);
        color: white;
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border-color: transparent;
    }
    
    /* Enhanced Metric Cards with Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--accent-gradient);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.12);
        transform: translateY(-6px);
        box-shadow: 
            0 15px 45px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card h4 {
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .metric-card h3 {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    /* Enhanced Tab Content */
    .tab-content {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Enhanced Hero Section */
    .hero-section {
        background: var(--primary-gradient);
        color: white;
        text-align: center;
        padding: 3rem 2rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        box-shadow: 
            0 15px 50px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        animation: heroFloat 6s ease-in-out infinite;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: heroShimmer 12s linear infinite;
    }
    
    @keyframes heroFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    @keyframes heroShimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .hero-section h1, .hero-section h2, .hero-section h3 {
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 2;
    }
    
    /* Enhanced Professional Section */
    .professional-section {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .professional-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-gradient);
        border-radius: 20px 20px 0 0;
    }
    
    /* Enhanced Section Titles */
    .section-title {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        text-align: center;
        position: relative;
        padding-bottom: 1rem;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 3px;
        background: var(--accent-gradient);
        border-radius: 3px;
    }
    
    .section-description {
        font-size: 1.1rem;
        color: var(--text-secondary);
        line-height: 1.7;
        margin-bottom: 2rem;
        text-align: center;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    /* Enhanced Recruiter Note */
    .recruiter-note {
        background: rgba(79, 172, 254, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-left: 4px solid #4facfe;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.2);
        position: relative;
    }
    
    .recruiter-note-content {
        margin: 0;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Enhanced Quick Prompt Section */
    .quick-prompt-section {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .quick-prompt-title {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .quick-prompts-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .quick-prompt-item {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 0.95rem;
        color: var(--text-secondary);
        text-align: left;
        position: relative;
        overflow: hidden;
    }
    
    .quick-prompt-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .quick-prompt-item:hover {
        background: var(--primary-gradient);
        color: white;
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
        border-color: transparent;
    }
    
    .quick-prompt-item:hover::before {
        left: 100%;
    }
    
    /* Enhanced Sidebar Styling with proper dark theme */
    .stSidebar > div:first-child {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: inset 0 0 50px rgba(102, 126, 234, 0.1);
    }

    .stSidebar .stMarkdown {
        color: #e0e6ed;
    }

    .stSidebar h3 {
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }

    .stSidebar .stButton > button {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.4);
        color: #e0e6ed;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .stSidebar .stButton > button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: transparent;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .stSidebar .stToggle > label {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.4);
    }

    .stSidebar .stMetric {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }

    .stSidebar .stMetric [data-testid="metric-container"] > div:first-child {
        color: #ffffff;
        font-weight: 600;
    }

    .stSidebar .stMetric [data-testid="metric-container"] > div:nth-child(2) {
        color: #4facfe;
        font-weight: 700;
    }

    .stSidebar .stMetric [data-testid="metric-container"] > div:last-child {
        color: #00f2fe;
        font-size: 0.9rem;
    }

    
    /* Enhanced Progress Bars */
    .stProgress > div > div > div {
        background: var(--accent-gradient);
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(79, 172, 254, 0.3);
    }
    
    .stProgress > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    /* Enhanced Scrollbar */
    ::-webkit-scrollbar { 
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track { 
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb { 
        background: var(--primary-gradient);
        border-radius: 4px; 
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-gradient);
    }
    
    /* Enhanced Chat Messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: var(--secondary-gradient);
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: var(--primary-gradient);
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: var(--text-primary);
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: var(--primary-gradient);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border-color: transparent;
    }
    
    /* Enhanced Input Fields */
    .stTextInput > div > div > input,
    .stChatInput > div > div > input {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: var(--text-primary);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stTextInput > div > div > input:focus,
    .stChatInput > div > div > input:focus {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary-gradient);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced Toggle */
    .stToggle > label {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
    }
    
    /* Enhanced Metrics */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric [data-testid="metric-container"] > div:first-child {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* Responsive Design */
    @media (max-width: 1024px) {
        .hero-section {
            padding: 2.5rem 1.5rem;
        }
        .professional-section {
            padding: 2rem;
        }
        .quick-prompts-grid {
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }
    }
    
    @media (max-width: 768px) {
        .project-card { 
            padding: 2rem; 
        }
        .section-title {
            font-size: 1.7rem;
        }
        .section-description {
            font-size: 1rem;
        }
        .hero-section {
            padding: 2rem 1rem;
        }
        .quick-prompts-grid {
            grid-template-columns: 1fr;
            gap: 0.8rem;
        }
    }
    
    @media (max-width: 480px) {
        .main .block-container {
            padding: 1rem 0.5rem 3rem 0.5rem;
        }
        .hero-section {
            padding: 1.5rem 1rem;
        }
        .professional-section {
            padding: 1.5rem;
        }
        .project-card {
            padding: 1.5rem;
        }
    }
    
    /* Enhanced Loading States */
    .stSpinner > div {
        border-color: rgba(102, 126, 234, 0.3) transparent rgba(102, 126, 234, 0.3) transparent;
        animation: spinnerGlow 1s linear infinite;
    }
    
    @keyframes spinnerGlow {
        0% { 
            border-color: rgba(102, 126, 234, 0.3) transparent rgba(102, 126, 234, 0.3) transparent;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.2);
        }
        50% { 
            border-color: rgba(102, 126, 234, 0.8) transparent rgba(102, 126, 234, 0.8) transparent;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
        }
        100% { 
            border-color: rgba(102, 126, 234, 0.3) transparent rgba(102, 126, 234, 0.3) transparent;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.2);
        }
    }
    
    /* Smooth scroll behavior */
    html {
        scroll-behavior: smooth;
    }
    
    /* Ensure visibility on all platforms */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    </style>
    """, unsafe_allow_html=True)

# Profile photo and header section
    col1, col2 = st.columns([1, 4])

    with col1:
        # Profile photo with professional styling
        try:
            st.markdown("""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 1rem 0;
            ">
            """, unsafe_allow_html=True)
            
            st.image("ph.jpg", width=150, use_container_width=False)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except:
            # Fallback if photo doesn't exist
            st.markdown("""
            <div style="
                width: 150px;
                height: 150px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 4rem;
                margin: 1rem auto;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
                border: 4px solid rgba(255, 255, 255, 0.3);
            ">🎯</div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="hero-section" style="padding: 2rem 1rem; margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.5rem; margin: 0; font-weight: 800; text-shadow: 0 3px 6px rgba(0,0,0,0.4);">Petros Venieris</h1>
            <h2 style="font-size: 1.5rem; margin: 0.5rem 0; font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                Data Scientist &amp; Machine Learning Engineer
            </h2>
            <h3 style="font-size: 1.2rem; margin: 0.5rem 0; font-weight: 500; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                MS Analytics | Georgia Institute of Technology | 3.91 GPA (Top 5% of Cohort)
            </h3>
            <div style="
                display: flex;
                flex-wrap: wrap;
                gap: 0.8rem;
                margin-top: 1.5rem;
            ">
                <span style="
                    background: rgba(255, 255, 255, 0.15);
                    padding: 0.4rem 0.8rem;
                    border-radius: 15px;
                    font-size: 0.9rem;
                    font-weight: 500;
                    backdrop-filter: blur(10px);
                    color: white;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                ">🏆 Sandia Labs Research</span>
                <span style="
                    background: rgba(255, 255, 255, 0.15);
                    padding: 0.4rem 0.8rem;
                    border-radius: 15px;
                    font-size: 0.9rem;
                    font-weight: 500;
                    backdrop-filter: blur(10px);
                    color: white;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                ">📊Master of Science in Analytics</span>
                <span style="
                    background: rgba(255, 255, 255, 0.15);
                    padding: 0.4rem 0.8rem;
                    border-radius: 15px;
                    font-size: 0.9rem;
                    font-weight: 500;
                    backdrop-filter: blur(10px);
                    color: white;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                ">🚀Impactful projects</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Initialize chatbot with scroll trigger
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ResumeAssistantChatbot()
        st.session_state.messages = []
        st.session_state.scroll_trigger = 0

    # Main sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 1.5rem; 
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.8), rgba(118, 75, 162, 0.8)); 
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            color: white; 
            border-radius: 15px; 
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        ">
            <h3 style="margin: 0; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🎯 Quick Navigation</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; opacity: 0.9; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">Click for instant answers ↓</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick question prompts with enhanced styling
        quick_questions = {
            "👤 Professional Summary": PreMadeResponses.get_professional_summary,
            "🎓 Academic Excellence": PreMadeResponses.get_academic_excellence,
            "💼 Work Experience": PreMadeResponses.get_work_experience,
            "🚀 Data Science Projects": PreMadeResponses.get_data_science_projects,
            "🛠️ Technical Skills": PreMadeResponses.get_technical_skills,
            "🏆 Key Achievements": PreMadeResponses.get_key_achievements,
            "📞 Contact Information": PreMadeResponses.get_contact_information
        }
        
        for i, (button_text, response_func) in enumerate(quick_questions.items()):
            unique_key = f"sidebar_q_{hash(button_text) % 10000}_{i}"
            if st.button(button_text, key=unique_key, use_container_width=True):
                question = f"Tell me about {button_text.lower()}"
                response = response_func()
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.scroll_trigger = time.time()
                # Force switch to AI Assistant tab and lock it
                st.session_state.active_tab_index = 0
                st.query_params["tab"] = "0"
                st.session_state.trigger_scroll = True
                st.session_state.force_ai_tab = True  # Add flag to force AI tab
                st.session_state.from_sidebar = True  # Track that this came from sidebar
                st.rerun()
        
        st.markdown("---")
        
        # Settings section
        st.markdown("### ⚙️ Assistant Settings")
        use_ai = st.toggle("🤖 AI-Enhanced Responses", value=st.session_state.chatbot.config.enable_openai)
        if use_ai != st.session_state.chatbot.config.enable_openai:
            st.session_state.chatbot.config.enable_openai = use_ai
            st.rerun()
        
        # 🧠 RAG Experimental Feature Toggle
        if RAG_AVAILABLE:
            use_rag = st.toggle("🧠 RAG Neural Search (Experimental)", 
                              value=st.session_state.chatbot.config.enable_rag,
                              help="🚀 Enable Vector Embeddings & Semantic Retrieval for enhanced AI responses")
            if use_rag != st.session_state.chatbot.config.enable_rag:
                st.session_state.chatbot.config.enable_rag = use_rag
                # Reinitialize RAG system if toggled on
                if use_rag and not st.session_state.chatbot.rag_system:
                    st.session_state.chatbot.rag_system = RAGEnhancedSystem(st.session_state.chatbot.resume_data)
                st.rerun()
        else:
            st.info("🧠 RAG features require: `pip install sentence-transformers faiss-cpu langchain PyPDF2 chromadb`")
        
        if st.button("🗑️ Clear Chat History", key="clear_chat_sidebar", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chatbot.conversation_history = []
            st.rerun()
        
        # Stats and achievements sidebar
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        st.metric("Academic Performance", "3.91/4.0", "Top 5%")
        st.metric("ML Accuracy", "90%", "+15% vs baseline")
        st.metric("Research Projects", "4", "Including Sandia")
        st.metric("Data Points", "1.1M+", "Processed")

    # Initialize active tab from query params or session state
    query_params = st.query_params
    if "tab" in query_params:
        try:
            active_tab_index = int(query_params["tab"])
            st.session_state.active_tab_index = active_tab_index
        except:
            st.session_state.active_tab_index = 0
    elif 'active_tab_index' not in st.session_state:
        st.session_state.active_tab_index = 0

    # Main tab navigation using radio buttons
    tab_names = [
        "🤖 AI Assistant", 
        "🎓 Georgia Tech Excellence",
        "🏭 Sandia Labs Research", 
        "🏠 Real Estate Advisor",
        "⚽ BetMax Analytics",
        "🚀 Project Portfolio",
        "📊 Skills & Performance"
    ]
    
    # Handle force_ai_tab flag - only force during initial redirect, not persistently
    is_forced = (st.session_state.get('force_ai_tab', False) or 
                st.session_state.get('from_sidebar', False))
    
    if is_forced:
        st.session_state.active_tab_index = 0
        st.query_params["tab"] = "0"
        selected_tab = 0
        # Clear flags immediately after setting the tab - don't persist
        st.session_state.force_ai_tab = False
        st.session_state.from_sidebar = False
    else:
        # Clear any remaining locks to ensure normal operation
        if 'sticky_ai_lock' in st.session_state:
            del st.session_state.sticky_ai_lock
        if 'lock_timestamp' in st.session_state:
            del st.session_state.lock_timestamp
    
    # Clever solution: Use different keys for forced vs normal tab selection
    if is_forced:
        # Force AI Assistant tab selection with a special key
        st.radio(
            "",
            options=range(len(tab_names)),
            format_func=lambda x: tab_names[x],
            index=0,  # Always show AI Assistant as selected when forcing
            horizontal=True,
            key="tab_selector_forced",  # Different key when forced
            label_visibility="collapsed"
        )
        selected_tab = 0  # Override any radio selection
    else:
        # Normal tab selection with regular key
        selected_tab = st.radio(
            "",
            options=range(len(tab_names)),
            format_func=lambda x: tab_names[x],
            index=st.session_state.active_tab_index,
            horizontal=True,
            key="tab_selector_normal",  # Different key when normal
            label_visibility="collapsed"
        )
        
        # Update session state and query params normally
        if selected_tab != st.session_state.active_tab_index:
            st.session_state.active_tab_index = selected_tab
            st.query_params["tab"] = str(selected_tab)
    
    # Display content based on selected tab
    if selected_tab == 0:  # AI Assistant
        # Add chat section anchor
        st.markdown('<div id="chat-section"></div>', unsafe_allow_html=True)
        
        # Handle scroll trigger when navigating to AI Assistant
        if st.session_state.get('trigger_scroll', False):
            st.session_state.trigger_scroll = False
            force_scroll_to_bottom()
        
        
        # AI chat assistant section
        st.markdown("""
        <div class="professional-section">
            <h2 class="section-title">AI-Powered Resume Assistant</h2>
            <p class="section-description">
                Discover Petros's Data Science expertise, quantifiable achievements, technical competencies, and project outcomes. 
                Receive detailed, recruiter-focused insights with specific performance metrics and demonstrated business impact.
            </p>
            <div class="recruiter-note">
                <p class="recruiter-note-content">
                    📋 <strong>Note for Recruiters:</strong> This intelligent assistant is optimized for
                    evaluating my CV. For optimal performance and token efficiency, please formulate concise, 
                    targeted questions. Additional details are available through the structured navigation tabs above or the fast questions left. Please use them for navigation and keep the chat concise.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick question buttons
        st.markdown("""
        <div class="quick-prompt-section">
            <h3 class="quick-prompt-title">💡 Suggested Questions for Recruiters</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Recruiter-focused suggested prompts that highlight strongest qualifications
        suggested_prompts = [
            "What production ML systems has he built and maintained?",
            "How does his Georgia Tech academic performance compare to other candidates?",
            "Tell me about his collaboration with Sandia National Laboratories",
            "What quantifiable business results has he delivered?",
            "How experienced is he with real-time data processing and ETL pipelines?",
            "What makes him qualified for senior data science positions?",
            "Describe his experience with modern AI and LLM technologies",
            "How does his 4-year BetMax project demonstrate technical leadership?",
            "What advanced statistical methods and ML algorithms does he use?",
            "Tell me about his teamwork and leadership experience",
            "How ready is he for immediate contribution to data science teams?",
            "What evidence shows his ability to work on high-impact projects?"
        ]
        
        # Create columns for the prompts with enhanced styling
        cols = st.columns(3)
        for i, prompt in enumerate(suggested_prompts):
            col_idx = i % 3
            with cols[col_idx]:
                unique_key = f"suggest_{hash(prompt) % 10000}_{i}"
                if st.button(prompt, key=unique_key, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    response = st.session_state.chatbot.get_response(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.scroll_trigger = time.time()
                    # Force stay on AI Assistant tab
                    st.session_state.active_tab_index = 0
                    st.query_params["tab"] = "0"
                    st.session_state.force_ai_tab = True
                    st.rerun()
        
        # Display chat messages with enhanced styling
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Auto-scroll after displaying messages
            if st.session_state.messages or st.session_state.get('scroll_trigger', 0) > 0:
                force_scroll_to_bottom()
        
    
    elif selected_tab == 1:  # Georgia Tech Excellence
        # Georgia Tech education details
        GeorgaTechShowcase.display_program_showcase()
        
        st.markdown("### 📈 Academic Performance Context")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>3.91/4.0 GPA</h4>
                <p>Top 5% Performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>#9 Global Ranking</h4>
                <p>Elite Program Status</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>36 Credit Hours</h4>
                <p>Intensive Curriculum</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == 2:  # Sandia Labs Research
        # Sandia Labs experience section
        st.markdown("""
        <div class="tab-content">
            <div style="
                background: linear-gradient(135deg, rgba(30, 58, 138, 0.9) 0%, rgba(59, 130, 246, 0.9) 100%);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                color: white;
                padding: 2.5rem;
                border-radius: 20px;
                margin: 1.5rem 0;
                text-align: center;
                box-shadow: 0 15px 50px rgba(30, 58, 138, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
            ">
                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(circle at 30% 70%, rgba(255,255,255,0.1) 0%, transparent 70%); animation: shimmer 8s linear infinite;"></div>
                <h2 style="position: relative; z-index: 2; font-weight: 800; text-shadow: 0 3px 6px rgba(0,0,0,0.3);">🏭 Sandia National Laboratories</h2>
                <h3 style="position: relative; z-index: 2; font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">Elite Research Partnership - Predictive Maintenance AI</h3>
                <h4 style="position: relative; z-index: 2; font-weight: 500; opacity: 0.9; text-shadow: 0 1px 3px rgba(0,0,0,0.3);">Behavioral Analysis & Failure Detection Systems</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔬 Research Excellence & Innovation")
            st.markdown("""
            **Elite Partnership Status:**
            • Prestigious collaboration between Georgia Tech and Sandia National Laboratories
            • Focus on AI-driven failure detection for critical infrastructure
            • SCANIA truck dataset analysis with 1.1M+ data points
            • Research publication potential with national laboratory scientists

            **Technical Innovation:**
            • Pioneered behavioral vs. technical failure analysis methodology
            • Advanced psychological variables investigation in mechanical failures
            • Preventive training recommendations through behavioral insights
            • pandas, NumPy, matplotlib, seaborn, scikit-learn (pipelines, imputation, metrics), XGBoost, Bootstrap up‑sampling, cost‑sensitive thresholding, Custom pipeline classes to prevent data leakage; fixed random seeds
            """)
            
            st.markdown("### 📊 Quantifiable Research Impact")
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("ROC-AUC Score", "0.732", "+15% vs baseline")
                st.metric("Data Scale", "1.1M+", "data points")
            with col1b:
                st.metric("Cost Reduction", "20%", "potential savings")
                st.metric("Features Engineered", "533", "behavioral + technical")
        
        with col2:
            st.markdown("### 🎯 Business & Academic Impact")
            st.markdown("""
            **Real-World Applications:**
            • Targeted driver training programs based on behavioral analysis
            • Reduced safety incidents through predictive interventions
            • Operational cost optimization via predictive maintenance
            • Enhanced equipment reliability and performance protocols

            **Academic & Research Excellence:**
            • Partnership with premier national research institution
            • Potential co-authored publication with Sandia scientists
            • Interdisciplinary approach combining psychology and engineering
            • Advanced statistical and machine learning methodologies
            """)
            
            st.markdown("### 🔗 Research Documentation")
            st.markdown("""
            **Available Resources:**
            • [GitHub Repository](https://github.com/venie1/Predictive-Maintenance-Behavioral-vs.-Technical-Analysis-of-Truck-Failures)
            • Comprehensive technical report and methodology
            • Statistical analysis and model performance metrics
            • Complete data preprocessing and feature engineering pipelines

            **Technical Deliverables:**
            • Production-ready ML models with ensemble methods
            • Real-time processing capabilities (<100ms latency)
            • Behavioral pattern analysis algorithms
            • Cost-benefit analysis and ROI projections
            """)
    
    elif selected_tab == 3:  # Real Estate Advisor
        # Real Estate project details
        st.markdown("""
        <div class="tab-content">
            <div style="
                background: linear-gradient(135deg, rgba(22, 160, 133, 0.9) 0%, rgba(46, 204, 113, 0.9) 100%);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                color: white;
                padding: 2.5rem;
                border-radius: 20px;
                margin: 1.5rem 0;
                text-align: center;
                box-shadow: 0 15px 50px rgba(22, 160, 133, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
            ">
                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(circle at 70% 30%, rgba(255,255,255,0.1) 0%, transparent 70%); animation: shimmer 8s linear infinite;"></div>
                <h2 style="position: relative; z-index: 2; font-weight: 800; text-shadow: 0 3px 6px rgba(0,0,0,0.3);">🏠 RealEstateAdvisor Platform</h2>
                <h3 style="position: relative; z-index: 2; font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">Advanced Predictive Analytics for Real Estate Markets</h3>
                <h4 style="position: relative; z-index: 2; font-weight: 500; opacity: 0.9; text-shadow: 0 1px 3px rgba(0,0,0,0.3);">6% MAPE Accuracy | 12-15% Portfolio Optimization</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Project Overview & Business Impact")
            st.markdown("""
            **Revolutionary Real Estate Forecasting:**
            • End-to-end platform for city-level price prediction across multiple markets
            • Integration of macroeconomic, demographic, and real estate market indicators
            • Advanced time-series forecasting with multiple ML algorithms
            • Production-ready deployment with interactive Power BI dashboards

            **Exceptional Performance Metrics:**
            • **6% MAPE** for 1-3 month forecasting horizons
            • **7-10% MAPE** for 6-12 month long-term predictions
            • **12-15% portfolio return uplift** demonstrated in backtesting simulations
            • **80% automation** of data preprocessing and feature engineering pipelines
            """)
            
            st.markdown("### 📊 Technical Architecture & Innovation")
            st.markdown("""
            **Advanced Data Integration:**
            • **Redfin API**: Real-time property listing and transaction data
            • **FRED Economic Data**: Macroeconomic indicators and interest rates
            • **Census Bureau**: Demographic and socioeconomic variables
            • **Custom Feature Engineering**: 50+ lagged and rolling window features

            **Multi-Algorithm Ensemble Approach:**
            • **RidgeCV**: Regularized linear regression with cross-validation
            • **Random Forest**: Non-linear pattern recognition and feature importance
            • **XGBoost**: Gradient boosting for complex market dynamics
            • **Prophet**: Time-series decomposition with seasonal adjustments
            """)
        
        with col2:
            st.markdown("### 🚀 Production Deployment & Business Value")
            st.markdown("""
            **Interactive Dashboard & Visualization:**
            • **Power BI Integration**: Dynamic geographic mapping and trend analysis
            • **Real-time Updates**: Automated data refresh and model retraining
            • **User-friendly Interface**: Intuitive controls for market exploration
            • **Export Capabilities**: Professional reports for stakeholder presentation

            **Quantifiable Business Outcomes:**
            • **Portfolio Optimization**: 12-15% improvement in investment returns
            • **Risk Assessment**: Advanced uncertainty quantification and confidence intervals
            • **Market Timing**: Optimal entry/exit point identification
            • **Competitive Intelligence**: Comparative market analysis across cities
            """)
            
            st.markdown("### 🛠️ Technical Excellence & Innovation")
            st.markdown("""
            **Advanced ML Engineering:**
            • **Automated Preprocessing**: 80% reduction in manual data preparation
            • **Cross-validation Framework**: Robust model validation with time-series splits
            • **Feature Selection**: Statistical significance testing and correlation analysis
            • **Model Interpretability**: SHAP values and feature importance analysis

            **Open Source & Documentation:**
            • **GitHub Repository**: [RealEstateAdvisor](https://github.com/venie1/RealEstateAdvisor)
            • **Complete Codebase**: Production-ready Python implementation
            • **Comprehensive Documentation**: Technical methodology and business applications
            • **Reproducible Results**: Standardized evaluation metrics and benchmarks
            """)
            
            # Performance metrics chart
            st.markdown("### 📈 Performance Benchmarks")
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Short-term MAPE", "6%", "1-3 months")
            with col2b:
                st.metric("Long-term MAPE", "7-10%", "6-12 months")
            with col2c:
                st.metric("Portfolio Uplift", "12-15%", "backtesting ROI")
    
    elif selected_tab == 4:  # BetMax Analytics
        # BetMax sports analytics project
        st.markdown("""
        <div class="tab-content">
            <div style="
                background: linear-gradient(135deg, rgba(5, 150, 105, 0.9) 0%, rgba(16, 185, 129, 0.9) 100%);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                color: white;
                padding: 2.5rem;
                border-radius: 20px;
                margin: 1.5rem 0;
                text-align: center;
                box-shadow: 0 15px 50px rgba(5, 150, 105, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
            ">
                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, transparent 70%); animation: shimmer 8s linear infinite;"></div>
                <h2 style="position: relative; z-index: 2; font-weight: 800; text-shadow: 0 3px 6px rgba(0,0,0,0.3);">⚽ BetMax Sports Analytics</h2>
                <h3 style="position: relative; z-index: 2; font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">Advanced Football Prediction System</h3>
                <h4 style="position: relative; z-index: 2; font-weight: 500; opacity: 0.9; text-shadow: 0 1px 3px rgba(0,0,0,0.3);">4+ Years of Consistent Performance</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏈 Project Genesis & Foundation")
            st.markdown("""
            **Partnership & Domain Expertise:**
            • Co-founded with longtime football partner and fellow player
            • Deep domain knowledge from years of playing experience
            • Extensive research into goal-scoring patterns and game dynamics
            • Specialized focus on second-half goal occurrence prediction

            **Research-Driven Approach:**
            • Multi-year analysis of factors affecting second-half scoring
            • Mathematical formula development through heuristics and domain expertise
            • Hand-crafted initial models based on football knowledge
            • Iterative data-driven refinement and optimization
            """)
            
            st.markdown("### 🎯 Technical Architecture & Evolution")
            st.markdown("""
            **System Development Phases:**
            • **Phase 1:** Heuristic mathematical formula creation
            • **Phase 2:** Automated web scraping and data collection
            • **Phase 3:** Real-time notification and alert systems
            • **Phase 4:** Machine Learning integration and model optimization

            **Production Technology Stack:**
            • Python ecosystem for core development and analysis
            • Selenium for automated, real-time data collection
            • ETL pipelines for data processing and feature engineering
            • Email notification automation for stakeholders
            • Advanced ML algorithms for prediction optimization
            """)
        
        with col2:
            st.markdown("### 📈 Performance Metrics & Results")
            st.markdown("""
            **Exceptional Accuracy & Performance:**
            • 90% accuracy for high-confidence predictions
            • Significantly outperformed 75-80% baseline rates
            • Consistent performance validated across 1,000+ matches
            • Real-time predictions delivered within 3-second SLA
            • Automated risk assessment and confidence scoring

            **Quantifiable Business Impact:**
            • High estimated ROI through accurate prediction systems
            • Automated decision-making and risk management
            • Evidence-based betting strategies with performance tracking
            • Measurable improvements in prediction accuracy over time
            """)
            
            st.markdown("### 🔧 Advanced System Capabilities")
            st.markdown("""
            **Production-Grade Features:**
            • Real-time data scraping and processing pipelines
            • Automated high-confidence prediction alerts
            • Historical performance tracking and analysis
            • Machine Learning model continuous optimization
            • Statistical validation against public benchmarks

            **Open Source & Documentation:**
            • [GitHub Repository](https://github.com/venie1/BetPredictor)
            • Complete methodology and technical documentation
            • Reproducible results with performance validation
            • Educational resource for sports analytics community
            """)
    
    # Replace the project cards section in tab6 with this fixed version:

    # Replace the entire tab6 section with this Streamlit-native approach:

    elif selected_tab == 5:  # Project Portfolio
        # Project portfolio section
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(139, 69, 19, 0.9) 0%, rgba(205, 133, 63, 0.9) 100%);
            backdrop-filter: blur(20px);
            color: white;
            padding: 2.5rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 15px 50px rgba(139, 69, 19, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
        ">
            <h2 style="margin: 0; font-weight: 800;">🚀 Complete Project Portfolio</h2>
            <h3 style="margin: 0.5rem 0; font-weight: 600;">Research, Production Systems & Academic Excellence</h3>
            <h4 style="margin: 0.5rem 0; font-weight: 500; opacity: 0.9;">From National Labs to Real-World Applications</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Project overview metrics using Streamlit columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Research Projects", "4", "Including Sandia Labs")
        with col2:
            st.metric("Production Systems", "2", "Live ML Applications")
        with col3:
            st.metric("Data Points", "1.1M+", "Processed")
        with col4:
            st.metric("Open Source", "Available", "GitHub Repos")
        
        # AI Assistant Integration for Projects
        st.markdown("### 🤖 Ask AI Assistant About Projects")
        if st.button("💬 Get Detailed Project Analysis", key="ai_projects", use_container_width=True):
            question = "Tell me about Petros's data science projects and technical achievements"
            response = st.session_state.chatbot.get_response(question)
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.active_tab_index = 0
            st.query_params["tab"] = "0"
            st.session_state.force_ai_tab = True
            st.rerun()
        
        # Define projects data
        projects = [
            {
                "name": "🤖 AI Resume Agent",
                "type": "AI Engineering/Open Source",
                "period": "2025 - Current",
                "location": "Personal Project, Open Source",
                "description": "Interactive AI-powered resume assistant using OpenAI GPT integration. Features conversational AI, advanced prompt engineering, session management, and real-time chat interface. Built with Streamlit for deployment and demonstrates practical LLM application development.",
                "achievements": [
                    "🧠 OpenAI GPT-4 API integration with custom prompt engineering",
                    "💬 Conversational AI interface with intelligent response system",
                    "🔧 Advanced session state management and real-time chat functionality",
                    "🎨 Professional UI/UX with interactive navigation and visualizations",
                    "📊 Dynamic data visualization integration with Plotly charts",
                    "🚀 Streamlit deployment with responsive design and scroll automation"
                ],
                "tech_stack": ["Python", "OpenAI API", "Streamlit", "Plotly", "Session Management", "LLM Integration"],
                "github": "https://github.com/venie1/ResumeAgent",
                "color": "#9c27b0"
            },
            {
                "name": "🏭 Sandia Labs Predictive Maintenance",
                "type": "Work/Research (NDA Protected)",
                "period": "May 2025 – Aug 2025",
                "location": "Sandia National Laboratories, Georgia Tech",
                "description": "Behavioral vs. technical analysis of truck failures for predictive maintenance. Modeled both driver behaviors and mechanical indicators to forecast component failures using large-scale, real-world telematics data (1.1M time-steps, 23,550 vehicles).",
                "achievements": [
                    "🔢 Merged 1.1M time-steps from 23,550 vehicles; engineered 533 features",
                    "🧠 Partitioned behavioral (24) and technical (81) features for hybrid modeling",
                    "📊 Used Random Forest, Gradient Boosting, XGBoost; AUC up to 0.732, 20% cost reduction",
                    "💼 Business impact: targeted driver training, reduced safety incidents, cost savings",  
                    "📋 Report and summary available; code proprietary (NDA)"
                ],
                "tech_stack": ["Python", "pandas", "NumPy", "scikit-learn", "XGBoost", "matplotlib", "seaborn"],
                "github": "https://github.com/venie1/Predictive-Maintenance-Behavioral-vs.-Technical-Analysis-of-Truck-Failures",
                "color": "#1e40af"
            },
            {
                "name": "🏠 RealEstateAdvisor Platform",
                "type": "School/Open Source",
                "period": "July 2025",
                "location": "Georgia Tech, Open Source",
                "description": "End-to-end real estate forecasting platform for city-level price prediction. Used macroeconomic, socio-economic, and market features. Achieved competitive MAPE for 1-/3-month (6%) and 6-/12-month (7–10%) horizons.",
                "achievements": [
                    "📊 Merged Redfin, FRED, Census data; engineered lagged/rolling features",
                    "🤖 Trained RidgeCV, Random Forest, XGBoost, Prophet models",
                    "⚡ Automated 80% of preprocessing; Power BI dashboard with dynamic maps",
                    "💰 Simulated 12–15% portfolio return uplift in backtests",
                    "🌟 Open source, code and results available"
                ],
                "tech_stack": ["Python", "Prophet", "Scikit-learn", "Streamlit", "GeoPandas", "SQL", "APIs"],
                "github": "https://github.com/venie1/RealEstateAdvisor",
                "color": "#059669"
            },
            {
                "name": "⚽ BetMax Sports Analytics",
                "type": "Self-Employed/Production",
                "period": "Jan 2019 – Dec 2022",
                "location": "Self-Employed, Athens, Greece",
                "description": "Co-founded and built an automated live football match prediction system for second-half goal occurrence (over 0.5 goals). Achieved 90% accuracy for high-confidence picks (vs 75–80% base rate). Real-time and historical data scraping, automated notifications.",
                "achievements": [
                    "🎯 Developed and maintained with partner; processed 1,000+ matches",
                    "🔄 Automated data scraping (Selenium) and real-time ETL pipelines",
                    "📧 Automated high-confidence email alerts for stakeholders",
                    "📈 Benchmarked against public league statistics (SoccerSTATS, AFootballReport)",
                    "🌟 Open source, code and methodology available"
                ],
                "tech_stack": ["Python", "Selenium", "Scikit-learn", "Pandas", "SQL", "API Integration", "Power BI", "Git"],
                "github": "https://github.com/venie1/BetPredictor",
                "color": "#dc2626"
            },
            {
                "name": "🦠 COVID-19 Forecasting Application",
                "type": "Bachelor Thesis (10/10 Grade)",
                "period": "2022",
                "location": "University of Piraeus, Bachelor Thesis",
                "description": "Designed and deployed a Flask app for COVID-19 case analysis and forecasting. Automated data scraping, ETL, and notifications. Benchmarked seven time-series models for 14-day forecasts. Reduced RMSE by 15% over naive baseline (Prophet RMSE: 88.8).",
                "achievements": [
                    "🔄 Automated data scraping and ETL from ECDC using Selenium",
                    "⚙️ Engineered time-series features and uncertainty quantification",
                    "📊 Benchmarked Polynomial, SVM, Holt's, ARIMA/SARIMA, Prophet",
                    "📱 Automated notifications and interactive dashboards (Plotly, Flask)",
                    "🏆 Grade: 10/10, delivered actionable insights to public health stakeholders"
                ],
                "tech_stack": ["Python", "Selenium", "Flask", "Prophet", "Plotly", "Streamlit", "Pandas", "APIs", "Statistical Modeling"],
                "github": "https://github.com/venie1/Covid-prediction-application",
                "color": "#7c2d12"
            }
        ]
        
        # Display each project using Streamlit native components
        for i, project in enumerate(projects, 1):
            # Create a container for each project
            with st.container():
                # Project header with custom styling
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {project['color']}20, {project['color']}10);
                    border-left: 4px solid {project['color']};
                    border-radius: 10px;
                    padding: 1rem;
                    margin: 1.5rem 0;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h3 style="margin: 0; color: #ffffff; font-size: 1.3rem;">{i}. {project['name']}</h3>
                        <span style="
                            background: {project['color']}40;
                            color: #ffffff;
                            padding: 0.3rem 0.8rem;
                            border-radius: 15px;
                            font-size: 0.8rem;
                            font-weight: 500;
                        ">{project['type']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Project details using columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📅 Duration:** {project['period']}")
                    st.markdown(f"**📍 Location:** {project['location']}")
                    st.markdown(f"**📝 Description:**")
                    st.write(project['description'])
                    
                    st.markdown("**🎯 Key Technical Achievements:**")
                    for achievement in project['achievements']:
                        st.markdown(f"• {achievement}")
                
                with col2:
                    st.markdown("**🛠️ Technology Stack:**")
                    # Create technology badges using Streamlit
                    tech_cols = st.columns(2)
                    for idx, tech in enumerate(project['tech_stack']):
                        with tech_cols[idx % 2]:
                            st.markdown(f"""
                            <span style="
                                background: rgba(255, 255, 255, 0.1);
                                color: #ffffff;
                                padding: 0.2rem 0.5rem;
                                border-radius: 10px;
                                font-size: 0.8rem;
                                margin: 0.2rem;
                                display: inline-block;
                                border: 1px solid rgba(255, 255, 255, 0.2);
                            ">{tech}</span>
                            """, unsafe_allow_html=True)
                    
                    # GitHub link
                    if project.get('github'):
                        st.markdown("**🔗 Repository:**")
                        st.markdown(f"[📂 View GitHub Repository]({project['github']})")
                
                # Add separator
                st.markdown("---")
    elif selected_tab == 6:  # Skills & Performance       
        # Performance Metrics Dashboard using Streamlit native metrics
        st.markdown("### 🎯 Performance Excellence Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sports ML Accuracy", "90%", "vs 75-80% baseline", delta_color="normal")
        with col2:
            st.metric("Real Estate MAPE", "6%", "1-3 month horizon", delta_color="inverse")
        with col3:
            st.metric("Predictive Maintenance", "0.732", "ROC-AUC Score", delta_color="normal")
        with col4:
            st.metric("Academic Performance", "3.91", "Top 5% of cohort", delta_color="normal")
        
        # AI Assistant Integration for Skills
        st.markdown("### 🤖 Ask AI Assistant About Skills")
        if st.button("💬 Get Technical Skills Analysis", key="ai_skills", use_container_width=True):
            question = "Tell me about Petros's technical skills and experience"
            response = st.session_state.chatbot.get_response(question)
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.active_tab_index = 0
            st.query_params["tab"] = "0"
            st.session_state.force_ai_tab = True
            st.rerun()
        
        # Technical Skills using experience-based approach
        st.markdown("### 🛠️ Technical Skills & Experience")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🐍 Core Data Science & ML**")
            skills_data = [
                ("Python", "4+ years", "8 projects"),
                ("Scikit-learn", "3+ years", "6 projects"), 
                ("Pandas/NumPy", "4+ years", "All projects"),
                ("XGBoost", "2+ years", "3 projects"),
                ("Statistical Modeling", "3+ years", "Academic + Research"),
                ("Data Visualization", "3+ years", "Power BI + Plotly"),
                ("SQL", "3+ years", "Database integration"),
                ("OpenAI API", "Current", "AI Resume Agent")
            ]
            
            for skill, experience, projects in skills_data:
                st.markdown(f"**{skill}**")
                st.markdown(f"📅 {experience} | 📁 {projects}")
                st.markdown("---")
        
        with col2:
            st.markdown("**🎯 Specialized Domains**")
            domain_skills = [
                ("Time Series Forecasting", "Prophet + ARIMA", "Real Estate + COVID"),
                ("Predictive Maintenance", "Sandia Labs Research", "500+ features"),
                ("Sports Analytics", "BetMax Production", "4+ years live system"),
                ("LLM Integration", "OpenAI GPT-4", "Conversational AI")
            ]
            
            for skill, method, application in domain_skills:
                st.markdown(f"**{skill}**")
                st.markdown(f"🔧 {method} | 🎯 {application}")
                st.markdown("---")
        
        # Project Impact Visualization
        st.markdown("### 📈 Project Impact Analysis")
        
        impact_data = [
            ("🤖 AI Resume Agent", "Current Production", "OpenAI GPT-4 integration, conversational AI"),
            ("🏭 Sandia Labs Research", "Research Impact", "National laboratory collaboration, 500+ features"),
            ("⚽ BetMax Sports Analytics", "4+ Years Live", "90% accuracy, 1,000+ matches processed"),
            ("🏠 Real Estate Advisor", "Portfolio Platform", "98% explained variance, 12-15% uplift"),
            ("🦠 COVID Forecasting", "Academic Excellence", "Perfect 10/10 thesis grade, public health impact")
        ]
        
        for project_name, status, description in impact_data:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{project_name}**")
                st.markdown(f"🎯 *{status}*")
            with col2:
                st.markdown(f"{description}")
            st.markdown("---")
        
        # Contact information
        st.markdown("""
        ---
        ### 💼 Ready to Connect?
        
        Contact Petros directly to discuss how his advanced data science expertise and proven track record 
        can drive measurable results for your organization.
        
        **📧 Email:** pgvenieris@outlook.com
        """)

    # Global chat input - only show when on AI Assistant tab
    if st.session_state.active_tab_index == 0:
        if prompt := st.chat_input("Ask me anything about Petros's background..."):
            # Process user input
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Generating response..."):
                response = st.session_state.chatbot.get_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.session_state.scroll_trigger = time.time()
            # Stay on chat tab after response
            st.session_state.active_tab_index = 0
            st.query_params["tab"] = "0"
            st.rerun()

# Run the application
if __name__ == "__main__":
    main()
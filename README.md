#  🤖 AI Resume Agent - Interactive Career Assistant


  **Update, What's new:
✅ RAG (Retrieval-Augmented Generation) 
✅ Vector Embeddings with Sentence Transformers (Hugging Face ecosystem)
✅ FAISS Vector Store for lightning-fast semantic search
✅ LangChain Integration with RecursiveCharacterTextSplitter
✅ Multi-Modal AI Architecture with OpenAI GPT-4 API
✅ Cosine Similarity scoring for contextual relevance
✅ Neural Semantic Search capabilities

Complete Technical Stack:
• Hugging Face Transformers (all-MiniLM-L6-v2 model) OpenAI GPT-4 for advanced LLM integration
• Streamlit for production deployment
• PyPDF2 for document processing
  what does it mean? 
  
I can upload reports, resumes, articles, posts and with the new system my data is automatically, efficiently and cheaply read so that the agent can give a targeted clear answer. So my resume assistant 
can tell you my information effectively always following the guidelines I have given him, and if I want to update the information I simply upload a new file. With embedding and semantic search i save
space,money,time and processing power. 

  
  https://resumeagent-5lxhhftrdnxwxgeys3dg2h.streamlit.app/
  
  https://www.python.org/downloads/
  
  https://openai.com/

  Skip the boring resume! Chat with my AI assistant to learn about my professional background.

  An intelligent conversational system that transforms traditional resume viewing into an interactive experience
  using OpenAI GPT-4, advanced prompt engineering, and comprehensive data analysis.

  🚀 https://resumeagent-5lxhhftrdnxwxgeys3dg2h.streamlit.app/

  Ask questions like:
  - "What's Petros's experience with machine learning?"
  - "Tell me about his projects at Georgia Tech"
  - "How does his background fit a data scientist role?"
  - "What programming languages does he use?"

  ✨ Key Features

  - 🤖 AI-Powered Chat Interface: Natural conversations using OpenAI GPT-4
  - 📊 Interactive Data Visualizations: Skills charts and project timelines using Plotly
  - 🎯 Intelligent Response System: Context-aware responses with keyword matching
  - 💼 Comprehensive Resume Data: Detailed information about education, work experience, and projects
  - 📱 Professional UI: Clean, responsive design with custom styling
  - 🔄 Real-time Processing: Instant responses with conversation memory

  # Core Technologies
  🤖 AI: OpenAI GPT-4, Advanced Prompt Engineering
  🐍 Backend: Python, Streamlit
  📊 Data: Pandas, NumPy
  📈 Visualization: Plotly, Matplotlib, Seaborn
  🎨 Frontend: Streamlit Components, Custom CSS/HTML

  📁 Project Structure

  ResumeAgent/
  ├── ps.py                    # Main Streamlit application (3000+ lines)
  ├── requirements.txt         # Python dependencies
  ├── README.md               # Project documentation
  ├── ph.jpg                  # Profile photo
  ├── ph.png                  # Profile photo (PNG)

  🚀 Installation

  Prerequisites

  - Python 3.8+
  - OpenAI API key

  Quick Start

  1. Clone the repository
  git clone https://github.com/venie1/ResumeAgent.git
  cd ResumeAgent

  2. Install dependencies
  pip install -r requirements.txt

  3. Set up OpenAI API key
  # Set environment variable
  export OPENAI_API_KEY="your-api-key-here"

  4. Run the application
  streamlit run ps.py

  5. Open your browser
  Navigate to http://localhost:8501

  🏗 Architecture

  The application consists of several key classes:

  ResumeData

  - Comprehensive data structure containing personal info, education, work experience, and projects
  - Structured data for easy access and processing

  ChatbotConfig

  - OpenAI configuration and prompt templates
  - System prompts optimized for resume-focused conversations

  IntelligentResponseSystem

  - Advanced keyword matching and context analysis
  - OpenAI integration with fallback responses
  - Multi-language support and conversation memory

  Core Features

  - Smart Scroll Management: Auto-scroll functionality for better UX
  - Data Visualization: Interactive charts showing skills and project timelines
  - Professional Styling: Custom CSS for polished appearance

  📋 Dependencies

  # Core Framework
  streamlit>=1.24.1

  # AI Integration
  openai>=0.27.0

  # Data Processing
  pandas>=2.0.0
  numpy>=1.25.0
  python-dotenv>=1.0.0

  # Machine Learning (for data analysis)
  scikit-learn>=1.2.0
  xgboost>=1.7.0
  prophet>=1.1.1

  # Visualization
  plotly>=5.14.0
  matplotlib>=3.7.0
  seaborn>=0.12.0
  geopandas>=0.13.0

  💡 Key Components

  1. Conversational AI

  - OpenAI GPT-4 integration
  - Context-aware responses
  - Professional resume-focused prompts

  2. Data Processing

  - Structured resume information
  - Skill categorization and analysis
  - Project timeline generation

  3. Interactive Visualizations

  - Skills distribution charts
  - Project timeline visualization
  - Professional experience mapping

  4. User Experience

  - Responsive chat interface
  - Auto-scroll functionality
  - Professional styling and branding

  🔧 Configuration

  The application uses environment variables for configuration:

  # Required
  OPENAI_API_KEY=your-openai-api-key

  # Optional Streamlit configuration
  STREAMLIT_SERVER_PORT=8501
  STREAMLIT_THEME_BASE=light

  🚀 Deployment

  The application is deployed on Streamlit Cloud and accessible at:
  https://resumeagent-5lxhhftrdnxwxgeys3dg2h.streamlit.app/

  🎯 Use Cases

  - Interactive Resume Viewing: Engaging way to explore professional background
  - Interview Preparation: Practice questions about experience and skills
  - Career Counseling: Get insights about career progression and opportunities
  - Portfolio Showcase: Demonstrate technical skills through interactive experience

  📞 Contact

  - Email: pgvenieris@outlook.com
  - LinkedIn: https://linkedin.com/in/petrosvenieris
  - GitHub: https://github.com/venie1
  - Live Demo: https://resumeagent-5lxhhftrdnxwxgeys3dg2h.streamlit.app/

  📜 License

  This project showcases technical skills and serves as an interactive portfolio piece.

  Built with ❤️ by Petros Venieris Data Scientist & ML Engineer | Georgia Tech Graduate

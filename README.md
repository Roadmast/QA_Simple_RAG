# QA_Simple_RAG Streamlit Application

## Overview
QA_RAG is a Streamlit-based Question Answering system leveraging Retrieval-Augmented Generation (RAG) for efficient and accurate responses. This repository contains the necessary code and configuration files to run the application locally or within a Docker container. 🌟✨🚀

## Folder Structure
```
QA_Simple_RAG
├── app.py                              # Main Streamlit application script
├── requirements.txt                    # Python dependencies
├── Finalcial_Q&A_RAG_System.ipynb      # Jupyter Notebook for development and experimentation
├── .env                                # Environment variables configuration
├── Dockerfile                          # Docker configuration for containerization
```

## Prerequisites
Make sure you have the following installed: 🌟✨🚀

- Python 3.8 or higher
- pip (Python package manager)
- Docker (optional, for containerized deployment)

## Installation

### Local Setup

1. Clone the repository: 🌟✨🚀
   ```bash
   git clone https://github.com/your_username/QA_Simple_RAG.git
   cd QA_Simple_RAG
   ```

2. Create a virtual environment and activate it: 🌟✨🚀
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies: 🌟✨🚀
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the `.env` file with your required environment variables (if applicable). 🌟✨🚀

5. Run the application: 🌟✨🚀
   ```bash
   streamlit run app.py
   ```

6. Open your browser and navigate to `http://localhost:8501` to access the application. 🌟✨🚀

### Docker Setup

1. Build the Docker image: 🌟✨🚀
   ```bash
   docker build -t qa_rag_app .
   ```

2. Run the Docker container: 🌟✨🚀
   ```bash
   docker run -p 8501:8501 --env-file .env qa_rag_app
   ```

3. Open your browser and navigate to `http://localhost:8501` to access the application. 🌟✨🚀

## Files Description

- **app.py**: Contains the main logic for the Streamlit application. 🌟✨🚀
- **requirements.txt**: Lists the Python dependencies required to run the application. 🌟✨🚀
- **Financial_Q&A_RAG_System.ipynb**: Jupyter Notebook used for exploratory development and prototyping. 🌟✨🚀
- **.env**: Stores environment-specific variables such as API keys or configuration settings. 🌟✨🚀
- **Dockerfile**: Defines the Docker image for containerized deployment. 🌟✨🚀

## Features
- **Streamlit Interface**: Interactive web-based interface for user queries. 🌟✨🚀
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval and generative language models for accurate responses. 🌟✨🚀
- **Docker Support**: Containerized deployment for consistent and portable execution. 🌟✨🚀

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes. 🌟✨🚀


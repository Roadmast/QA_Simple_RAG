# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY app.py .
COPY .env .

# Expose the Streamlit default port (8501)
EXPOSE 8501

# Set environment variables to configure Streamlit (optional)
ENV STREAMLIT_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]

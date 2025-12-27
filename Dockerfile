# Base Image: Lightweight Python
FROM python:3.9-slim

# Set Working Directory
WORKDIR /app

# Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY . .

# Expose Streamlit Port
EXPOSE 8501

# Default Command: Launch the Web Dashboard
# To run the Terminal App instead, use: docker run -it sentinel-ai python manager_chat.py
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

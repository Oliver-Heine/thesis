FROM mcr.microsoft.com/playwright/python:v1.53.0-jammy

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install

# Copy project
COPY . .

# Expose Flask port
EXPOSE 8090

# Start server
CMD ["python", "main.py"]
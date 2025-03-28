# Use the official Python image
FROM python:3.13

# Set the working directory
WORKDIR /options-app

# Copy only dependencies first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (code will be overridden in development mode)
COPY . .

# Expose port 3000
EXPOSE 3000

# Default command (will be overridden by docker-compose)
CMD ["python", "app.py"]

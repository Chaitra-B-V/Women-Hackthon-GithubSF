FROM python:3.10-bullseye

COPY requirements.txt requirements.txt

# Upgrade pip and install requirements
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Although Cloud Run does not require EXPOSE to function, it's good practice
# to indicate which port your application listens on.
EXPOSE 8080

# Use environment variables to configure the service.
# DEBUG variable can be set to False in production environments.
ENV DEBUG=True
ENV NAME=World
# Here we set a default value for PORT, but Cloud Run will provide its own PORT value
ENV PORT=8080

# The CMD instruction has been updated to use a shell form to ensure that
# the PORT environment variable is correctly used.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --reload

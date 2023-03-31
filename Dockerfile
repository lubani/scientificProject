# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory to /app
WORKDIR /app
RUN echo "deb https://ftp.us.debian.org/debian testing main contrib non-free" | tee -a /etc/apt/sources.list  \
    && apt-get update && apt-get install -y x11-apps
ENV DISPLAY host.docker.internal:0


# Copy the current directory contents into the container at /app
COPY interface2.py /app
COPY requirements.txt /app
COPY README.md /app
COPY /dist /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt


# Define environment variable
ENV NAME venv

# Run app.py when the container launches
CMD ["python", "interface2.py"]

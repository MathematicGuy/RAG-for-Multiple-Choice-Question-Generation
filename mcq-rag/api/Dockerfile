# slim tag for minimal python image version
FROM python:3.11-slim

#? sets the working dir inside container to /app
# allow command (COPY, RUN, etc..) execute inside this directory
WORKDIR /app

#? the cmd update & install apt from git
# apt mean advance package tool
# -y automate answers 'yes' to prompts during install
RUN apt-get update && apt-get install -y git

#? the cmd install requirements copied from the requirements.txt file without caching the file
# copies the requirements file from host-machine into current working dir /app
COPY requirements.txt .
# --no-cache-dir literally mean not caching installed file
RUN pip install --no-cache-dir -r requirements.txt

#? copy everything from the current dir (the first ".") on your host machine into the container's working dir /app (the second ".")
COPY . .

#? Listen to port 8000.
# make sure to run: docker run -p 8000:8000.
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"] 

# docker build -t your-dockerhub-username/enhanced-rag-mcq:latest .
# docker run -p 8000:8000 3e82eb834526   (get docker_id from 'docker images' command)

# docker build -t mcq-api:local
# docker run --rm -p 8000:8000 mcq-api:local
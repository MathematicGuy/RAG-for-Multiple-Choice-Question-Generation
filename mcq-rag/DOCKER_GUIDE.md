

# Docker Guide for Enhanced RAG MCQ API

This guide will help you set up and run the Docker image for the `app.py` file, which serves as the API for the Enhanced RAG MCQ Generator.

## Prerequisites

1. **Docker Installed**: Ensure Docker is installed on your system. You can download it from [Docker's official website](https://www.docker.com/).
2. **Clone the Repository**: Clone the repository containing the `app.py` file and navigate to the `mcq-rag/api` directory.

```bash
# Clone the repository
git clone <repository-url>

# Navigate to the API directory
cd RAG-for-Multiple-Choice-Question-Generation/mcq-rag/api
```

## Build the Docker Image

Run the following command to build the Docker image:

```bash
docker build -t enhanced-rag-mcq:latest .
```

- `-t enhanced-rag-mcq:latest`: Tags the image with the name `enhanced-rag-mcq` and the tag `latest`.
- `.`: Refers to the current directory as the build context.

## Run the Docker Container

Run the following command to start the container:

```bash
docker run -p 8000:8000 enhanced-rag-mcq:latest
```

- `-p 8000:8000`: Maps port 8000 on your local machine to port 8000 in the container.
- `enhanced-rag-mcq:latest`: Specifies the image to run.

## Access the API

Once the container is running, you can access the API at:

```
http://localhost:8000
```

## API Endpoints

### `/generate/`
- **Method**: POST
- **Description**: Generates multiple-choice questions based on the uploaded document and provided parameters.
- **Parameters**:
  - `file`: The PDF file to process.
  - `topics`: A comma-separated list of topics.
  - `n_questions`: Number of questions per topic.
  - `difficulty`: Difficulty level (e.g., `easy`, `medium`, `hard`).
  - `qtype`: Question type (e.g., `definition`, `application`).

## Example Request

Use a tool like `curl` or Postman to send a request to the API. Example using `curl`:

```bash
curl -X POST "http://localhost:8000/generate/" \
  -F "file=@example.pdf" \
  -F "topics=Topic1,Topic2" \
  -F "n_questions=5" \
  -F "difficulty=medium" \
  -F "qtype=definition"
```

## Stop the Container

To stop the running container, press `Ctrl+C` in the terminal where the container is running.

Alternatively, you can stop it using Docker commands:

```bash
# List running containers
docker ps

# Stop the container
docker stop <container-id>
```

Replace `<container-id>` with the ID of the running container.

## Clean Up

To remove the Docker image and container:

```bash
# Remove the container
docker rm <container-id>

# Remove the image
docker rmi enhanced-rag-mcq:latest
```

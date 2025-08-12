# Docker Notes and Explanations

This file provides detailed explanations of Docker commands, flags, and the Docker workflow used in this project.

## Docker Workflow Overview

1. **Build**: Create a Docker image from the Dockerfile
2. **Run**: Start a container from the image
3. **Access**: Interact with the running application
4. **Stop**: Stop the running container
5. **Clean Up**: Remove containers and images when no longer needed

## Dockerfile Explanation

### Base Image
```dockerfile
FROM python:3.11-slim
```
- **FROM**: Specifies the base image to use
- **python:3.11-slim**: A lightweight Python 3.11 image based on Debian slim
- **Benefits**: Smaller size compared to full Python image, includes Python and pip pre-installed

### Environment Variables
```dockerfile
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"
```
- **ENV**: Sets environment variables in the container
- **DEBIAN_FRONTEND=noninteractive**: Prevents interactive prompts during package installation
- **PATH**: Adds uv installation directory to the system PATH (done automatically, don't need to change PATH to your path )

### System Dependencies
```dockerfile
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    curl \
    git \
    && apt-get upgrade -y \
    && ln -s /usr/bin/python3.11 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
```
- **RUN**: Executes commands during image build
- **apt-get update**: Updates package lists
- **apt-get install -y**: Installs packages without prompting for confirmation
- **&&**: Chains commands together (all must succeed)
- **apt-get clean**: Removes downloaded package files
- **rm -rf /var/lib/apt/lists/***: Removes package lists to reduce image size
- **ln -s**: Creates a symbolic link for python3

### UV Installation
```dockerfile
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
```
- **curl -LsSf**: Downloads the uv installer script
  - **-L**: Follow redirects
  - **-s**: Silent mode (no progress bar)
  - **-S**: Show errors even in silent mode
  - **-f**: Fail silently on server errors
- **| sh**: Pipes the downloaded script to shell for execution

### Working Directory
```dockerfile
WORKDIR /app
```
- **WORKDIR**: Sets the working directory for subsequent commands
- All COPY, RUN, and CMD commands will execute from this directory

### Dependency Installation
```dockerfile
COPY requirements.txt .
RUN uv pip install -r requirements.txt
```
- **COPY**: Copies files from host to container
- **requirements.txt .**: Copies requirements.txt to current directory (/app)
- **uv pip install -r**: Uses uv to install Python packages from requirements file

### Application Code
```dockerfile
COPY . .
```
- Copies all files from current directory on host to /app in container
- The first `.` refers to the build context (where Dockerfile is located)
- The second `.` refers to the current working directory in container (/app)

### Port Exposure
```dockerfile
EXPOSE 8000
```
- **EXPOSE**: Documents which port the application uses
- Does not actually publish the port (use -p flag when running)

### Application Startup
```dockerfile
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```
- **CMD**: Default command to run when container starts
- **uvicorn**: ASGI server for Python web applications
- **app:app**: Module name and application instance
- **--host 0.0.0.0**: Listen on all network interfaces
- **--port 8000**: Listen on port 8000
- **--workers 4**: Use 4 worker processes

## Docker Commands and Flags

### Building the Image
```bash
docker build -t heval1st/enhanced-rag-mcq:local .
```
- **docker build**: Command to build an image from Dockerfile
- **-t enhanced-rag-mcq:local**: Tags the image with name and version
  - **-t**: Short for --tag
  - **enhanced-rag-mcq**: Image name
  - **local**: Tag (version)
- **.**: Build context (current directory)

### Running the Container
```bash
docker run -p 8000:8000 heval1st/enhanced-rag-mcq:local
```
- **docker run**: Command to create and start a container
- **-p 8000:8000**: Port mapping
  - **-p**: Short for --publish
  - **8000:8000**: host_port:container_port
- **enhanced-rag-mcq:local**: Image to run

### Additional Run Flags
```bash
docker run -d -p 8000:8000 --name mcq-api enhanced-rag-mcq:local
```
- **-d**: Run in detached mode (background)
- **--name mcq-api**: Assign a name to the container
- **--rm**: Automatically remove container when it stops

### Container Management Commands
```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop <container-id-or-name>

# Remove a container
docker rm <container-id-or-name>

# View container logs
docker logs <container-id-or-name>

# Execute commands in running container
docker exec -it <container-id-or-name> /bin/bash
```

### Image Management Commands
```bash
# List all images
docker images

# Remove an image
docker rmi <image-id-or-name>

# Remove all unused images
docker image prune

# Remove all unused containers, networks, images
docker system prune
```

## Best Practices Applied

1. **Layer Optimization**: Combining RUN commands to reduce layers
2. **Cache Optimization**: Copying requirements.txt before source code
3. **Security**: Cleaning up package caches and temporary files
4. **Size Optimization**: Using slim base image and removing unnecessary files
5. **Non-root User**: Could be added for better security
6. **.dockerignore**: Should be used to exclude unnecessary files

## Troubleshooting

### Common Issues
1. **Port already in use**: Use different host port (-p 8001:8000)
2. **Permission denied**: Check file permissions or run with sudo
3. **Build fails**: Check Dockerfile syntax and dependency availability
4. **Container exits immediately**: Check application logs with `docker logs`

### Debug Commands
```bash
# Build with verbose output
docker build --no-cache -t enhanced-rag-mcq:local .

# Run container in interactive mode
docker run -it enhanced-rag-mcq:local /bin/bash

# Check container resource usage
docker stats <container-id>
```

## Development Workflow

1. **Development**: Make changes to source code
2. **Build**: `docker build -t enhanced-rag-mcq:local .`
3. **Test**: `docker run -p 8000:8000 enhanced-rag-mcq:local`
4. **Verify**: Test API endpoints
5. **Deploy**: Push to registry if needed

## Production Considerations

1. **Multi-stage builds**: Separate build and runtime environments
2. **Health checks**: Add HEALTHCHECK instruction
3. **Non-root user**: Run application as non-root user
4. **Resource limits**: Set memory and CPU limits when running
5. **Monitoring**: Add logging and monitoring solutions
6. **Secrets management**: Use Docker secrets for sensitive data

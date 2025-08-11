FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04

# Optional: set working directory
WORKDIR /app

# Set up time zone
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Copy requirements first for caching
COPY ./requirements.txt .

# Install system and Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy only necessary files for runtime
COPY ./pretrained_wts /app/pretrained_weights/
COPY ./infer_VFA.py /app/
COPY ./MIR /app/MIR
RUN sudo chmod -R a+rw /app/MIR
# Install your custom package (editable if you plan to modify)
WORKDIR /app/MIR
RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/input /app/output /app/json_dir
RUN chmod -R 777 /app/output
# Switch back to /app as the working directory
WORKDIR /app

CMD ["python3","-u","./infer_VFA.py"]
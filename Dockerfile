
# Use the base image
FROM quay.io/opendatahub-contrib/workbench-images:cuda-jupyter-tensorflow-c9s-py311_2023c_latest

# Install necessary dependencies
RUN pip install tensorflow onnx onnxruntime boto3

# Copy the Python script into the container
COPY model.py /workspace/

# Set up environment variables for MinIO server access
ENV MINIO_ENDPOINT=https://minio-s3-minio-operator.apps.cluster-8hg7m.sandbox258.opentlc.com
ENV MINIO_ACCESS_KEY=key
ENV MINIO_SECRET_KEY=secret
ENV AWS_REGION=us-east

# Run the Python script when the container starts
CMD ["python", "/workspace/model.py"]

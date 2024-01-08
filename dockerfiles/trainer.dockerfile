#base image
FROM python:3.10-slim

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

# Set working directory
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
# Possible to use run make requirements


#
COPY first_cc_project/ first_cc_project/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "first_cc_project/train_model.py"]

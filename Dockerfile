FROM continuumio/miniconda3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential

# Create the environment:
COPY . .

RUN conda env create -f environment.yml

# Activate the environment, and make sure it's activated:
#RUN conda activate latentsdf

EXPOSE 5000

# The code to run when container is started:
# ENTRYPOINT ["python", "app.py"]
# ENTRYPOINT ["conda", "run", "-n", "latentsdf", "python", "app.py"]
ENTRYPOINT ["conda", "run", "-n", "latentsdf", "python", "app.py"]
# CMD ["bash", "-c", "activate latentsdf && python app.py"]
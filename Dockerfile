# Utiliser une image Python officielle avec Java (nécessaire pour PySpark)
FROM python:3.11-slim

# Installer les dépendances système nécessaires pour PySpark, Java, et les bibliothèques audio
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    openjdk-21-jdk-headless \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Définir les variables d'environnement Java
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'ensemble du projet
COPY . .

# Variables d'environnement pour PySpark
ENV PYSPARK_PYTHON=/usr/local/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3
ENV SPARK_LOCAL_IP=127.0.0.1

# Créer les répertoires nécessaires
RUN mkdir -p /app/temp /app/models /app/logs /data /output

ENTRYPOINT ["sh", "-c", "python src/extract_wav_files.py --test && python src/feature_extraction.py"]

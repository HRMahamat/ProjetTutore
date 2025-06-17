# Base image officielle
FROM python:3.11-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

# Exposer le port (remplace 8501 si besoin)
EXPOSE 8501

# Commande pour lancer l'app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]

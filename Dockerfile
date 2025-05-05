# Utiliser l'image officielle uv basée sur Debian Bookworm
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Installer make, curl et autres dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Synchroniser les dépendances avec uv
RUN uv sync --frozen

# Copier le reste de l'application
COPY . .

# Commande par défaut pour lancer l'application via uv run
CMD ["uv", "run", "main.py"]

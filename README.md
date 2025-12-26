# dev-translator

Development workspace for the translator service (SeamlessM4T via HuggingFace).

- Builds the translator proxy/backend for local testing
- Exposes API_PORT as configured in docker-compose (default 7104)
- Use `docker build -t inference/translator:local .` to build locally

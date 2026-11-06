# Deploying to Hugging Face Spaces (free, 16 GB RAM)

1. Create an account at https://huggingface.co and create an access token with **Write** permission (Settings -> Access Tokens).
2. Create a Space: https://huggingface.co/new-space -> name it, SDK **Docker** (Blank), hardware **CPU basic (free)**, visibility Public or Private -> Create.
3. Upload the project files (Dockerfile, README.md, app.py, config.py, requirements.txt, modules/, utils/, scripts/, data/, tests/, .streamlit/, assets/) - do NOT upload venv/ or models_cache/.
   - Web upload: Space page -> Files -> Add file -> Upload files, or
   - Git (below).
4. Wait for the build (10-25 min the first time; models are downloaded during the build). Watch the **Logs** tab.
5. Open the Space URL: https://huggingface.co/spaces/<username>/<space-name>

## Git method (PowerShell)
```powershell
git lfs install
git clone https://huggingface.co/spaces/<username>/<space-name>
# copy the project files into the cloned folder, then:
cd <space-name>
git add .
git commit -m "Deploy QA app"
git push
```
Use your username and the access token as the password when asked.

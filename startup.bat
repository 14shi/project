
start cmd /k "cd /d D:\main && conda activate search && uvicorn server:app --host 0.0.0.0 --port 8000 &"

start cmd /k "cd /d D:\main && conda activate search && streamlit run app.py --server.runOnSave true"
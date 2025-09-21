from fastapi import APIRouter, File, UploadFile
from crud.vector_store import upload_file_pdf, merge_results, orchestrate, show_chat_history
import tempfile, os



neo = APIRouter()


@neo.post("/uploadfile_PDF/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
        response = await upload_file_pdf(file, temp_path)
        os.remove(temp_path)  # Clean up temp file
        return response
    except Exception as e:
        return {"error": str(e)}

    
@neo.get("/query_rag/")
async def query_rag_endpoint(question: str):
    try:
        question = await orchestrate(question)
        if question["action"] == "direct":
            return {"answer": question["response"]}
        question = question["response"]
        response = await merge_results(question)
        return response
    except Exception as e:
        return {"error": str(e)}
    
    
@neo.get("/chat_history/")
async def get_chat_history():
    try:
        history = await show_chat_history()
        return {"chat_history": history}
    except Exception as e:
        return {"error": str(e)}
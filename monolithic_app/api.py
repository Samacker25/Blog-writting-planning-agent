from fastapi import FastAPI
from pydantic import BaseModel
from backend import app as graph_app
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class BlogRequest(BaseModel):
    topic: str
    as_of: str

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
def health():
    return {"status": "ok"}


# Generate blog
@app.post("/generate")
async def generate_blog(req: BlogRequest):
    try:
        inputs = {
            "topic": req.topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": req.as_of,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }

        result = graph_app.invoke(inputs)
        return result

    except Exception as e:
        return {"error": str(e)}
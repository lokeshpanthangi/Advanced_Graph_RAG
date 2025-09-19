from urllib import response
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jVector, Neo4jGraph, GraphCypherQAChain
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import fitz  # PyMuPDF
from dotenv import load_dotenv
import os, base64, uuid, json



load_dotenv()

global chathistory
chathistory = []




### Get Creds ###
NEO4J_URL = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE")
NEO4J_INDEX_NAME = os.getenv("AURA_INSTANCENAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")






### INIT ###
model = ChatOpenAI(model="gpt-4o", temperature=0.3)
vision_model = ChatOpenAI(model="gpt-4o", temperature=0.3)
pc = Pinecone(api_key=pinecone_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
parser = StrOutputParser()
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASS,
    database=NEO4J_DB
)
vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)
graph_qa = GraphCypherQAChain.from_llm(
    llm=model,
    graph=graph,
    verbose=True,                 # log generated Cypher
    allow_dangerous_requests=True # allow writes (careful!)
)



### Helpers ###
def make_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def process_table(tbl, file_name, page_num, row_threshold: int = 50):
    rows = tbl.extract()
    tid = make_id("table")

    table_meta = {
        "id": tid,
        "source_file": file_name,
        "page": page_num,
        "type": "table",
        "bbox": tbl.bbox,
        "rows_count": len(rows),
        "cols_count": len(rows[0]) if rows else 0,
    }

    docs = []
    if len(rows) <= row_threshold:
        table_text = "\n".join(["\t".join([cell or "" for cell in row]) for row in rows])
        docs.append(Document(page_content=table_text, metadata=table_meta))
    else:
        docs.append(Document(page_content=f"[Large Table with {len(rows)} rows]", metadata=table_meta))
        for idx, row in enumerate(rows):
            rid = make_id("row")
            row_text = "\t".join([cell or "" for cell in row])
            row_meta = {
                "id": rid,
                "table_id": tid,
                "source_file": file_name,
                "page": page_num,
                "type": "row",
                "row_index": idx,
            }
            docs.append(Document(page_content=row_text, metadata=row_meta))
    return docs




def summarize_image_openai(image_bytes: bytes, image_ext: str = "png") -> str:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    msg = HumanMessage(content=[
        {"type": "text", "text": "Summarize this image in 3 sentences and 5 bullets."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_ext};base64,{image_b64}"}
        }
    ])

    response = vision_model.invoke([msg])
    return response.content if hasattr(response, "content") else str(response)





def process_image(doc, xref, file_name, page_num, img_index):
    imgdict = doc.extract_image(xref)
    image_bytes = imgdict["image"]
    image_ext = imgdict["ext"]
    mime = f"image/{image_ext}"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    summary = summarize_image_openai(image_bytes, image_ext)

    iid = make_id("image")
    meta = {
        "id": iid,
        "source_file": file_name,
        "page": page_num,
        "type": "image",
        "ext": image_ext,
        "mime": mime,
        "index": img_index,
        "width": imgdict.get("width"),
        "height": imgdict.get("height"),
        "image_base64": image_b64,
    }
    return Document(page_content=summary, metadata=meta)



async def upload_file_pdf(file, temp_path):

    doc = fitz.open(temp_path)

    text = ""
    docs = []
    table_docs = []
    image_docs = []

    for page_num, page in enumerate(doc):
        # ---- Text ----
        text += page.get_text()

        # ---- Tables ----
        tables = page.find_tables()
        if tables and tables.tables:
            for tbl in tables.tables:
                table_docs.extend(process_table(tbl, file.filename, page_num))

        # ---- Images ----
        image_list = page.get_images(full=True)
        for img_index, imginfo in enumerate(image_list):
            xref = imginfo[0]
            idoc = process_image(doc, xref, file.filename, page_num, img_index)
            image_docs.append(idoc)

    doc.close()

    # ---- Split text into chunks ----
    chunks = splitter.split_text(text)
    chunk_docs = []
    for i, c in enumerate(chunks):
        cid = make_id("chunk")
        meta = {"id": cid, "source_file": file.filename, "type": "text", "order": i}
        chunk_docs.append(Document(page_content=c, metadata=meta))

    # ✅ Separate docs
    non_table_docs = chunk_docs + image_docs   # goes to Pinecone
    only_table_docs = table_docs               # goes to Neo4j

    # ---- Store TABLES in Neo4j ----
    if only_table_docs:
        neo4j_vector = Neo4jVector.from_documents(
            documents=only_table_docs,
            embedding=embeddings,
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASS,
            database=NEO4J_DB,
            index_name=NEO4J_INDEX_NAME,
            node_label="Chunk",
            text_node_property="text",
        )

        graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASS,
            database=NEO4J_DB
        )

        # Create Document node
        document_id = make_id("document")
        graph.query(
            """
            MERGE (d:Document {id: $doc_id})
            SET d.name = $name, d.uploaded_at = datetime()
            """,
            {"doc_id": document_id, "name": file.filename}
        )

        # Label and link tables/rows
        for d in only_table_docs:
            nid = d.metadata["id"]
            ntype = d.metadata["type"]
            meta_json = json.dumps(d.metadata)

            graph.query(
                """
                MATCH (c:Chunk {id: $id})
                SET c.text = $text, c.metadata = $meta
                MERGE (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                {"id": nid, "text": d.page_content, "meta": meta_json, "doc_id": document_id}
            )

            if ntype == "table":
                graph.query("MATCH (c:Chunk {id:$id}) SET c:Table", {"id": nid})
            elif ntype == "row":
                table_id = d.metadata["table_id"]
                graph.query(
                    """
                    MATCH (r:Chunk {id:$rid}), (t:Chunk {id:$tid})
                    SET r:Row
                    MERGE (t)-[:HAS_ROW]->(r)
                    """,
                    {"rid": nid, "tid": table_id}
                )

    # ---- Store TEXT + IMAGES in Pinecone ----
    if non_table_docs:
        vector_store.add_documents(non_table_docs)

    return {
        "status": "ok",
        "filename": file.filename,
        "num_chunks": len(chunk_docs),
        "num_tables": len([t for t in table_docs if t.metadata["type"] == "table"]),
        "num_rows": len([t for t in table_docs if t.metadata["type"] == "row"]),
        "num_images": len(image_docs),
    }



async def query_graphrag(question: str):
    try:
        response = graph_qa.invoke({"query": question})
        return {"answer": response["answer"], "cypher": response.get("cypher", "")}

    except Exception as e:
        return {"error": str(e)}


async def merge_results(question: str):
    docs = vector_store.similarity_search(question, k=5)
    graph_docs = await query_graphrag(question)

    if graph_docs == [] or "error" in graph_docs:
        graph_docs = "No relevant data found in graph database."

    prompt_template = PromptTemplate(
        input_variables=["question", "docs", "graph_docs"],
        template="""
You are an expert at combining insights from unstructured text and structured graph data.

You will receive two types of context:
1. **Vector DB (text & image summaries):**
{docs}

2. **Graph DB (tables, relationships, structured facts):**
{graph_docs}

Task:
- Use both sources to answer the question as accurately as possible.
- Prefer Graph DB for factual/structured details, and Vector DB for descriptive/explanatory details.
- If the answer cannot be found in either context, reply exactly with: "I don't know based on the provided data."

Question: {question}
"""
    )

    chain = prompt_template | model | parser

    response = chain.invoke({
        "docs": docs,
        "graph_docs": graph_docs,
        "question": question
    })
    chathistory.append({"AI response": response})
    return {"answer": response}



async def queryenhancer(question: str):
    chathistory.append({"question": question})
    prompt_template = PromptTemplate(
        input_variables=["question", "chat_history"],
        template="""You are an Expert Query Enhancer.
You will be provided with a user question and the chat history.
and according to that i want you to recreate the question in a more enhanced way.
so that it is more descriptive and can give better results and will be used in Retriving the data from the vector database.
answer with only the enhanced question and nothing else.
Chat History:
{chat_history}
Question: {question}
Enhanced Question:"""
    )
    chain = prompt_template | model | parser
    response = chain.invoke({
        "question": question,
        "chat_history": chathistory
    })
    chathistory.append({"Enhanced question": response})
    return response


async def show_chat_history():
    return chathistory
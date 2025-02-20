import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config import config
from data_retrieval import PaperRetriever
from vector_store import VectorStore
from knowledge_graph import ResearchKnowledgeGraph
from agent_system import AgentSystem
from experiment_runner import ExperimentRunner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AIRA - Autonomous AI Research Agent")

class AIRA:
    def __init__(self):
        self.config = config
        self.paper_retriever = PaperRetriever(config)
        self.vector_store = VectorStore(config)
        self.knowledge_graph = ResearchKnowledgeGraph()
        self.agent_system = AgentSystem()
        self.experiment_runner = ExperimentRunner()
        
        # Initialize storage directories
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize the system"""
        logger.info("Initializing AIRA system...")
        
        # Start paper retrieval
        papers = await self.paper_retriever.process_papers()
        
        # Add papers to vector store
        self.vector_store.add_documents(papers)
        
        # Build knowledge graph
        for paper in papers:
            self.knowledge_graph.add_paper(paper)
            
        logger.info("AIRA system initialized successfully")
        
    async def update_knowledge(self):
        """Update system knowledge"""
        logger.info("Updating knowledge base...")
        
        # Fetch new papers
        new_papers = await self.paper_retriever.process_papers()
        
        # Update vector store and knowledge graph
        self.vector_store.add_documents(new_papers)
        for paper in new_papers:
            self.knowledge_graph.add_paper(paper)
            
        logger.info(f"Added {len(new_papers)} new papers to knowledge base")
        
    async def process_query(self, query: str, agent_type: str = "retrieval") -> Dict[str, Any]:
        """Process a user query"""
        try:
            # Execute query using appropriate agent
            result = await self.agent_system.execute_task(query, agent_type)
            
            # Enhance response with related information
            if agent_type == "retrieval":
                # Add vector search results
                vector_results = self.vector_store.search(query)
                result["vector_results"] = vector_results
                
                # Add knowledge graph insights
                if "paper_id" in result:
                    related_papers = self.knowledge_graph.find_related_papers(result["paper_id"])
                    result["related_papers"] = related_papers
                    
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize AIRA instance
aira = AIRA()

# API Models
class Query(BaseModel):
    text: str
    agent_type: str = "retrieval"

class ExperimentRequest(BaseModel):
    code: str
    environment: Optional[Dict[str, Any]] = None
    timeout: int = 3600

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize AIRA on startup"""
    await aira.initialize()

@app.post("/query")
async def process_query(query: Query):
    """Process a user query"""
    return await aira.process_query(query.text, query.agent_type)

@app.post("/update")
async def update_knowledge(background_tasks: BackgroundTasks):
    """Update knowledge base"""
    background_tasks.add_task(aira.update_knowledge)
    return {"status": "Update started"}

@app.post("/experiment")
async def run_experiment(request: ExperimentRequest):
    """Run a code experiment"""
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = await aira.experiment_runner.run_experiment(
        request.code,
        experiment_id,
        request.environment,
        request.timeout
    )
    return result

@app.get("/experiment/{experiment_id}")
async def get_experiment_status(experiment_id: str):
    """Get experiment status"""
    result = aira.experiment_runner.get_experiment_status(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store_size": len(aira.vector_store.store),
        "knowledge_graph_size": len(aira.knowledge_graph.graph),
        "agents": list(aira.agent_system.agents.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

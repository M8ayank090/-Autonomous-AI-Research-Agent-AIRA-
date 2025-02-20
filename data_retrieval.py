import arxiv
import requests
import datetime
from typing import List, Dict, Any
from scholarly import scholarly
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperRetriever:
    def __init__(self, config):
        self.config = config
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/v1"
        
    async def fetch_arxiv_papers(self, category: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Fetch papers from ArXiv for a given category"""
        try:
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            async for result in search:
                paper = {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary,
                    'published': result.published,
                    'pdf_url': result.pdf_url,
                    'arxiv_id': result.entry_id,
                    'categories': result.categories
                }
                papers.append(paper)
                
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching ArXiv papers: {str(e)}")
            return []

    async def fetch_semantic_scholar_paper(self, paper_id: str) -> Dict[str, Any]:
        """Fetch paper details from Semantic Scholar"""
        headers = {
            'x-api-key': self.config.SEMANTIC_SCHOLAR_API_KEY
        }
        
        try:
            response = requests.get(
                f"{self.semantic_scholar_base_url}/paper/{paper_id}",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching Semantic Scholar paper: {str(e)}")
            return {}

    async def fetch_citations(self, paper_id: str) -> List[Dict[str, Any]]:
        """Fetch citation information for a paper"""
        headers = {
            'x-api-key': self.config.SEMANTIC_SCHOLAR_API_KEY
        }
        
        try:
            response = requests.get(
                f"{self.semantic_scholar_base_url}/paper/{paper_id}/citations",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching citations: {str(e)}")
            return []

    async def process_papers(self) -> List[Dict[str, Any]]:
        """Main method to process papers from all sources"""
        all_papers = []
        
        # Fetch from all configured ArXiv categories
        for category in self.config.ARXIV_CATEGORIES:
            papers = await self.fetch_arxiv_papers(
                category, 
                self.config.MAX_PAPERS_PER_QUERY
            )
            
            # Enrich with Semantic Scholar data
            for paper in papers:
                semantic_data = await self.fetch_semantic_scholar_paper(paper['arxiv_id'])
                if semantic_data:
                    paper.update({
                        'citations': await self.fetch_citations(paper['arxiv_id']),
                        'influence_score': semantic_data.get('influenceScore'),
                        'citation_count': semantic_data.get('citationCount'),
                        'reference_count': semantic_data.get('referenceCount')
                    })
                    
            all_papers.extend(papers)
            
        return all_papers

    def save_papers(self, papers: List[Dict[str, Any]], output_path: str):
        """Save processed papers to disk"""
        import json
        from pathlib import Path
        
        output_file = Path(output_path) / f"papers_{datetime.date.today()}.json"
        with open(output_file, 'w') as f:
            json.dump(papers, f, indent=2)
        logger.info(f"Saved {len(papers)} papers to {output_file}")

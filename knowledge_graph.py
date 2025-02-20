from typing import List, Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    id: str
    type: str  # paper, author, concept, etc.
    properties: Dict[str, Any]
    created_at: datetime = datetime.now()

@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relationship: str
    weight: float
    properties: Dict[str, Any]
    created_at: datetime = datetime.now()

class ResearchKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.vectorizer = TfidfVectorizer()
        self.concept_embeddings = {}
        
    def add_paper(self, paper: Dict[str, Any]) -> str:
        """Add a paper to the knowledge graph"""
        paper_id = paper.get('arxiv_id') or paper.get('id')
        
        # Create paper node
        self.graph.add_node(
            paper_id,
            type='paper',
            title=paper['title'],
            authors=paper['authors'],
            published=paper['published'],
            summary=paper.get('summary', ''),
            properties=paper
        )
        
        # Add author nodes and relationships
        for author in paper['authors']:
            author_id = f"author_{author.replace(' ', '_')}"
            self.graph.add_node(
                author_id,
                type='author',
                name=author,
                properties={'papers': [paper_id]}
            )
            self.graph.add_edge(
                author_id,
                paper_id,
                relationship='authored',
                weight=1.0
            )
            
        # Extract and add concepts
        concepts = self._extract_concepts(paper['summary'])
        for concept, score in concepts:
            concept_id = f"concept_{concept.replace(' ', '_')}"
            self.graph.add_node(
                concept_id,
                type='concept',
                name=concept,
                properties={'papers': [paper_id]}
            )
            self.graph.add_edge(
                paper_id,
                concept_id,
                relationship='contains',
                weight=score
            )
            
        return paper_id
        
    def _extract_concepts(self, text: str) -> List[tuple[str, float]]:
        """Extract key concepts from text using TF-IDF"""
        # Convert text to TF-IDF features
        tfidf_matrix = self.vectorizer.fit_transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top scoring terms
        scores = zip(feature_names, tfidf_matrix.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Return top 10 concepts with scores
        return sorted_scores[:10]
        
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Add a relationship between nodes"""
        if properties is None:
            properties = {}
            
        self.graph.add_edge(
            source_id,
            target_id,
            relationship=relationship,
            weight=weight,
            properties=properties
        )
        
    def find_related_papers(self, paper_id: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """Find papers related to a given paper"""
        if paper_id not in self.graph:
            return []
            
        related_papers = []
        for node in nx.single_source_shortest_path_length(self.graph, paper_id, cutoff=max_distance):
            if node != paper_id and self.graph.nodes[node]['type'] == 'paper':
                distance = nx.shortest_path_length(self.graph, paper_id, node)
                paper_data = self.graph.nodes[node]
                related_papers.append({
                    'id': node,
                    'title': paper_data['title'],
                    'distance': distance,
                    'relationship_path': nx.shortest_path(self.graph, paper_id, node)
                })
                
        return sorted(related_papers, key=lambda x: x['distance'])
        
    def find_research_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """Find clusters of related research papers"""
        # Create paper similarity matrix
        papers = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'paper']
        texts = [self.graph.nodes[p]['summary'] for p in papers]
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create similarity graph
        similarity_graph = nx.Graph()
        for i, paper1 in enumerate(papers):
            for j, paper2 in enumerate(papers[i+1:], i+1):
                if similarity_matrix[i, j] > 0.3:  # Similarity threshold
                    similarity_graph.add_edge(paper1, paper2, weight=similarity_matrix[i, j])
                    
        # Find communities
        communities = list(nx.community.louvain_communities(similarity_graph))
        return [list(c) for c in communities if len(c) >= min_cluster_size]
        
    def get_author_impact(self, author_id: str) -> Dict[str, Any]:
        """Calculate impact metrics for an author"""
        if author_id not in self.graph:
            return {}
            
        # Get author's papers
        papers = [n for n in self.graph.neighbors(author_id)
                 if self.graph.nodes[n]['type'] == 'paper']
                 
        # Calculate metrics
        total_citations = sum(
            len(list(self.graph.predecessors(p)))
            for p in papers
        )
        
        h_index = self._calculate_h_index(papers)
        
        return {
            'name': self.graph.nodes[author_id]['name'],
            'paper_count': len(papers),
            'total_citations': total_citations,
            'h_index': h_index,
            'papers': papers
        }
        
    def _calculate_h_index(self, papers: List[str]) -> int:
        """Calculate h-index for a list of papers"""
        citations = [
            len(list(self.graph.predecessors(p)))
            for p in papers
        ]
        citations.sort(reverse=True)
        
        h = 0
        for i, c in enumerate(citations, 1):
            if c >= i:
                h = i
            else:
                break
        return h
        
    def visualize(self, output_path: str):
        """Create a visualization of the knowledge graph"""
        plt.figure(figsize=(15, 10))
        
        # Create position layout
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        node_colors = {
            'paper': 'lightblue',
            'author': 'lightgreen',
            'concept': 'lightpink'
        }
        
        for node_type, color in node_colors.items():
            nodes = [n for n, d in self.graph.nodes(data=True)
                    if d['type'] == node_type]
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=nodes,
                node_color=color,
                node_size=1000,
                alpha=0.7
            )
            
        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color='gray',
            arrows=True,
            alpha=0.5
        )
        
        # Add labels
        labels = {
            n: f"{d['type']}\n{d.get('title', d.get('name', ''))[:20]}..."
            for n, d in self.graph.nodes(data=True)
        }
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Research Knowledge Graph")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        
    def save_graph(self, filepath: str):
        """Save the knowledge graph to a file"""
        data = nx.node_link_data(self.graph)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    def load_graph(self, filepath: str):
        """Load the knowledge graph from a file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)

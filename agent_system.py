from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from config import config
import logging
from dataclasses import dataclass
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    """Track agent performance metrics"""
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    feedback_score: float = 0.0
    last_updated: datetime = datetime.now()

class CustomPromptTemplate(BaseChatPromptTemplate):
    """Custom prompt template for the agent system"""
    template: str
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> List[HumanMessage]:
        # Process intermediate steps
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action}\nObservation: {observation}\n"
            
        # Format tools and add to kwargs
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Add system message for agent type if specified
        messages = []
        if "system_message" in kwargs:
            messages.append(SystemMessage(content=kwargs["system_message"]))
            
        messages.append(HumanMessage(content=self.template.format(**kwargs)))
        return messages

class AgentOutputParser:
    """Parse agent outputs and validate actions"""
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
            
        # Extract action and input
        action_match = re.search(r"Action: (.*?)\nAction Input: (.*)", text, re.DOTALL)
        if not action_match:
            raise ValueError(f"Could not parse action from text: {text}")
            
        action = action_match.group(1).strip()
        action_input = action_match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=text)

class ResearchAgent:
    """Base class for research agents"""
    def __init__(
        self,
        agent_type: str,
        tools: List[Tool],
        memory: Optional[ConversationBufferMemory] = None,
        model_name: str = config.LLM_MODEL
    ):
        self.agent_type = agent_type
        self.tools = tools
        self.memory = memory or ConversationBufferMemory(
            memory_key=config.MEMORY_KEY,
            return_messages=True
        )
        self.metrics = AgentMetrics()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        
        # Create agent executor
        self.executor = self._create_executor()
        
    def _create_executor(self) -> AgentExecutor:
        """Create an agent executor with tools and memory"""
        prompt = CustomPromptTemplate(
            template=self._get_prompt_template(),
            tools=self.tools,
            input_variables=["input", "intermediate_steps", "chat_history", "system_message"]
        )
        
        # Create the agent
        agent = LLMSingleActionAgent(
            llm_chain=self.llm,
            output_parser=AgentOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
        
    def _get_prompt_template(self) -> str:
        """Get prompt template based on agent type"""
        templates = {
            "retrieval": """System: You are a research retrieval agent specializing in finding relevant academic papers and research content.

Previous conversation:
{chat_history}

Current task: {input}

Available tools:
{tools}

Think through this step-by-step:
1) What specific information am I looking for?
2) Which tools would be most helpful?
3) How should I combine and validate the results?

{agent_scratchpad}""",

            "summarization": """System: You are a research summarization agent that creates clear, structured summaries of academic content.

Previous conversation:
{chat_history}

Current task: {input}

Available tools:
{tools}

Think through this step-by-step:
1) What are the key points to extract?
2) How should I structure the summary?
3) What details need additional clarification?

{agent_scratchpad}""",

            "knowledge_graph": """System: You are a knowledge graph agent that connects and relates research concepts.

Previous conversation:
{chat_history}

Current task: {input}

Available tools:
{tools}

Think through this step-by-step:
1) What entities and relationships should I identify?
2) How do these concepts connect?
3) What patterns or clusters emerge?

{agent_scratchpad}""",

            "execution": """System: You are a code execution agent that runs experiments and analyzes results.

Previous conversation:
{chat_history}

Current task: {input}

Available tools:
{tools}

Think through this step-by-step:
1) What experiment setup is needed?
2) How should I validate the results?
3) What visualizations would be helpful?

{agent_scratchpad}"""
        }
        
        return templates.get(self.agent_type, templates["retrieval"])
        
    async def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute a task and track metrics"""
        start_time = datetime.now()
        
        try:
            result = await self.executor.arun(input=task)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.metrics.successful_tasks += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.successful_tasks - 1) + execution_time)
                / self.metrics.successful_tasks
            )
            self.metrics.last_updated = datetime.now()
            
            return {
                "status": "success",
                "result": result,
                "execution_time": execution_time,
                "agent_type": self.agent_type
            }
            
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            self.metrics.failed_tasks += 1
            return {
                "status": "error",
                "error": str(e),
                "agent_type": self.agent_type
            }
            
    def update_feedback(self, feedback_score: float):
        """Update agent's feedback score"""
        self.metrics.feedback_score = (
            (self.metrics.feedback_score * (self.metrics.successful_tasks + self.metrics.failed_tasks - 1)
            + feedback_score)
            / (self.metrics.successful_tasks + self.metrics.failed_tasks)
        )

class AgentSystem:
    """Manages multiple research agents and their interactions"""
    def __init__(self):
        self.agents: Dict[str, ResearchAgent] = {}
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> Dict[str, List[Tool]]:
        """Initialize tools for each agent type"""
        tools = {
            "retrieval": [
                Tool(
                    name="search_papers",
                    func=self._search_papers,
                    description="Search for research papers using keywords"
                ),
                Tool(
                    name="fetch_citations",
                    func=self._fetch_citations,
                    description="Fetch citation information for a paper"
                )
            ],
            "summarization": [
                Tool(
                    name="generate_summary",
                    func=self._generate_summary,
                    description="Generate a structured summary of a paper"
                ),
                Tool(
                    name="extract_key_points",
                    func=self._extract_key_points,
                    description="Extract key points from a paper"
                )
            ],
            "knowledge_graph": [
                Tool(
                    name="create_graph",
                    func=self._create_graph,
                    description="Create a knowledge graph from papers"
                ),
                Tool(
                    name="find_relationships",
                    func=self._find_relationships,
                    description="Find relationships between research concepts"
                )
            ],
            "execution": [
                Tool(
                    name="run_experiment",
                    func=self._run_experiment,
                    description="Run a code experiment"
                ),
                Tool(
                    name="analyze_results",
                    func=self._analyze_results,
                    description="Analyze experimental results"
                )
            ]
        }
        return tools
        
    def get_agent(self, agent_type: str) -> ResearchAgent:
        """Get or create an agent of specified type"""
        if agent_type not in self.agents:
            self.agents[agent_type] = ResearchAgent(
                agent_type=agent_type,
                tools=self.tools[agent_type]
            )
        return self.agents[agent_type]
        
    async def execute_task(self, task: str, agent_type: str) -> Dict[str, Any]:
        """Execute a task using the specified agent"""
        agent = self.get_agent(agent_type)
        return await agent.execute_task(task)
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        return {
            agent_type: agent.metrics for agent_type, agent in self.agents.items()
        }
        
    # Tool implementation methods
    async def _search_papers(self, query: str) -> str:
        """Implementation of paper search tool"""
        # Add actual implementation
        pass
        
    async def _fetch_citations(self, paper_id: str) -> str:
        """Implementation of citation fetching tool"""
        # Add actual implementation
        pass
        
    async def _generate_summary(self, text: str) -> str:
        """Implementation of summary generation tool"""
        # Add actual implementation
        pass
        
    async def _extract_key_points(self, text: str) -> str:
        """Implementation of key points extraction tool"""
        # Add actual implementation
        pass
        
    async def _create_graph(self, papers: List[Dict[str, Any]]) -> str:
        """Implementation of knowledge graph creation tool"""
        # Add actual implementation
        pass
        
    async def _find_relationships(self, concepts: List[str]) -> str:
        """Implementation of relationship finding tool"""
        # Add actual implementation
        pass
        
    async def _run_experiment(self, code: str) -> str:
        """Implementation of experiment execution tool"""
        # Add actual implementation
        pass
        
    async def _analyze_results(self, results: Dict[str, Any]) -> str:
        """Implementation of results analysis tool"""
        # Add actual implementation
        pass

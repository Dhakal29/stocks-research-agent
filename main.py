import uvicorn                                                                                                    
from starlette.applications import Starlette                                                                      
                                                                                                                    
from a2a.server.request_handlers import DefaultRequestHandler                                                     
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes                                     
from a2a.server.tasks import InMemoryTaskStore                                                                    
from a2a.types import (                                                                                           
    AgentCapabilities,                                                                                            
    AgentCard,                                                                                                    
    AgentInterface,                                                                                               
    AgentSkill,                                                                                                   
)                                                                                                                 
                                                                                                                    
from nepse_agent.agent import NepseResearchAgent, configure_research_logging
from nepse_agent.server import NepseAgentExecutor
                                                                                                                    
HOST = "127.0.0.1"                                                                                                
PORT = 8000                                                                                                       
                                                                                                                    
nepse_skill = AgentSkill(                                                                                         
    id="nepse_summarizer",                                                                                        
    name="Market and Web Research",
    description="Answers market summaries, stock questions, comparisons, news and other research queries using multiple web sources.",
    input_modes=["text/plain"],                                                                                   
    output_modes=["text/markdown", "application/json"],
    tags=["finance", "nepse", "nepal-share-market", "stocks", "web-search"],
    examples=["Give me today's market summary", "Compare NABIL and EBL", "Latest Nepal economic news", "NABIL"],
)                                                                                                                 
                                                                                                                    
agent_card = AgentCard(                                                                                           
    name="NEPSE Market Research Assistant",
    description="An A2A research agent answering natural-language questions with current evidence across the web, specializing in Nepal markets.",
    version="1.0.0",                                                                                              
    default_input_modes=["text/plain"],                                                                           
    default_output_modes=["text/markdown", "application/json"],
    capabilities=AgentCapabilities(streaming=False),                                                              
    supported_interfaces=[                                                                                        
        AgentInterface(                                                                                           
            protocol_binding="JSONRPC",                                                                           
            url=f"http://{HOST}:{PORT}",                                                                          
            protocol_version="1.0",                                                                               
        )                                                                                                         
    ],                                                                                                            
    skills=[nepse_skill],                                                                                         
)                                                                                                                 
                                                                                                                    
task_store = InMemoryTaskStore()                                                                                  
request_handler = DefaultRequestHandler(                                                                          
    agent_executor=NepseAgentExecutor(NepseResearchAgent()),
    task_store=task_store,
    agent_card=agent_card,
)                                                                                                                 
                                                                                                                    
app = Starlette(                                                                                                  
    routes=(                                                                                                      
        create_agent_card_routes(agent_card=agent_card)                                                           
        + create_jsonrpc_routes(request_handler=request_handler, rpc_url="/")
    )                                                                                                             
)                                                                                                                 
                                                                                                                    
if __name__ == "__main__":                                                                                        
    configure_research_logging()
    print(f"NEPSE A2A Agent running at: http://{HOST}:{PORT}")                                                    
    print(f"Discovery Card: http://{HOST}:{PORT}/.well-known/agent-card.json")
    uvicorn.run(app, host=HOST, port=PORT) 

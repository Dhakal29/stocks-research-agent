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
                                                                                                                    
from nepse_agent.agent import NepseResearchAgent
from nepse_agent.server import NepseAgentExecutor
                                                                                                                    
HOST = "127.0.0.1"                                                                                                
PORT = 8000                                                                                                       
                                                                                                                    
nepse_skill = AgentSkill(                                                                                         
    id="nepse_summarizer",                                                                                        
    name="NEPSE Stock Summarizer",                                                                                
    description="Takes a NEPSE stock symbol (e.g. NABIL, SHIVM, CHCL), searches recent web news, and returns an investment summary.",                                                                                               
    input_modes=["text/plain"],                                                                                   
    output_modes=["text/markdown", "application/json"],
    tags=["finance", "nepse", "nepal-share-market", "stocks"],                                                    
    examples=["NABIL", "HDL", "SHIVM", "Analyze CIT"],                                                            
)                                                                                                                 
                                                                                                                    
agent_card = AgentCard(                                                                                           
    name="NEPSE Stock Assistant",                                                                                 
    description="An A2A-compliant agent specializing in Nepal Stock Exchange (NEPSE) company research and news summaries.",                                                                                                        
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
    print(f"NEPSE A2A Agent running at: http://{HOST}:{PORT}")                                                    
    print(f"Discovery Card: http://{HOST}:{PORT}/.well-known/agent-card.json")
    uvicorn.run(app, host=HOST, port=PORT) 

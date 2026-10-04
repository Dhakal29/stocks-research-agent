import os                                                                                                         
from duckduckgo_search import DDGS                                                                                
from google import genai                                                                                          
                                                                                                                    
from a2a.helpers import (                                                                                         
    get_message_text,                                                                                             
    new_task_from_user_message,                                                                                   
    new_text_message,                                                                                             
    new_text_part,                                                                                                
)                                                                                                                 
from a2a.server.agent_execution import AgentExecutor, RequestContext                                              
from a2a.server.events import EventQueue                                                                          
from a2a.server.tasks import TaskUpdater                                                                          
from a2a.types import TaskState                                                                                   
                                                                                                                    
                                                                                                                    
class NepseStockAgent:                                                                                            
    """Agent that searches NEPSE stock details and summarizes via LLM."""                                         
                                                                                                                    
    def __init__(self):                                                                                           
        # Configure Gemini client (requires GEMINI_API_KEY environment variable)                                  
        api_key = os.getenv("GEMINI_API_KEY")                                                                     
        self.ai_client = genai.Client(api_key=api_key) if api_key else None                                       
                                                                                                                    
    def search_nepse_data(self, symbol: str) -> str:                                                              
        """Search recent news, market prices, and updates for the stock symbol."""                                
        query = f"NEPSE {symbol} stock price news Nepal Share Market"                                             
        search_results = []                                                                                       
                                                                                                                    
        try:                                                                                                      
            with DDGS() as ddgs:                                                                                  
                results = list(ddgs.text(query, max_results=5))                                                   
                for r in results:                                                                                 
                    search_results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.           
get('href')}\n")                                                                                                    
        except Exception as e:                                                                                    
            return f"Error fetching web results: {str(e)}"                                                        
                                                                                                                    
        return "\n---\n".join(search_results) if search_results else "No recent web data found."                  
                                                                                                                    
    async def summarize(self, symbol: str) -> str:                                                                
        symbol = symbol.strip().upper()                                                                           
                                                                                                                    
        # 1. Fetch web search context                                                                             
        web_context = self.search_nepse_data(symbol)                                                              
                                                                                                                    
        # 2. If Gemini API is available, summarize with LLM                                                       
        if self.ai_client:                                                                                        
            prompt = f"""                                                                                         
            You are a Nepal Stock Exchange (NEPSE) financial research assistant.                                  
            Here is recent web search information for company symbol '{symbol}':                                  
                                                                                                                    
            {web_context}                                                                                         
                                                                                                                    
            Please provide a concise analysis structured as:                                                      
            1. **Company Overview**: Name & industry sector.                                                      
            2. **Recent Market Insights / News**: Key takeaways from the search results.                          
            3. **Key Highlights / Cautions**: Any dividends, quarters report, price trends or warnings.           
            4. **Sources**: Mention the URLs provided in search results.                                          
            """                                                                                                   
            response = self.ai_client.models.generate_content(                                                    
                model="gemini-2.5-flash",                                                                         
                contents=prompt,                                                                                  
            )                                                                                                     
            return response.text                                                                                  
        else:                                                                                                     
            # Fallback if no GEMINI_API_KEY is provided                                                           
            return (                                                                                              
                f"### NEPSE Summary for {symbol} (Raw Data)\n\n"                                                  
                f"{web_context}\n\n"                                                                              
                f"*(Tip: Set GEMINI_API_KEY environment variable to enable AI summarization)*"                    
            )                                                                                                     
                                                                                                                    
                                                                                                                    
class NepseAgentExecutor(AgentExecutor):                                                                          
    def __init__(self):                                                                                           
        self.agent = NepseStockAgent()                                                                            
                                                                                                                    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:                            
        # 1. Initialize A2A Task                                                                                  
        task = context.current_task or new_task_from_user_message(context.message)                                
        await event_queue.enqueue_event(task)                                                                     
                                                                                                                    
        updater = TaskUpdater(                                                                                    
            event_queue=event_queue,                                                                              
            task_id=task.id,                                                                                      
            context_id=task.context_id,                                                                           
        )                                                                                                         
                                                                                                                    
        # 2. Read symbol from input message                                                                       
        symbol = get_message_text(context.message)                                                                
        if not symbol:                                                                                            
            symbol = "NABIL"                                                                                      
                                                                                                                    
        # 3. Send progress update to client                                                                       
        await updater.update_status(                                                                              
            state=TaskState.TASK_STATE_WORKING,                                                                   
            message=new_text_message(f"Searching NEPSE data and latest news for '{symbol}'..."),                  
        )                                                                                                         
                                                                                                                    
        # 4. Run the search & summarization                                                                       
        summary_result = await self.agent.summarize(symbol)                                                       
                                                                                                                    
        # 5. Return completed A2A task result                                                                     
        await updater.update_status(                                                                              
            state=TaskState.TASK_STATE_COMPLETED,                                                                 
            message=new_text_message(                                                                             
                role="agent",                                                                                     
                parts=[new_text_part(summary_result)],                                                            
            ),                                                                                                    
        )                                                                                                         
                                                                                                                    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:                             
        pass            
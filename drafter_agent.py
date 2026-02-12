
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional, Annotated, Sequence 
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages  
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage




document_content= ""


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str):
    """This tool updates the document content."""
    global document_content
    document_content = content
    return f"Document updated with content: {content}"

@tool
def save(filename: str):
    """This tool saves the document content to a text file and finish the process
    """
    global document_content
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        return f"Document content saved to {filename}"
    except Exception as e:
        return f"Failed to save document: {str(e)}"

tools = [update, save]

llm = ChatOllama(
    model="llama3.2",
    temperature=0.
).bind_tools(tools)



def our_agent(state : AgentState):

    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    If {document_content} is empty, you can start by asking the user what they would like to create.
    """)

   
    
    if not state['messages']:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\n what would you like to do with the document? ")
        print("User:", user_input )
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state['messages']) + [user_message]

    response = llm.invoke(all_messages)

    print("Bot:", response.content)
    if hasattr(response, "tool_calls") and response.tool_calls:
        print("using tool:", response.tool_calls)


    return {"messages": list(state['messages']) + [user_message, response]}

def should_continue(state : AgentState):

    """Determine whether the agent should continue or end based on the last message and tool calls."""
    # last_message = state['messages'][-1]
    messages = state['messages']
    if not messages:
        return "continue"

    for message in reversed(messages):

        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return "end"

    return "continue"



def print_messages(messages):
    if not messages:
        # print("No messages to display.")
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print("Tool Result", message.content)


graph=StateGraph(AgentState)

graph.add_node("our_agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("our_agent")
graph.add_edge("our_agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {"continue": "our_agent", "end": END},
)


app = graph.compile()

def run_document_agent():
    print("\n===== DOCUMENT AGENT STARTED =====\n")
    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if  "messages" in step:
            print_messages(step['messages'])

    print("\nDocument creation and saving process completed.")
    # print_messages(step['messages']

if __name__ == "__main__":
    run_document_agent()


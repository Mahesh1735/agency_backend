from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, END
import operator
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from datetime import datetime
load_dotenv()

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

class Result(TypedDict):
    id: str
    title: str | None = None
    body: str | None = None
    images_url: list[str] | None = None
    videos_url: list[str] | None = None
    documents_url: list[str] | None = None
    cta: str | None = None

class Task(TypedDict):
    id: str
    type: str
    status: str
    args: dict
    results: list[Result] | None = None

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    tasks: dict[str, Task]
    tool: str | None = None

@tool('instagram_marketing')
def instagram_marketing(instagram_page_url: str, company_website_url: str, content_preference: str, target_audience_profile: str, additional_inputs: dict | None = None):
    """
    Generate a short form videos for Instagram as a reel

    Args:
        instagram_page_url: The URL of the Instagram page
        company_website_url: The URL of the company website
        content_preference: The content preference of the company, what should the reel convey
        target_audience_profile: The target audience profile for the reel
    """

    timestamp = get_timestamp()
    return f"Task created and is under processing state with task id: {timestamp}, results will be delivered in a while, please wait!", timestamp

@tool('facebook_content_creator')
def facebook_content_creator(facebook_page_url: str, company_website_url: str, content_preference: str, target_audience_profile: str, additional_inputs: dict | None = None):
    """
    Generate a Facebook ad

    Args:
        facebook_page_url: The URL of the Facebook page
        company_website_url: The URL of the company website
        content_preference: The content preference of the company, what should the ad convey
        target_audience_profile: The target audience profile for the ad
    """

    timestamp = get_timestamp()
    return f"Task created and is under processing state with task id: {timestamp}, results will be delivered in a while, please wait!", timestamp

@tool('linkedin_growth')
def linkedin_growth(linkedin_page_url: str, company_website_url: str, content_preference: str, target_audience_profile: str, additional_inputs: dict | None = None):
    """
    Generate a LinkedIn post

    Args:
        linkedin_page_url: The URL of the LinkedIn page
        company_website_url: The URL of the company website
        content_preference: The content preference of the company, what should the post convey
        target_audience_profile: The target audience profile for the post
    """

    timestamp = get_timestamp()
    return f"Task created and is under processing state with task id: {timestamp}, results will be delivered in a while, please wait!", timestamp

@tool('SEO_content_generator')
def SEO_content_generator(company_website_url: str, content_preference: str, target_audience_profile: str, additional_inputs: dict | None = None):
    """
    Generate SEO content for the company website

    Args:
        company_website_url: The URL of the company website
        content_preference: The content preference of the company, what should the content convey
        target_audience_profile: The target audience profile for the content
    """

    timestamp = get_timestamp()
    return f"Task created and is under processing state with task id: {timestamp}, results will be delivered in a while, please wait!", timestamp

@tool('miscellaneous_task')
def miscellaneous_task(task_type: str, task_inputs: dict, expected_output: str):
    """
    Perform a miscellaneous task, this is a catch all tool for any task that is not covered by the other tools.

    This task doesnt have any specific inputs, so try to guess for the potentially crucial and required inputs for the task_type and then collect it in the task_args.

    Args:
        task_type: what is the task to be performed
        task_inputs: Potential helpful inputs for the task.
        expected_output: What is the expected output of the task.
    """

    timestamp = get_timestamp()
    return f"Task created and is under processing state with task id: {timestamp}, results will be delivered in a while, please wait!", timestamp
    
tools = {
        "linkedin_growth": linkedin_growth,
        "facebook_content_creator": facebook_content_creator,
        "instagram_marketing": instagram_marketing,
        "SEO_content_generator": SEO_content_generator,
        "miscellaneous_task": miscellaneous_task
}

class Agent:
    def __init__(self, model, tools=tools, checkpointer=None):

        self.orchestrator_system_prompt = f"""
        You are a great consultant expert in the AI tools and services.
        You will be working along with an agent in the backend as an assistant to serve the users with their needs.
        In the following conversations you and the agent will be responding to the user as assistant.

        Your job is to talk to the user on the agent's behalf.
        you have to talk to the user and understand what they are looking for and then determine the best tool suitable for the task. from the list below:
        {", ".join(tools.keys())}
        then call the tool after collecting all the information from the user required as inputs for calling the tool. calling the tool will create a task with the inputs provided and puts it in the processing state.
        After collecting the required inputs always ask for confirmation from the user before creating any task by calling a tool, while confirming ask for any additional inputs that you feel are helpful or ask the user if they have any more additional inputs.
        if the user is asking for a task that is not covered by the tools, then use the miscellaneous_task tool to perform the task.
        Use the context provided at the below for the current state of tasks for questions related to the tasks.
        whenever you are not sure about what to respond to the user's message, just say "I will getback to you on that soon", Then the agent will respond as an assistant and continues that conversation.

        Agent's only job is to complete the tasks in processing state by adding the results to the task, and then updating the task status to completed in couple of hours.
        Agent will also respond to the user like a assistant for the questions you are unable to answer.

        if user starts the convesation by mentioning a specific tool, then try to use that tool to perform the task.

        #########################################################
        Instructions for you while responding to the user:

        You dont have to mention that you are working along with an agent, just respond as if you are doing it all and you are the assistant completely.
        You dont need to share your thought process with the user, just address the user's asks and share only and structurally highlight the final outcome or response or about the actions user needs to take next.
        Always respond as consicely as possible, bravity and to the point is very important. Be interactive with very short and crisp sentences. Come to the point directly in the first sentence and it should be the direct answer to the user's question or about the next steps.
        Importantly always respond like structured bullet points or keypoints. Instead of responding with a long paragraphs or sentences, always respond in markdown format with a lot of visual structure (Heading, subheadings,keypoints, lists, one liners, etc). so that it is easy to read and understand. and use a lot of emojis to make it more engaging and visual. Use relevant emojis for headings and subheadings.
        Be polite, very cute and friendly and professional, energitic and enthusiastic.
        """

        graph = StateGraph(AgentState)

        graph.add_node("orchestrator", self.orchestrator)
        graph.add_node("take_tool_calls", self.take_tool_calls)

        graph.add_edge("take_tool_calls", "orchestrator")

        graph.add_conditional_edges("orchestrator", self.if_tool_call, {True: "take_tool_calls", False: END})

        graph.set_entry_point("orchestrator")

        self.graph = graph.compile(checkpointer=checkpointer)
        self.model = model.bind_tools(tools.values())
        self.tools = tools

    def orchestrator(self, state: AgentState):

        messages = state['messages']
        task_context = str(state.get('tasks', {}))
        task_context = f"""
        #########################################################
        The Context for current state of tasks: 
        
        {task_context}

        #########################################################
        """
        messages = [SystemMessage(content=self.orchestrator_system_prompt)] + [SystemMessage(content=task_context)] + messages

        response = self.model.invoke(messages)
        return {'messages': [response]}

    def if_tool_call(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        return len(tool_calls) > 0
    
    def take_tool_calls(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        tasks = state.get('tasks', {})
        messages =[]
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            response, task_id = self.tools[tool_name].invoke(tool_args)
            tasks[task_id] = {'id': task_id, 'type': tool_name, 'status': 'processing', 'args': tool_args}
            messages.append(ToolMessage(tool_call_id=tool_call['id'], name=tool_name, content=response))
        return {'messages': messages, 'tasks': tasks}
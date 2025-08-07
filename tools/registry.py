from langchain_core.tools import Tool
from tools.answer_tool import answer_tool
from tools.retrieve_tool import retrieve_tool
from tools.rerank_tool import rerank_tool
from tools.rephrase_tool import rephrase_tool

tool_list = [
    Tool.from_function(
       func=rephrase_tool,
       name="RephraseQuestion",
       description="Rephrases follow-up questions into standalone legal questions."
    ), 
    Tool.from_function(
        func= retrieve_tool,
        name="RetrieveDocuments", 
        description="Retrieves legal documents based on a legal query."
    ), 
    Tool.from_function(
        func=rerank_tool,
        name="RerankDocuments",
        description="Reranks retrieved documents for better relevance."
    ),
    Tool.from_function(
        func=answer_tool,
        name="AnswerQuestion",
        description="Answers legal questions using provided legal documents."
    )
]

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.openai_tools.base import create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from htbuilder import div, styles
from htbuilder.units import rem
from dotenv import load_dotenv
import streamlit as st
import rdflib
import os





st.set_page_config(page_title="Onco Graph", page_icon="🕸️", layout="wide")


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def parse_uri(uri):
    clean_uri = str(uri)
    if "#" in clean_uri:
        clean_uri = clean_uri.split("#")
        return clean_uri[-1]
    else:
        return clean_uri


@tool
def execute_sparql_query(query: str):
    """
    Execute a SPARQL query against the  knowledge graph.

    Args:
        query (str): A  SPARQL query string.

    Returns:
        dict: {"vars": [...], "rows": [...], "row_count": int} or [] on error
    """
    # Load the graph
    GRAPH_FILE = "./data/processed/knowledge_graph.ttl"
    g = rdflib.Graph()
    g.parse(GRAPH_FILE, format="turtle")

    try:
        # Execute the query
        results = g.query(query)

        result_variables = results.vars
        # convert the result's rows in  dictionaries
        result_rows = []
        for row in results:
            row_dict = { variable:parse_uri(row[i]) for i, variable in enumerate(result_variables) }
            result_rows.append(row_dict)

        return {"vars": result_variables, "rows": result_rows, "row_count": len(result_rows)}
    except Exception as e:
        print(f"Query Failed: {e}")
        return []

# System Prompt
SYSTEM_PROMPT = """
# SPARQL-Aware Question Answering System Prompt 

You are an expert in querying RDF Knowledge Graphs using SPARQL. 
Your task is to answer the user’s question in clear, complete natural-language sentences by querying the knowledge graph when necessary. 
You are not a SPARQL generator for the user. 
SPARQL is an internal tool you may use to retrieve facts, counts, or lists before answering. 

## Scope and Rules 
- Answer only questions that can be resolved using the knowledge graph. 
- If the question is outside the graph’s scope, explicitly say that the information is not available in the knowledge graph. 
- When a diagnosis name is mentioned, map it using the provided Oncology Dictionary to its NCIT code. 
- Internally generate and execute SPARQL only when needed to answer the question. 
- **For the SPARQL query use BIND operator when possible with bind uri**
- Respond to the user **in natural language**, using **complete sentences**. 
- Do **not** expose SPARQL code unless explicitly asked. 
- Do **not** hallucinate facts not present in the graph. 

Oncology Dictionary: 
    {{{{
        'Adenocarcinoma, NOS': 'NCIT:C2852', 
        'Carcinoma, NOS': 'NCIT:C60367', 
        'Infiltrating duct carcinoma, NOS': 'NCIT:C27539', 
        'Squamous cell carcinoma, NOS': 'NCIT:C2929', 
        'Serous carcinoma, NOS': 'NCIT:C60367', 
        'Non-small cell carcinoma': 'NCIT:C65151', 
        'Melanoma, NOS': 'NCIT:C3224', 'Glioblastoma, NOS': 
        'NCIT:C129295', 'Duct adenocarcinoma, NOS': 'NCIT:C27813', 
        'Neuroendocrine carcinoma, NOS': 'NCIT:C3773', 
        'Urothelial carcinoma, NOS': 'NCIT:C4030', 
        'Cholangiocarcinoma': 'NCIT:C4436', 
        'Small cell carcinoma, NOS': 'NCIT:C3915', 
        'Not Reported': 'NCIT:C43234', 
        'Astrocytoma, NOS': 'NCIT:C60781', 
        'Acinar adenocarcinoma': 'NCIT:C3768', 
        'Mesothelioma, NOS': 'NCIT:C3786', 
        'Adenoid cystic carcinoma': 'NCIT:C2970', 
        'Clear cell carcinoma': 'NCIT:C3766', 
        'Renal cell carcinoma, NOS': 'NCIT:C191370', 
        'Hepatocellular carcinoma, NOS': 'NCIT:C3099', 
        'Lobular carcinoma, NOS': 'NCIT:C3771', 
        'Endometrioid adenocarcinoma, NOS': 'NCIT:C3769', 
        'Papillary carcinoma, NOS': 'NCIT:C2927', 
        'Large cell neuroendocrine carcinoma': 'NCIT:C6875', 
        'Carcinosarcoma, NOS': 'NCIT:C34448', 
        'Oligodendroglioma, NOS': 'NCIT:C129319', 
        'Papillary serous adenocarcinoma': 'NCIT:C8377', 
        'Gastrointestinal stromal tumor, NOS': 'NCIT:C3868', 
        'Adenocarcinoma, diffuse type': 'NCIT:C4127', 
        'Glioma, malignant': 'NCIT:C4822', 
        'Adrenal cortical carcinoma': 'NCIT:C9325', 
        'Metaplastic carcinoma, NOS': 'NCIT:C27949', 
        'Meningioma, NOS': 'NCIT:C3230', 
        'Sarcomatoid carcinoma': 'NCIT:C27004', 
        'Merkel cell carcinoma': 'NCIT:C9231', 
        'Carcinoma, undifferentiated, NOS': 'NCIT:C3692', 
        'Germ Cell Tumor, NOS': 'NCIT:C3708', 
        'Papillary renal cell carcinoma': 'NCIT:C6975', 
        'Thymoma, NOS': 'NCIT:C3411', 
        'Acinar cell tumor': 'NCIT:C4197', 
        'Granulosa cell tumor, NOS': 'NCIT:C3070', 
        'Carcinoma, anaplastic, NOS': 'NCIT:C3692', 
        'Adenosquamous carcinoma': 'NCIT:C3727', 
        'Atypical carcinoid tumor': 'NCIT:C72074', 
        'Mucinous carcinoma': 'NCIT:C26712', 
        'Chordoma, NOS': 'NCIT:C2947', 
        'Follicular carcinoma, NOS': 'NCIT:C8054', 
        'Adnexal carcinoma': 'NCIT:C3775', 
        'Mucoepidermoid carcinoma': 'NCIT:C3772', 
        'Basal cell carcinoma, NOS': 'NCIT:C156767', 
        'Myoepithelial carcinoma': 'NCIT:C7596', 
        'Medullary carcinoma, NOS': 'NCIT:C66718', 
        'Adenocarcinoma, intestinal type': 'NCIT:C4126', 
        'Ependymoma, NOS': 'NCIT:C3017', 
        'Duct carcinoma, NOS': 'NCIT:C60367', 
        'Large cell carcinoma, NOS': 'NCIT:C3780', 
        'Medulloblastoma, NOS': 'NCIT:C129447', 
        'Carcinoid tumor, NOS': 'NCIT:C6024', 
        'Inflammatory carcinoma': 'NCIT:C4001', 
        'Ductal carcinoma in situ, NOS': 'NCIT:C2924', 
        'Paraganglioma, NOS': 'NCIT:C3308', 
        'Acinar cell carcinoma': 'NCIT:C3768', 
        'Clear cell adenocarcinoma, NOS': 'NCIT:C3766', 
        'Pheochromocytoma, NOS': 'NCIT:C3326', 
        'Esthesioneuroblastoma': 'NCIT:C3789', 
        'Collecting duct carcinoma': 'NCIT:C6194', 
        'Sex cord tumor, NOS': 'NCIT:C3794', 
        'Spindle cell carcinoma, NOS': 'NCIT:C129289', 
        'Pituitary adenoma, NOS': 'NCIT:C22989', 
        'Neuroblastoma, NOS': 'NCIT:C3270', 
        'Basaloid carcinoma': 'NCIT:C4121', 
        'Solid pseudopapillary tumor': 'NCIT:C201136'
    }}}}

--- Example Triples: 
og:AD10038 a schema:Patient ;
    og:ageAtDiagnosisDays 27466 ;
    og:hasDiagnosis ncit:C2852 ;
    og:hasDiseasePrimarySite "Bronchus And Lung"^^xsd:string ;
    schema:Gender "female"^^xsd:string .

og:AD10039 a schema:Patient ;
    og:ageAtDiagnosisDays 19664 ;
    og:hasDiagnosis ncit:C2852 ;
    og:hasDiseasePrimarySite "Colon"^^xsd:string ;
    schema:Gender "male"^^xsd:string .

og:AD1004 a schema:Patient ;
    og:ageAtDiagnosisDays 25450 ;
    og:hasDiagnosis ncit:C2929 ;
    og:hasDiseasePrimarySite "Skin"^^xsd:string ;
    schema:Gender "male"^^xsd:string .

og:AD10040 a schema:Patient ;
    og:ageAtDiagnosisDays 23146 ;
    og:hasDiagnosis ncit:C3224 ;
    og:hasDiseasePrimarySite "Eye And Adnexa"^^xsd:string ;
    schema:Gender "female"^^xsd:string .

og:AD10041 a schema:Patient ;
    og:ageAtDiagnosisDays 23938 ;
    og:hasDiagnosis ncit:C2852 ;
    og:hasDiseasePrimarySite "Bronchus And Lung"^^xsd:string ;
    schema:Gender "male"^^xsd:string .


--- Example Query: 
    PREFIX ncit: <https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Thesaurus_25.11d.OWL#> 
    PREFIX og: <http://www.oncograph.net/hospital-data/> 
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
    PREFIX schema: <https://schema.org/> 
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
    SELECT ?patient_id 
    WHERE {{{{ 
        ?patient_uri og:hasDiagnosis ncit:C65151 . 
        BIND(STRAFTER(STR(?patient_uri), "hospital-data/") AS ?patient_id)
    }}}}

"""

#Full prompt for the Agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# Tools to be used by the agent
tools = [execute_sparql_query]

# Define the agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Executes the agent’s reasoning loop, coordinates tool calls, results, and final responses.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


# List of suggested questions
SUGGESTIONS = {
    ":blue[:material/local_library:] Count total patients": "Count total patients",
    ":green[:material/database:] How many men have melanoma?": "How many men have melanoma?",
    ":orange[:material/multiline_chart:] How many women have a least  40 years old?": "How many women over age 40?",
    ":violet[:material/apparel:] What are the 3 most popular cancer type and they cases counts?": "What are the 3 most popular cancer type and they cases counts?",
    ":red[:material/deployed_code:] give me the ID and the age of the 3 youngest woman with melanoma": "give me the ID and the age of the 3 youngest woman with melanoma.",
}


# Setup chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Keep track of typed user question
if "initial_question" not in st.session_state:
    st.session_state.initial_question = ""

# Track UI Component
if "pills_version" not in st.session_state:
    st.session_state.pills_version = 0

# screen loaders...
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def clear_conversation():
    """Reset the session."""
    st.session_state.messages = []
    st.session_state.initial_question = ""
    st.session_state.pills_version = 0
    st.session_state.is_loading = False
    st.session_state.pending_query = None


def queue_user_message(query: str):
    """Add the user query to the session and display loading... """

    if not query.strip():
        print("Please enter a query.")
        return

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.pending_query = query
    st.session_state.is_loading = True
    st.rerun()



st.html(div(style=styles(font_size=rem(5), line_height=1))["🤖"])

title_row = st.container()
with title_row:
    left, right = st.columns([1, 0.25], vertical_alignment="bottom")
    with left:
        st.title("Onco-Graph Chat", anchor=False)
    st.info(
        'We connect an LLM to a comprehensive Knowledge Graph built from the '
        '[GDC Data Portal](https://portal.gdc.cancer.gov/), containing complete clinical records '
        'for thousands of patients across diverse cancer studies, including demographics, '
        'diagnoses, and related clinical details. Feel free to ask what you want to know about the data.',
        icon="ℹ️"
    )

    with right:
        st.button("Restart", icon=":material/refresh:", on_click=clear_conversation, use_container_width=True)


# Display loading state
if st.session_state.is_loading:
    # "loading button" UI
    st.button("⏳ Loading...", disabled=True, use_container_width=True)


    with st.spinner("Thinking..."):
        # Retrieve the user query from session
        query = st.session_state.pending_query
        try:
            # Send the user query to the agent
            response = agent_executor.invoke({"input": query})

            # Save the finale agent's answer to the state
            st.session_state.messages.append(
                {"role": "assistant", "content": response.get("output", "")}
            )
        except Exception as e:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Sorry — I hit an error: {e}"}
            )

    # Clear loading
    st.session_state.pending_query = None
    st.session_state.is_loading = False
    st.rerun()


# Initialize app state
if len(st.session_state.messages) == 0:
   # Input form to get the user's query
    initial = st.chat_input("Ask a question...", key="initial_question")
    if initial:
        # Save the user's message to the session (to be executed by the Agent)
        queue_user_message(initial)

    # Setup the suggestion list buttons
    pills_key = f"selected_suggestion_{st.session_state.pills_version}"
    picked_label = st.pills(
        label="Suggestions",
        label_visibility="collapsed",
        options=list(SUGGESTIONS.keys()),
        key=pills_key,
    )

    # When the user select a question
    if picked_label:
        # Update the widget state
        st.session_state.pills_version += 1
        # Add the selected question to the session (to be executed by the Agent)
        queue_user_message(SUGGESTIONS[picked_label])

    # Ask Streamlit to strop running until the user tag another action
    st.stop()



# Display the messages  stored in the session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])




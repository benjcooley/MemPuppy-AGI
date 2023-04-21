import openai
import os
import pinecone
import yaml
from dotenv import load_dotenv
import nltk
from langchain.text_splitter import NLTKTextSplitter
from datetime import datetime
import urllib
from bs4 import BeautifulSoup

# Download NLTK for Reading
nltk.download('punkt')

# Initialize Text Splitter
text_splitter = NLTKTextSplitter(chunk_size=2500)

# Load default environment variables (.env)
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4"

def generate(context, query, messages):
    context_message = []
    if context != "":
        context_message = [{"role": "user", "content": context}]
    query_message = []
    if query != "":
        query_message = [{"role": "user", "content": query}]
    completion = openai.ChatCompletion.create(
    model=OPENAI_MODEL,
    messages=[
        {"role": "system", "content": "You are an intelligent agent with thoughts and memories. You have a memory which stores" +\
         "your past thoughts and actions and also how this user has interacted with you." },
        {"role": "system", "content": "If you have something you'd like to remember during the conversation, you have something you want to remember you can write REMEMBER THIS: <something " +\
         "to remember> followed by your response with RESPONSE: <response>. Make sure to only use those two prompts when responding though. This is optional and you can just respond normally if you wish." },
        {"role": "system", "content": "Keep your thoughts relatively simple and concise."}
        ] + messages + query_message + context_message
    )

    response = completion.choices[0].message["content"]
    messages += query_message + [{"role": "assistant", "content": response}]

    return response

def getHtmlText(url):
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
#PINECONE_API_ENV = "asia-southeast1-gcp"
    
# Prompt Initialization
with open('prompts.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

# Counter Initialization
with open('memory_count.yaml', 'r') as f:
    counter = yaml.load(f, Loader=yaml.FullLoader)

# internalThoughtPrompt = data['internal_thought']
# externalThoughtPrompt = data['external_thought']
# internalMemoryPrompt = data['internal_thought_memory']
# externalMemoryPrompt = data['external_thought_memory']

# Thought types, used in Pinecone Namespace
THOUGHTS = "Thoughts"
QUERIES = "Queries"
INFORMATION = "Information"
ACTIONS = "Actions"
FACTS = "Facts"

# Top matches length
k_n = 3

# initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# initialize openAI
openai.api_key = OPENAI_API_KEY # you can just copy and paste your key here if you want

def get_ada_embedding(text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]


class Agent():
    def __init__(self, table_name, user_name) -> None:
        self.messages = []
        self.previous_memories = ""
        self.previous_response = ""
        self.table_name = table_name
        self.user_name = user_name
        self.memory = None
        self.thought_id_count = int(counter['count'])
        self.last_message = ""
        self.logging = False
        self.first_query = True

    # Keep Remebering!
    # def __del__(self) -> None:
    #     with open('memory_count.yaml', 'w') as f:
    #         yaml.dump({'count': str(self.thought_id_count)}, f)
    

    def createIndex(self, table_name=None):
        # Create Pinecone index
        if(table_name):
            self.table_name = table_name

        if(self.table_name == None):
            return

        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )

        # Give memory
        self.memory = pinecone.Index(self.table_name)

    # Adds new Memory to agent, types are: THOUGHTS, ACTIONS, QUERIES, INFORMATION
    def updateMemory(self, new_thought, thought_type):
        with open('memory_count.yaml', 'w') as f:
             yaml.dump({'count': str(self.thought_id_count)}, f)

        now = str(datetime.now())

        if thought_type==ACTIONS:
            # Not needed since already in prompts.yaml as external thought memory
            return

        vector = get_ada_embedding(new_thought)
        upsert_response = self.memory.upsert(
        vectors=[
            {
            'id':f"thought-{self.thought_id_count}", 
            'values':vector, 
            'metadata':
                {"thought_string": new_thought,
                 "time": now
                }
            }],
	    namespace=thought_type,
        )

        self.thought_id_count += 1

    def updateFacts(self, facts):
        with open('memory_count.yaml', 'w') as f:
             yaml.dump({'count': str(self.thought_id_count)}, f)

        now = str(datetime.now())

        for fact in facts:
            if len(fact) == 2:
                if "Unknown" in fact[1] or "No specific" in fact[1] or "was not provided" in fact[1]:
                    continue
                vector = get_ada_embedding(fact[0])
                upsert_response = self.memory.upsert(
                vectors=[
                    {
                    'id':f"thought-{self.thought_id_count}", 
                    'values':vector, 
                    'metadata':
                        {"thought_string": fact[0] + " ANSWER: " + fact[1],
                        "time": now
                        }
                    }],
                namespace=FACTS,
                )
                self.thought_id_count += 1

    def queryFacts(self, questions, top_k=5):
        results = []
        for question in questions:
            query_embedding = get_ada_embedding(question)
            query_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=FACTS)
            results += query_results["matches"]
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        del sorted_results[top_k:]
        return "\nANSWERS:\n\n" + "\n".join([(str(item.metadata["thought_string"])) for item in sorted_results])

    # Agent thinks about given query based on top k related memories. Internal thought is passed to external thought
    def internalThought(self, query) -> str:

        results = []

        if self.first_query:
            user_embedding = get_ada_embedding("Who is {user}?".replace("{user}", self.user_name))
            user_results = self.memory.query(user_embedding, top_k=2, include_metadata=True, namespace=THOUGHTS)
            subject_embedding = get_ada_embedding("What was the previous conversaton with this user about?")
            subject_results = self.memory.query(subject_embedding, top_k=2, include_metadata=True, namespace=THOUGHTS)
            results = results + user_results.matches + subject_results.matches

        query_embedding = get_ada_embedding(query)
        query_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=QUERIES)
        thought_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=THOUGHTS)
        results = results + query_results.matches + thought_results.matches

        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        top_matches = "\n\n".join([(str(item.metadata["thought_string"])) for item in sorted_results])
        if self.logging:
            print(top_matches)
        
        internalThoughtPrompt = data['internal_thought']
        internalThoughtPrompt = internalThoughtPrompt \
            .replace("{query}", query) \
            .replace("{previous_memories}", self.previous_memories) \
            .replace("{top_matches}", top_matches) \
            .replace("{previous_response}", self.previous_response)
        #if self.logging:
        #    print("------------INTERNAL THOUGHT PROMPT------------")
        #    print(internalThoughtPrompt)
        internal_thought = generate("", internalThoughtPrompt, []) # OPENAI CALL: top_matches and query text is used here
        
        # Debugging purposes
        if self.logging:
            print("------------INTERNAL THOUGHT ------------")
            print(internal_thought)

        preproc_thoughts = internal_thought \
            .replace("None.", "") \
            .replace("None", "") \
            .replace("N/A", "") \
            .replace("MEMORIES SUMMARY:", "~") \
            .replace("THINGS I WANT TO REMEMBER:", "~") \
            .replace("QUESTIONS I HAVE:", "~") \
            .replace("-->", "^") \
            .strip("~\n ") \
            .split("~")

        memories_summary = []
        remember_list = []
        questions_list = []

        try:
            memories_summary = nltk.sent_tokenize(preproc_thoughts[0].strip("\n "))
            remember_list = [[q.strip(" ") for q in v.split("^")] for v in preproc_thoughts[1].strip("\n ").split("\n")]
            questions_list = nltk.sent_tokenize(preproc_thoughts[2].strip("\n "))
        except:
            pass

        # Only keep the actions for the internal though
        memories = " ".join(memories_summary)

        self.updateFacts(remember_list)
        answers = self.queryFacts(questions_list)

        internalMemoryPrompt = data['internal_thought_memory']
        internalMemoryPrompt = internalMemoryPrompt \
            .replace("{now}", str(datetime.now())) \
            .replace("{query}", query) \
            .replace("{memories}", memories)
        self.updateMemory(internalMemoryPrompt, THOUGHTS)

        return memories, answers

    def action(self, query) -> str:

        if query == "logging: on":
            self.logging = True
            print("Logging enabled.")
            return
        elif query == "logging: off":
            self.logging = False
            print("Logging disabled.")
            return
        
        memories, answers = self.internalThought(query)
        
        contextPrompt = data['context_prompt']
        contextPrompt = contextPrompt \
            .replace("{memories}", memories) \
            .replace("{answers}", answers) \
            .replace("{now}", str(datetime.now()))
        if self.logging:
            print("------------EXTERNAL THOUGHT PROMPT------------")
            print(contextPrompt)
        external_thought = generate(contextPrompt, query, self.messages) # OPENAI CALL: top_matches and query text is used here

        if self.logging:
            print("------------EXTERNAL THOUGHT------------")
            print(external_thought)

        processed_thoughts_map = {}

        if "RESPONSE:" in external_thought:
            processed_thoughts = external_thought \
                .replace("REMEMBER THIS:", "~memory^") \
                .replace("RESPONSE:", "~response^") \
                .strip("\n ~^") \
                .split("~")
            for thought in processed_thoughts:
                pair = thought.split("^")
                processed_thoughts_map[pair[0].strip(" \n")] = pair[1].strip(" \n")
        else:
            processed_thoughts_map["response"] = external_thought

        external_memories = ""

        if len(processed_thoughts_map) == 2:
            external_thought = processed_thoughts_map["response"]
            external_memories = processed_thoughts_map["memory"]
        else:
            external_thought = processed_thoughts_map["response"]

        externalMemoryPrompt = data['external_thought_memory']
        externalMemoryPrompt = externalMemoryPrompt \
            .replace("{now}", str(datetime.now())) \
            .replace("{query}", query) \
            .replace("{external_thought}", external_thought)
        self.updateMemory(externalMemoryPrompt, THOUGHTS)
        
        requestMemoryPrompt = data["request_memory"]
        requestMemoryPrompt = requestMemoryPrompt \
            .replace("{now}", str(datetime.now())) \
            .replace("{query}", query) \
            .replace("{response}", external_thought)
        self.updateMemory(requestMemoryPrompt, QUERIES)
        
        self.previous_memories = memories + "\n" + external_memories
        self.previous_response = external_thought
        self.first_query = False

        if self.logging:
            print("------------ RESPONSE ------------")
        
        return external_thought

    # Make agent think some information
    def think(self, text) -> str:
        self.updateMemory(text, THOUGHTS)

    # Make agent read some information
    def read(self, url) -> str:
        try:
            text = getHtmlText(url)
        except:
            print(f"Couldn't read {url}.")
            return

        summaryRequest = data['wepbage_summary_request']
        summaryRequest = summaryRequest \
            .replace("{text}", text[0:5000])
        
        try:
            summary = generate("", summaryRequest, []) 
        except:
            print(f"Couldn't summarize {url}.")
            return            

        summaryPrompt = data['wepbage_summary']
        summaryPrompt = summaryPrompt \
            .replace("{now}", str(datetime.now())) \
            .replace("{url}", url) \
            .replace("{summary}", summary)
        
        self.messages += [{"role": "assistant", "content": summaryPrompt}]

        self.updateMemory(summaryPrompt, THOUGHTS)

        print(summaryPrompt)
        print(generate(summaryPrompt, "", []))



   

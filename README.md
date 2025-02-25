# MemPuppy-AGI -- A ChatGPT which remembers who you are and everything you've talked about

Derived from the very nice Teenage-AGI by [@sean_pixel](https://twitter.com/sean_pixel) which was a fork of Baby-AGI by [@yoheinakajima](https://twitter.com/yoheinakajima).

## Objective (Teenage-AGI, MemPuppy-AGI)
Inspired by the several Auto-GPT related Projects (predominently BabyAGI) and the Paper ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442), this python project uses OpenAI and Pinecone to Give memory to an AI agent and also allows it to "think" before making an action (outputting text). Also, just by shutting down the AI, it doesn't forget its memories since it lives on Pinecone and its memory_counter saves the index that its on.

## Changes to Teenage-AGI for MemPuppy-AGI
- Rewrote Teenage's memory and thinking engine to use facts summarization and contextual summarized memory.
- Removed ACTIONS from the INTERNAL thought as it was influencing the chat and making it not consistent.
- Simplified the EXTERNAL thoughts to avoid distracting the AI during a reply with too much contextual info.
- Complete chat history so it works like GPT, knows what you've talked about in detail in the current session.
- Timestamps on memories so GPT knows when you talked about what. Helps it figure out context.

### Sections
- [How it Works](https://github.com/seanpixel/Teenage-AGI/blob/main/README.md#how-it-works)
- [How to Use](https://github.com/seanpixel/Teenage-AGI/blob/main/README.md#how-to-use)
- [Experiments](https://github.com/seanpixel/Teenage-AGI/blob/main/README.md#experiments)
- [Credits](https://github.com/seanpixel/Teenage-AGI/blob/main/README.md#credits)

## How it Works
### Here is what happens everytime the AI is queried by the user:
1. AI vectorizes the query and stores it in a Pinecone Vector Database
2. AI looks inside its memory and finds memories and past queries and FACTS that are relevant to the current query. If the query is the first query it searches for the user name and what the user was previously doing.
3. AI updates it's contextual memory summary paragraph and adds it to the memory data set.
4. AI thinks of five relevant simple facts to add to the facts vector data set.
5. AI thinks of five new questions to answer.
6. The database is searched for answers to the questions asked in step 5.
7. The chat is then appended with an "assistant" entry with the current memory summary, and the results of the questions query from step 6. This provides the 'context' for answering the next user query. (This context entry is not permanently stored in the chat.)
5. The user query is appended to the chat and a response is requested from the AI. The AI can also privately append information to his memory summary in this response.
6. AI stores the current query and its answer in its Pinecone vector database memory
7. (Timestamps are added to all memories to allow GPT to reason about time)

### The improvements of MemPuppy over Teenage-AGI are the following:
1. Facts vectors are easier for the search to identify and include than queries and responses. Facts can be atomic memories unrelated to contextual conversation, and GPT's recall of these facts can be more direct.
2. A contextual memory summary helps the AI understand what the subject of the conversation is, store that in a searchable form in the database, and append and modify it as the chat progresses. This contextual memory is usually the most relevant when recovering context in later chats.
3. Removing the ACTIONS block makes the conversation with the AI more natural.
4. Using the entire chat context in the chat in a standard form makes the AI behave more like traditional GPT. It knows what it's previously said and it's contextual focus is the end of the chat.
5. Timestamps allow GPT to understand when certain conversations occured and whether facts might be relevant now.
6. "read:" now reads a webpage from the internet.

Generally, MemPuppy just acts like GPT with a reasonably good long term memory. If you use it with logging disabled it will just appear to be GPT, who knows and recognizes you and remembers everything you've ever talked about but with a slightly long pause between responses.

## How to Use
1. Clone the repository via `git clone https://github.com/seanpixel/Teenage-AGI.git` and cd into the cloned repository.
2. Install required packages by doing: pip install -r requirements.txt
3. Create a .env file from the template `cp .env.template .env`
4. `open .env` and set your OpenAI and Pinecone API info.
5. Run `python main.py` and talk to the AI in the terminal

## Running in a docker container
You can run the system isolated in a container using docker-compose:
```
docker-compose run teenage-agi
```

## Experiments
Works just like ChatGPT in chat mode, but just remembers who you are and everything you've chatted about. Occasionally will have to ask you for a hint but usually only one. Really interesting how it behaves like an old friend who's chatting with you on the phone.

## Credits
Thank you to [@sean_pixel](https://twitter.com/sean_pixel) who wrote Teenage-AGI, [@yoheinakajima](https://twitter.com/yoheinakajima) who wrote Baby-AGI.

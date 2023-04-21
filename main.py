import agent
import os
from agent import Agent
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME") or "my-agent"
USER_NAME = os.getenv("USER_NAME") or "Anonymous"

agent = Agent(AGENT_NAME, USER_NAME)

# Creates Pinecone Index
agent.createIndex(AGENT_NAME)

print(agent.action(f"(I'm the system) GPT the user you will be chatting with is {USER_NAME}. Please offer friendly greeting and ask how you can help the user."))

while True:
    userInput = input()
    if userInput:
        if (userInput.startswith("read:")):
            agent.read(userInput[5:].strip(" "))
            print("Understood! The information is summarized and stored in my memory.")
        elif (userInput.startswith("think:")):
            agent.think(userInput[6:].strip(" "))
            print("Understood! I stored that thought into my memory.")
        else:
            print(agent.action(userInput), "\n")
    else:
        print("SYSTEM - Give a valid input")

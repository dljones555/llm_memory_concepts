from memory_stream_agent import MemoryStreamAgent

# initialize and load agent

agent = MemoryStreamAgent("topic_keywords.json")
conversation_id = "user123-session1"

# Simulate adding messages to the agent

messages = [
    "I'm flying to Berlin next week for a tech conference.",
    "The event will cover new programming languages and AI applications.",
    "I hope the weather isn't too cold.",
    "Also, I plan to try some German food while I'm there.",
    "After the trip, I need to prepare for a big project deadline at work."
]

for msg in messages:
    result = agent.add_message(msg, conversation_id=conversation_id)
    print(f"📥 Added message. Topic: {result['topic']}, Keywords: {result['keywords']}")

query = "What are the key AI trends from that tech conference?"
context = agent.get_relevant_context(query)

print("\n🔍 Relevant context:")
for item in context:
    print(f"- [{item['topic']}] {item['content']} (score: {item['score']})")

# retrieve context for new input

query = "What are the key AI trends from that tech conference?"
context = agent.get_relevant_context(query)

print("\n🔍 Relevant context:")
for item in context:
    print(f"- [{item['topic']}] {item['content']} (score: {item['score']})")

# check for topic drift

drift_notice = agent.emit_topic_drift_event()
if drift_notice:
    print(f"\n🚨 {drift_notice}")

# summarize conversation

summary = agent.get_conversation_summary(conversation_id)
print("\n📊 Conversation Summary:")
print(summary)

# split conversation by topic

old_id, new_id = agent.split_conversation(conversation_id, split_index=3)
print(f"\n🧩 Conversation split: {old_id} | {new_id}")

print("Old part summary:", agent.get_conversation_summary(old_id))
print("New part summary:", agent.get_conversation_summary(new_id))

# generate a gpt summary of the conversation

print("\n🤖 GPT Summary of full conversation:")
print(agent.create_openai_summary(old_id))  # Or use `new_id` as needed


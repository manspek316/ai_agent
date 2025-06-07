import lmstudio as lms
model = lms.llm("qwen3-0.6b")

for fragment in model.respond_stream("tell me in one line, what is the color of a lion"):
    print(fragment.content, end="", flush=True)
print() # Advance to a new line at the end of the response


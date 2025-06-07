import lmstudio as lms
model = lms.llm("qwen3-0.6b")
result = model.respond("\no_think count from one to 11?")
print(result)


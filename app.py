from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chain/c/N4XyA")
res = chain.invoke({ "language": "GEmany","text": "how to develop a llm" })
print(res)
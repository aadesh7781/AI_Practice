
MAX_CONTEXT = 8000

system_tokens = 500
history_tokens = 2000
retrieved_tokens = 2500
user_tokens = 200
output_tokens = 2000

used_tokens = (
    system_tokens
    + history_tokens
    + retrieved_tokens
    + user_tokens
    + output_tokens
)

remaining_tokens = MAX_CONTEXT - used_tokens

print("Maximum Context:", MAX_CONTEXT)
print("Used Tokens:", used_tokens)
print("Remaining Tokens:", remaining_tokens)

if remaining_tokens < 0:
    print("Context limit exceeded!")
else:
    print("Context fits within the budget.")

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Create a short story about a time-traveling scientist who accidentally alters history.",
]
# Create a sampling params object.
sampling_params = SamplingParams(n=8, best_of=32, presence_penalty=1.0, frequency_penalty=1.0, temperature=0.0, use_beam_search=True, early_stopping=True, max_tokens=256)

# Create an LLM.
llm = LLM(model="facebook/opt-13b")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    
    for generated_text in output.outputs:
        generated_text = generated_text.text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

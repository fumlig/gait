# todo

## misc

- simplify model services
- store models in library and import in services rather than redefining

- consider proper audio streaming with https://github.com/davidbrowne17/chatterbox-streaming
- separate provider protocols (only some providers need health checks and model listing)
- improve movel auto-downloading, listing and monitoring of progress
- add image models and image support to chat
- compare to vllm-omni (https://github.com/vllm-project/vllm-omni)
- improve voice cleaning script
- improve text preprocessing before voice
- upgrade gateway to python 3.14
- fix models dir env var for llamacpp (does it make sense to have?)
- remove models and voices directories from git?
- disable thinking through api
- add metadata to responses like audio sample rate
- move preprocessing to the provider level, so that it can be customized per provider
- need better thinking handling/per model logic for llama.cpp, how do we apply             extra_body={"chat_template_kwargs": {"enable_thinking": False}}? Maybe a model config on the client? Best approach might be model configs that are part of the provider
- should probably simplify - don't fake native voice models in chat, instead users should just use the voice provider manually while streaming
- if we really want to manually make it work we should do it as a new chat provider that merges the clients
- consider openresponses
- make sure we can configure models fully from the client side and possible remove serverside config of this stuff
- ```export LLAMA_CACHE="unsloth/Qwen3.5-35B-A3B-GGUF"
./llama.cpp/llama-cli \
    -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_XL \
    --ctx-size 16384 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00```
- try https://github.com/resemble-ai/resemble-enhance for voice prep
-

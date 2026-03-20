# todo

- Separate provider protocols (only some providers need health checks and model listing)
- Make auto-downloading and mounting of model weights more robust, add progress indication (should probably be possible to disable and track progress)
- Add image models and image support to chat
- Compare to vllm-omni (https://github.com/vllm-project/vllm-omni)
- Make sure chat, embeddings, completions, responses etc. have explicit models and protocols. Make sure tool calling functionality and all else works in integration tests (consider the OpenAI API docs https://developers.openai.com/api/docs and python client https://developers.openai.com/api/reference/python)
- Simplify voice cleaning script
- Move integration test to gateway

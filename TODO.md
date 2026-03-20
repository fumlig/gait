# todo

- Separate provider protocols (only some providers need health checks and model listing)
- Make auto-downloading and mounting of model weights more robust, add progress indication (should probably be possible to disable and track progress)
- Add image models and image support to chat
- Compare to vllm-omni (https://github.com/vllm-project/vllm-omni)
- Make sure chat, embeddings, completions, responses etc. have explicit models and protocols. Check tool calling functionality (and skills?) in integration tests.

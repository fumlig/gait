# todo

## misc

- Separate provider protocols (only some providers need health checks and model listing)
- Make auto-downloading and mounting of model weights more robust, add progress indication (should probably be possible to disable and track progress)
- Add image models and image support to chat
- Compare to vllm-omni (https://github.com/vllm-project/vllm-omni)
- Make sure chat, embeddings, completions, responses etc. have explicit models and protocols. Make sure tool calling functionality and all else works in integration tests (consider the OpenAI API docs https://developers.openai.com/api/docs and python client https://developers.openai.com/api/reference/python)
- Simplify voice cleaning script
- Preprocess text before voice (remove markdown etc)

## refactor

- [x] 1. Eliminate `_get_*_client` / try-except boilerplate in routes via FastAPI `Depends()`
- [ ] 2. Split `models.py` (612 lines) into domain modules
- [ ] 3. Lift proxy transport helpers (`_forward`, `_forward_stream`, `_stream_raw`) from `LlamacppClient` into `BaseProvider`
- [ ] 4. Extract shared engine idle-timeout logic into a base class (services)
- [ ] 5. Service tests: import the real app instead of rebuilding it

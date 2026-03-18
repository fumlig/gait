# todo

- Implement transcription using WhisperX
- ~~Reuse settings schema base~~ (done — `libs/trave-common/`)
- Lazy load models (don't load until first use) and unload after a configurable timeout. Put logic in shared library.
- Run locally without docker using venvs (also to support linting)
- Better READMEs. Less fluff, clear disclaimer about using agents.
- Create API docs (maybe in using FastAPI gateway) that follow OpenAI API structure. If possible we can maybe merge 
- Make sure volumes and other things that may be used by several services are put in the root
- ~~Extend chatterbox to more models~~ (done — turbo, original, multilingual)
- Integration tests using OpenAI API client
- Make sure docker builds are as fast and small as possible by utilizing caching layers

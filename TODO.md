# todo

- Rename trave-common to just common.
- Consider putting models behind simple interfaces, and rename services to generic names. Each service holds a list of models. Then put the loading and unloading logic on top of the interface.
- Lazy load models (don't load until first use) and unload after a configurable timeout. Put logic in shared library.
- Run locally without docker using venvs (also to support linting)
- Better READMEs. Less fluff, clear disclaimer about using agents.
- Create API docs (maybe in using FastAPI gateway) that follow OpenAI API structure. If possible we can maybe merge 
- Make sure volumes and other things that may be used by several services are put in the root
- Integration tests using OpenAI API client
- Make sure docker builds are as fast and small as possible by utilizing caching layers

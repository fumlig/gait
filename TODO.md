# todo

- Rename trave-common to just common.
- Consider putting models behind simple interfaces, and rename services to generic names. Each service holds a list of models. Then put the loading and unloading logic on top of the interface.
- Lazy load models (don't load until first use) and unload after a configurable timeout. Put logic in shared library.
- Run locally without docker using venvs (also to support linting)
- Better READMEs. Less fluff, clear disclaimer about using agents.
- Create API docs (maybe in using FastAPI gateway) that follow OpenAI API structure. If possible we can maybe merge 
- Make sure volumes and other things that may be used by several services are put in the root
- Integration tests using OpenAI API client and validation of outputs using real services
- Make sure docker builds are as fast and small as possible by utilizing caching layers

- Make backend services even thinner and have them accept kwargs over http. THe gateway is configured with all of it so that it can be overriden per request as well!




- Make model services even slimmer - almost just RPC/remote stubs that the backend protocols in the gateways call. Can we use some simple RPC scheme here?
- Turn the voice backend into its own service (that is just a file store). Provide access to the same volume to the chatterbox service.
- Add `ty` for typechecking
- Implement language model (Qwen3.5)

- Script to clean up voice sample and prepare it for use using ffmpeg (remove silence, normalize volume, filter out background noise,etc.)

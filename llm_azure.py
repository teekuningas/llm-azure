import os
from typing import Iterable, Iterator, List, Union

import click
import llm
import yaml
from llm import EmbeddingModel, hookimpl
from llm.default_plugins.openai_models import AsyncChat, Chat, _Shared
from openai import AsyncAzureOpenAI, AzureOpenAI

DEFAULT_KEY_ALIAS = "azure"
DEFAULT_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"


def _ensure_config_file():
    filepath = llm.user_dir() / "azure" / "config.yaml"
    if not filepath.exists():
        filepath.parent.mkdir(exist_ok=True)
        filepath.write_text("[]")
    return filepath


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def azure():
        "Commands for working with azure models"

    @azure.command()
    def config_file():
        "Display the path to the azure config file"
        click.echo(_ensure_config_file())


@hookimpl
def register_models(register):
    azure_path = _ensure_config_file()
    with open(azure_path) as f:
        azure_models = yaml.safe_load(f)

    for model in azure_models or []:
        if model.get('embedding_model'):
            continue

        # Use .pop() to remove these keys from the model dictionary
        # before it's unpacked with **model
        needs_key = model.pop("needs_key", DEFAULT_KEY_ALIAS)
        key_env_var = model.pop("key_env_var", DEFAULT_KEY_ENV_VAR)
        # The 'aliases' key is popped here to prevent it from being passed as a model parameter.
        # Other new parameters (vision, audio, reasoning, allows_system_prompt) are left in
        # the 'model' dictionary to be passed via **model to the AzureChat constructor.
        aliases = model.pop("aliases", [])

        # Instantiate the models before registering them
        chat_model = AzureChat(needs_key=needs_key, key_env_var=key_env_var, **model)
        async_chat_model = AzureAsyncChat(needs_key=needs_key, key_env_var=key_env_var, **model)

        # Pass all relevant model parameters directly to AzureChat/AzureAsyncChat
        # This assumes the model dict contains 'model_id', 'model_name', 'api_base', 'api_version'
        # and optionally 'attachment_types', 'can_stream', and the new vision, audio, reasoning,
        # and allows_system_prompt parameters.
        register(
            chat_model,
            async_chat_model,
            aliases=aliases,
        )


@hookimpl
def register_embedding_models(register):
    azure_path = _ensure_config_file()
    with open(azure_path) as f:
        azure_models = yaml.safe_load(f)

    for model in azure_models or []:
        if not model.get('embedding_model'):
            continue

        needs_key = model.pop("needs_key", DEFAULT_KEY_ALIAS)
        key_env_var = model.pop("key_env_var", DEFAULT_KEY_ENV_VAR)
        aliases = model.pop("aliases", [])
        model.pop('embedding_model') # Remove the flag before passing to constructor

        register(
            AzureEmbedding(needs_key=needs_key, key_env_var=key_env_var, **model),
            aliases=aliases,
        )


class AzureShared(_Shared):
    def __init__(self, model_id, model_name, api_base, api_version,
                 attachment_types=None, can_stream=True,
                 needs_key: str = DEFAULT_KEY_ALIAS,
                 key_env_var: str = DEFAULT_KEY_ENV_VAR,
                 vision=False, audio=False, reasoning=False,
                 allows_system_prompt=True, **kwargs):
        self.attachment_types = attachment_types or set()
        self.needs_key = needs_key
        self.key_env_var = key_env_var
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            can_stream=can_stream,
            vision=vision,
            audio=audio,
            reasoning=reasoning,
            allows_system_prompt=allows_system_prompt,
            api_base=api_base,
            api_version=api_version,
            **kwargs
        )

    def get_client(self, key, *, async_=False):
        kwargs = {
            "api_key": self.get_key(key),
            "api_version": self.api_version,
            "azure_endpoint": self.api_base,
        }
        if os.environ.get("LLM_OPENAI_SHOW_RESPONSES"):
            kwargs["http_client"] = self.logging_client()
        if async_:
            return AsyncAzureOpenAI(**kwargs)
        else:
            return AzureOpenAI(**kwargs)

class AzureChat(AzureShared, Chat):
    pass

class AzureAsyncChat(AzureShared, AsyncChat):
    pass

class AzureEmbedding(EmbeddingModel):
    batch_size = 100

    def __init__(self, model_id, model_name, api_base, api_version, needs_key: str = DEFAULT_KEY_ALIAS, key_env_var: str = DEFAULT_KEY_ENV_VAR, **kwargs):
        super().__init__(model_id=model_id, model_name=model_name, needs_key=needs_key, key_env_var=key_env_var, **kwargs)
        self.api_base = api_base
        self.api_version = api_version

    def embed_batch(self, items: Iterable[Union[str, bytes]]) -> Iterator[List[float]]:
        client = AzureOpenAI(
            api_key=self.get_key(),
            api_version=self.api_version,
            azure_endpoint=self.api_base,
        )
        kwargs = {
            "input": items,
            "model": self.model_name,
        }
        results = client.embeddings.create(**kwargs).data
        return ([float(r) for r in result.embedding] for result in results)

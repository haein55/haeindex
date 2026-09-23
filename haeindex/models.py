"""Create role-specific AWS Bedrock clients."""

from haeindex.bedrock import Bedrock, bedrock_model
from haeindex.bedrock_responses import BedrockResponses


def model_client(role: str, *, model: str | None = None, num_ctx: int = 16384):
    model_id = bedrock_model(role, model)
    client = (
        BedrockResponses if "openai.gpt-" in model_id and "gpt-oss-" not in model_id else Bedrock
    )
    return client(model_id, num_ctx=num_ctx)

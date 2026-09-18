"""Create role-specific AWS Bedrock clients."""

from haeindex.bedrock import Bedrock, bedrock_model


def model_client(role: str, *, model: str | None = None, num_ctx: int = 16384):
    return Bedrock(bedrock_model(role, model), num_ctx=num_ctx)

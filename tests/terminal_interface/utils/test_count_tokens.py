from unittest import mock

from interpreter.terminal_interface.utils import count_tokens as ct


def test_count_tokens_strips_model_prefix():
    mock_encoder = mock.Mock()
    mock_encoder.encode.return_value = [1, 2, 3]
    with mock.patch.object(ct, "tiktoken") as mock_tiktoken:
        mock_tiktoken.encoding_for_model.return_value = mock_encoder
        result = ct.count_tokens("hello", model="openai/gpt-4")
    assert result == 3
    mock_tiktoken.encoding_for_model.assert_called_with("gpt-4")


def test_count_tokens_falls_back_to_gpt4_encoder():
    mock_encoder = mock.Mock()
    mock_encoder.encode.return_value = [1]
    with mock.patch.object(ct, "tiktoken") as mock_tiktoken:
        mock_tiktoken.encoding_for_model.side_effect = [KeyError("unknown"), mock_encoder]
        result = ct.count_tokens("hello", model="unknown-model")
    assert result == 1


def test_count_tokens_returns_zero_on_failure():
    with mock.patch.object(ct, "tiktoken", side_effect=ImportError):
        assert ct.count_tokens("hello") == 0


def test_token_cost_rounds_result():
    with mock.patch.object(ct, "cost_per_token", return_value=(0.1234567, 0)):
        assert ct.token_cost(tokens=100, model="gpt-4") == 0.123457


def test_count_messages_tokens_sums_message_fields():
    with mock.patch.object(ct, "count_tokens", side_effect=[2, 3, 4, 5]):
        with mock.patch.object(ct, "token_cost", return_value=0.01):
            tokens, cost = ct.count_messages_tokens(
                messages=[
                    "plain",
                    {"message": "msg", "code": "code", "output": "out"},
                ],
                model="gpt-4",
            )
    assert tokens == 14
    assert cost == 0.01

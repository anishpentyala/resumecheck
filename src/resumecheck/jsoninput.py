"""Unambiguous raw JSON objects for recorded verification evidence."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            # Keys and values can contain private metadata or training content.
            raise ValueError('duplicate JSON object member in verification evidence')
        result[key] = value
    return result


def unambiguous_dumps(value, **kwargs):
    """Reject duplicate wire members created by JSON's object-key coercion.

    Validate the serialized form before any caller opens an output file. This
    preserves supported non-string keys when they do not collide, and the
    existing NaN/infinity value policy. It is not general lossless serialization.
    """
    import json

    text = json.dumps(value, **kwargs)
    json.loads(text, object_pairs_hook=unique_object)
    return text

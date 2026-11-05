"""Unambiguous raw JSON objects for recorded verification evidence."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            # Keys and values can contain private metadata or training content.
            raise ValueError('duplicate JSON object member in verification evidence')
        result[key] = value
    return result
